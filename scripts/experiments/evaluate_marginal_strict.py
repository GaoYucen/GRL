from __future__ import annotations

"""Strict NetHEPT marginal-gain validation.

This follow-up removes repeated seed sets across train/validation/test and adds
seed-mask ablations plus an overlap stress test. Run from repository root:

    python scripts/experiments/evaluate_marginal_strict.py
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from evaluate_marginal_predictability import (
    BASE_SEED,
    build_edge_arrays,
    conditional_metrics,
    label_state,
    materialize,
    set_seed,
    tensorize,
)

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from grl.baselines import select_degree_discount_nodes, select_high_degree_nodes
from grl.data import load_graph_from_config
from grl.models import MarginalGainPredictor, build_node_features, load_or_create_node2vec_embeddings

OUT = ROOT / "outputs" / "marginal_predictability" / "nethept_strict"
OUT.mkdir(parents=True, exist_ok=True)


def make_unique_states(graph_data, count, candidate_count, seed, forbidden=None):
    graph = graph_data.graph
    nodes = list(range(graph_data.num_nodes))
    rng = random.Random(seed)
    degree_rank = select_high_degree_nodes(graph, graph_data.num_nodes)
    dd_rank = select_degree_discount_nodes(graph, min(graph_data.num_nodes, 1000), 0.01)
    top_pool = degree_rank[: min(1500, len(degree_rank))]
    forbidden = set() if forbidden is None else set(forbidden)
    seen = set(forbidden)
    rows = []
    attempts = 0

    while len(rows) < count and attempts < count * 100:
        attempts += 1
        size = rng.randint(1, 9)
        mode = rng.randrange(4)
        if mode == 0:
            seeds = rng.sample(nodes, size)
        elif mode == 1:
            seeds = rng.sample(top_pool, size)
        elif mode == 2:
            hi = rng.sample(top_pool, max(1, size // 2))
            rest_pool = [n for n in nodes if n not in hi]
            seeds = hi + rng.sample(rest_pool, size - len(hi))
        else:
            base = degree_rank[: max(1, size - 1)]
            extra_pool = [n for n in top_pool if n not in base]
            seeds = list(base) + rng.sample(extra_pool, size - len(base))

        key = tuple(sorted(seeds))
        if key in seen:
            continue
        seen.add(key)

        selected = set(seeds)
        candidates = []
        for node in degree_rank:
            if node not in selected:
                candidates.append(node)
            if len(candidates) >= 6:
                break
        for node in dd_rank:
            if node not in selected and node not in candidates:
                candidates.append(node)
            if len(candidates) >= 10:
                break

        high_avail = [n for n in top_pool if n not in selected and n not in candidates]
        rng.shuffle(high_avail)
        candidates.extend(high_avail[: min(5, candidate_count - len(candidates))])
        available = [n for n in nodes if n not in selected and n not in candidates]
        rng.shuffle(available)
        candidates.extend(available[: max(0, candidate_count - len(candidates))])
        rows.append({"seeds": list(seeds), "candidates": candidates[:candidate_count]})

    if len(rows) != count:
        raise RuntimeError(f"Could only generate {len(rows)} unique states")
    return rows, seen


def predict_rows_mask_mode(model, embeddings, norm_degrees, rows, num_nodes, device, mode):
    out = []
    model.eval()
    with torch.no_grad():
        for i, row in enumerate(rows):
            mask = torch.zeros((num_nodes, 1), dtype=torch.float32, device=device)
            if mode == "actual":
                seeds = row["seeds"]
            elif mode == "zero":
                seeds = []
            elif mode == "shuffled":
                seeds = rows[(i + 7) % len(rows)]["seeds"]
            else:
                raise ValueError(mode)
            if seeds:
                mask[seeds, 0] = 1.0
            out.append([
                float(model(embeddings, norm_degrees, mask, cand).item())
                for cand in row["candidates"]
            ])
    return out


def overlap_stress(model, embeddings, norm_degrees, graph_data, edge_arrays, device, mc_runs=200, pairs=30):
    graph = graph_data.graph
    rng = random.Random(BASE_SEED + 8888)
    degree_rank = select_high_degree_nodes(graph, graph_data.num_nodes)
    candidates = [v for v in degree_rank if graph.out_degree(v) >= 8][:pairs]
    nodes = list(range(graph_data.num_nodes))
    records = []

    for i, cand in enumerate(candidates):
        nbrs = list(graph.successors(cand))
        rng.shuffle(nbrs)
        overlap = nbrs[:8]
        if len(overlap) < 8:
            fill = [n for n in nodes if n != cand and n not in overlap]
            overlap += rng.sample(fill, 8 - len(overlap))

        exclude = set(overlap) | {cand} | set(nbrs)
        far_pool = [n for n in nodes if n not in exclude]
        random_seeds = rng.sample(far_pool, 8)
        world_seed = BASE_SEED + 700000 + i * 1009

        y_far = float(label_state(*edge_arrays, graph.is_directed(), random_seeds, [cand], mc_runs, world_seed)[0])
        y_overlap = float(label_state(*edge_arrays, graph.is_directed(), overlap, [cand], mc_runs, world_seed)[0])

        def pred(seeds):
            mask = torch.zeros((graph_data.num_nodes, 1), dtype=torch.float32, device=device)
            mask[seeds, 0] = 1.0
            with torch.no_grad():
                return float(model(embeddings, norm_degrees, mask, cand).item())

        p_far = pred(random_seeds)
        p_overlap = pred(overlap)
        records.append({
            "candidate": cand,
            "true_far": y_far,
            "true_overlap": y_overlap,
            "pred_far": p_far,
            "pred_overlap": p_overlap,
            "true_drop": y_far - y_overlap,
            "pred_drop": p_far - p_overlap,
        })

    yt = np.asarray([r["true_drop"] for r in records], dtype=float)
    yp = np.asarray([r["pred_drop"] for r in records], dtype=float)
    sign_mask = np.abs(yt) > 1e-9
    corr = float(np.corrcoef(yt, yp)[0, 1]) if np.std(yt) > 0 and np.std(yp) > 0 else 0.0
    return {
        "pairs": len(records),
        "true_drop_mean": float(np.mean(yt)),
        "true_drop_std": float(np.std(yt)),
        "pred_drop_mean": float(np.mean(yp)),
        "drop_mae": float(np.mean(np.abs(yp - yt))),
        "drop_pearson": corr,
        "direction_accuracy": float(np.mean(np.sign(yp[sign_mask]) == np.sign(yt[sign_mask]))) if np.any(sign_mask) else 0.0,
        "records": records,
    }


def main():
    set_seed(BASE_SEED + 500)
    config = yaml.safe_load((ROOT / "configs" / "gnn_nethept.yaml").read_text())
    graph_data = load_graph_from_config(config)
    graph = graph_data.graph
    weights = np.asarray([float(d.get("weight", 0.0)) for _, _, d in graph.edges(data=True)])
    print(
        f"graph nodes={graph_data.num_nodes} edges={graph_data.num_edges} "
        f"weight_mean={weights.mean():.6f} min={weights.min():.6f} max={weights.max():.6f}",
        flush=True,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cache = ROOT / "outputs" / "marginal_predictability" / "nethept_grouped" / "model" / "marginal_node2vec_nethept.pth"
    if not cache.exists():
        cache = OUT / "marginal_node2vec_nethept.pth"

    embeddings = load_or_create_node2vec_embeddings(
        graph,
        cache,
        dimensions=64,
        walk_length=10,
        num_walks=4,
        window=5,
        workers=2,
        quiet=True,
    ).to(device)
    norm_degrees, _ = build_node_features(graph, device=device)
    edge_arrays = build_edge_arrays(graph)

    train_states, train_keys = make_unique_states(graph_data, 100, 20, BASE_SEED + 501)
    val_states, val_keys = make_unique_states(graph_data, 24, 20, BASE_SEED + 502, forbidden=train_keys)
    test_states, test_keys = make_unique_states(graph_data, 40, 24, BASE_SEED + 503, forbidden=train_keys | val_keys)
    assert not (train_keys & val_keys)
    assert not (train_keys & test_keys)
    assert not (val_keys & test_keys)

    print("strict: labeling train", flush=True)
    train_rows = materialize(train_states, edge_arrays, graph.is_directed(), mc_runs=15, seed_base=BASE_SEED + 510000)
    print("strict: labeling val", flush=True)
    val_rows = materialize(val_states, edge_arrays, graph.is_directed(), mc_runs=30, seed_base=BASE_SEED + 520000)
    print("strict: labeling test", flush=True)
    test_rows = materialize(test_states, edge_arrays, graph.is_directed(), mc_runs=100, seed_base=BASE_SEED + 530000)

    train_masks, train_cands, train_labels, _ = tensorize(train_rows, graph_data.num_nodes, device)
    val_masks, val_cands, val_labels, _ = tensorize(val_rows, graph_data.num_nodes, device)

    set_seed(BASE_SEED + 777)
    model = MarginalGainPredictor(embeddings.shape[1], hidden_dim=96).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_state = None
    best_val = float("inf")
    batch_size = 64
    n = train_labels.shape[0]
    history = []

    for epoch in range(60):
        model.train()
        order = torch.randperm(n, device=device)
        loss_sum = 0.0
        batches = 0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            b = len(idx)
            pred = model(
                embeddings.unsqueeze(0).expand(b, -1, -1),
                norm_degrees.unsqueeze(0).expand(b, -1, -1),
                train_masks[idx],
                train_cands[idx],
            )
            loss = F.mse_loss(pred, train_labels[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            loss_sum += float(loss.item())
            batches += 1

        model.eval()
        vp = []
        with torch.no_grad():
            for start in range(0, len(val_labels), batch_size):
                sl = slice(start, min(start + batch_size, len(val_labels)))
                b = len(val_labels[sl])
                vp.append(model(
                    embeddings.unsqueeze(0).expand(b, -1, -1),
                    norm_degrees.unsqueeze(0).expand(b, -1, -1),
                    val_masks[sl],
                    val_cands[sl],
                ))
        vp = torch.cat(vp)
        vrmse = float(torch.sqrt(F.mse_loss(vp, val_labels)).item())
        history.append({"epoch": epoch + 1, "train_mse": loss_sum / max(1, batches), "val_rmse": vrmse})
        if vrmse < best_val:
            best_val = vrmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0:
            print(f"strict epoch={epoch + 1} train_mse={history[-1]['train_mse']:.4f} val_rmse={vrmse:.4f}", flush=True)

    model.load_state_dict(best_state)
    actual_pred = predict_rows_mask_mode(model, embeddings, norm_degrees, test_rows, graph_data.num_nodes, device, "actual")
    zero_pred = predict_rows_mask_mode(model, embeddings, norm_degrees, test_rows, graph_data.num_nodes, device, "zero")
    shuffled_pred = predict_rows_mask_mode(model, embeddings, norm_degrees, test_rows, graph_data.num_nodes, device, "shuffled")

    actual = conditional_metrics(test_rows, actual_pred, graph)
    zero = conditional_metrics(test_rows, zero_pred, graph)
    shuffled = conditional_metrics(test_rows, shuffled_pred, graph)

    print("STRICT_ACTUAL", json.dumps({k: v for k, v in actual.items() if k != "by_seed_size"}), flush=True)
    print("STRICT_ZERO_MASK", json.dumps({k: v for k, v in zero.items() if k != "by_seed_size"}), flush=True)
    print("STRICT_SHUFFLED_MASK", json.dumps({k: v for k, v in shuffled.items() if k != "by_seed_size"}), flush=True)

    stress = overlap_stress(model, embeddings, norm_degrees, graph_data, edge_arrays, device, mc_runs=200, pairs=30)
    print("OVERLAP_STRESS", json.dumps({k: v for k, v in stress.items() if k != "records"}), flush=True)

    report = {
        "dataset": "NetHEPT",
        "graph": {
            "nodes": graph_data.num_nodes,
            "edges": graph_data.num_edges,
            "weight_mean": float(weights.mean()),
            "weight_min": float(weights.min()),
            "weight_max": float(weights.max()),
        },
        "device": str(device),
        "protocol": {
            "strict_disjoint_seed_sets": True,
            "train_states": len(train_rows),
            "train_candidates": 20,
            "train_mc": 15,
            "val_states": len(val_rows),
            "val_candidates": 20,
            "val_mc": 30,
            "test_states": len(test_rows),
            "test_candidates": 24,
            "test_mc": 100,
            "best_val_rmse": best_val,
        },
        "actual_seed_mask": actual,
        "zero_seed_mask": zero,
        "shuffled_seed_mask": shuffled,
        "overlap_stress": stress,
        "history": history,
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "best_val_rmse": best_val}, OUT / "model.pt")
    print(f"saved {OUT / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
