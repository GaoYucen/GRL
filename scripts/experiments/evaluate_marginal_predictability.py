from __future__ import annotations

"""Grouped conditional marginal-gain predictability experiment.

This script is the reproducible version of the first NetHEPT validation run.
It evaluates predictions *within the same seed set S* instead of pooling
ranking metrics across unrelated states.

Important: this first protocol intentionally mixes random and deterministic
seed-state generators. It provides a strong positive signal, but deterministic
prefix states can repeat across train/validation/test. Use
``evaluate_marginal_strict.py`` for the leakage-resistant follow-up.

Run from the repository root:
    python scripts/experiments/evaluate_marginal_predictability.py
"""

import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import pearsonr, spearmanr

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from grl.baselines import select_degree_discount_nodes, select_high_degree_nodes
from grl.data import load_graph_from_config
from grl.models import MarginalGainPredictor, build_node_features, load_or_create_node2vec_embeddings


BASE_SEED = 20260901
OUT = ROOT / "outputs" / "marginal_predictability" / "nethept_grouped"
OUT.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_states(graph_data, count: int, candidate_count: int, seed: int):
    graph = graph_data.graph
    nodes = list(range(graph_data.num_nodes))
    budget = 10
    rng = random.Random(seed)
    degree_rank = select_high_degree_nodes(graph, graph_data.num_nodes)
    dd_rank = select_degree_discount_nodes(graph, min(graph_data.num_nodes, 500), 0.01)
    top_pool = degree_rank[: min(500, len(degree_rank))]
    states = []

    for idx in range(count):
        size = idx % budget
        branch = idx % 4
        if size == 0:
            seeds = []
        elif branch == 0:
            seeds = rng.sample(nodes, size)
        elif branch == 1:
            seeds = degree_rank[:size]
        elif branch == 2:
            seeds = dd_rank[:size]
        else:
            seeds = rng.sample(top_pool, min(size, len(top_pool)))

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
        available = [n for n in nodes if n not in selected and n not in candidates]
        rng.shuffle(available)
        candidates.extend(available[: max(0, candidate_count - len(candidates))])
        states.append({"seeds": list(seeds), "candidates": candidates[:candidate_count]})
    return states


def build_edge_arrays(graph):
    us, vs, ps = [], [], []
    for u, v, data in graph.edges(data=True):
        us.append(int(u))
        vs.append(int(v))
        ps.append(float(data.get("weight", 0.0)))
    return np.asarray(us, dtype=np.int32), np.asarray(vs, dtype=np.int32), np.asarray(ps, dtype=np.float64)


def reach(adj, starts):
    seen = set(starts)
    stack = list(starts)
    while stack:
        u = stack.pop()
        for v in adj.get(u, ()):
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


def label_state(edge_u, edge_v, edge_p, directed, seeds, candidates, mc_runs: int, seed: int):
    """Label all candidates for one S using common live-edge worlds."""
    sums = np.zeros(len(candidates), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for _ in range(mc_runs):
        mask = rng.random(edge_p.shape[0]) < edge_p
        adj = defaultdict(list)
        live_u = edge_u[mask]
        live_v = edge_v[mask]
        for u, v in zip(live_u.tolist(), live_v.tolist()):
            adj[u].append(v)
            if not directed:
                adj[v].append(u)
        base = reach(adj, seeds)
        for j, cand in enumerate(candidates):
            cand_reach = reach(adj, [cand])
            sums[j] += len(cand_reach - base)
    return (sums / mc_runs).astype(np.float32)


def materialize(states, edge_arrays, directed, mc_runs: int, seed_base: int):
    rows = []
    t0 = time.perf_counter()
    for i, state in enumerate(states):
        labels = label_state(
            *edge_arrays,
            directed,
            state["seeds"],
            state["candidates"],
            mc_runs,
            seed_base + 1009 * i,
        )
        rows.append({"seeds": state["seeds"], "candidates": state["candidates"], "labels": labels.tolist()})
        if (i + 1) % 10 == 0 or i + 1 == len(states):
            print(f"labeled {i + 1}/{len(states)} states (mc={mc_runs}) elapsed={time.perf_counter() - t0:.1f}s", flush=True)
    return rows


def tensorize(rows, num_nodes, device):
    masks, candidates, labels, state_ids = [], [], [], []
    for sid, row in enumerate(rows):
        for cand, label in zip(row["candidates"], row["labels"]):
            mask = torch.zeros((num_nodes, 1), dtype=torch.float32)
            if row["seeds"]:
                mask[row["seeds"], 0] = 1.0
            masks.append(mask)
            candidates.append(cand)
            labels.append(label)
            state_ids.append(sid)
    return (
        torch.stack(masks).to(device),
        torch.tensor(candidates, dtype=torch.long, device=device),
        torch.tensor(labels, dtype=torch.float32, device=device).unsqueeze(-1),
        state_ids,
    )


def predict_rows(model, embeddings, norm_degrees, rows, num_nodes, device):
    predictions = []
    model.eval()
    with torch.no_grad():
        for row in rows:
            mask = torch.zeros((num_nodes, 1), dtype=torch.float32, device=device)
            if row["seeds"]:
                mask[row["seeds"], 0] = 1.0
            predictions.append([
                float(model(embeddings, norm_degrees, mask, cand).item())
                for cand in row["candidates"]
            ])
    return predictions


def conditional_metrics(rows, predictions, graph):
    all_y, all_p = [], []
    state_spearman = []
    pairwise_scores = []
    top1_hits = []
    top3_recalls = []
    gain_ratios = []
    regrets = []
    degree_ratios = []
    per_size = defaultdict(lambda: {"y": [], "p": []})

    degrees = dict(graph.out_degree() if graph.is_directed() else graph.degree())

    for row, pred in zip(rows, predictions):
        y = np.asarray(row["labels"], dtype=float)
        p = np.asarray(pred, dtype=float)
        all_y.extend(y.tolist())
        all_p.extend(p.tolist())
        per_size[len(row["seeds"])]["y"].extend(y.tolist())
        per_size[len(row["seeds"])]["p"].extend(p.tolist())

        rho = spearmanr(y, p).statistic if np.std(y) > 0 and np.std(p) > 0 else 0.0
        if not np.isfinite(rho):
            rho = 0.0
        state_spearman.append(float(rho))

        correct = total = 0
        for i in range(len(y)):
            for j in range(i + 1, len(y)):
                dy = y[i] - y[j]
                if abs(dy) < 1e-12:
                    continue
                dp = p[i] - p[j]
                total += 1
                if dp * dy > 0:
                    correct += 1
                elif abs(dp) < 1e-12:
                    correct += 0.5
        pairwise_scores.append(correct / total if total else 0.0)

        true_best = int(np.argmax(y))
        pred_best = int(np.argmax(p))
        top1_hits.append(float(pred_best == true_best))
        true_top3 = set(np.argsort(y)[-min(3, len(y)):].tolist())
        pred_top3 = set(np.argsort(p)[-min(3, len(p)):].tolist())
        top3_recalls.append(len(true_top3 & pred_top3) / len(true_top3))

        oracle_gain = float(y[true_best])
        chosen_gain = float(y[pred_best])
        gain_ratios.append(chosen_gain / oracle_gain if oracle_gain > 1e-12 else 1.0)
        regrets.append(oracle_gain - chosen_gain)

        degree_best = max(range(len(row["candidates"])), key=lambda k: degrees.get(row["candidates"][k], 0))
        degree_gain = float(y[degree_best])
        degree_ratios.append(degree_gain / oracle_gain if oracle_gain > 1e-12 else 1.0)

    y = np.asarray(all_y, dtype=float)
    p = np.asarray(all_p, dtype=float)
    errors = p - y
    pearson = pearsonr(y, p).statistic if np.std(y) > 0 and np.std(p) > 0 else 0.0
    global_spearman = spearmanr(y, p).statistic if np.std(y) > 0 and np.std(p) > 0 else 0.0

    by_seed_size = {}
    for size, vals in sorted(per_size.items()):
        yy = np.asarray(vals["y"], dtype=float)
        pp = np.asarray(vals["p"], dtype=float)
        by_seed_size[str(size)] = {
            "count": int(len(yy)),
            "mae": float(np.mean(np.abs(pp - yy))),
            "rmse": float(np.sqrt(np.mean((pp - yy) ** 2))),
        }

    return {
        "n_pairs": int(len(y)),
        "label_mean": float(np.mean(y)),
        "label_std": float(np.std(y)),
        "label_min": float(np.min(y)),
        "label_max": float(np.max(y)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "pearson": float(pearson),
        "global_spearman": float(global_spearman),
        "mean_conditional_spearman": float(np.mean(state_spearman)),
        "mean_pairwise_accuracy": float(np.mean(pairwise_scores)),
        "top1_accuracy": float(np.mean(top1_hits)),
        "top3_recall": float(np.mean(top3_recalls)),
        "mean_selected_gain_ratio": float(np.mean(gain_ratios)),
        "mean_regret": float(np.mean(regrets)),
        "degree_mean_gain_ratio": float(np.mean(degree_ratios)),
        "by_seed_size": by_seed_size,
    }


def main():
    set_seed(BASE_SEED)
    config = yaml.safe_load((ROOT / "configs" / "gnn_nethept.yaml").read_text())
    graph_data = load_graph_from_config(config)
    graph = graph_data.graph
    print(f"graph nodes={graph_data.num_nodes} edges={graph_data.num_edges}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = OUT / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = model_dir / "marginal_node2vec_nethept.pth"

    embedding_backend = "node2vec"
    try:
        embeddings = load_or_create_node2vec_embeddings(
            graph,
            embedding_path,
            dimensions=64,
            walk_length=10,
            num_walks=4,
            window=5,
            workers=2,
            quiet=True,
        )
    except Exception as exc:
        print(f"node2vec failed, using structural fallback: {exc!r}", flush=True)
        from grl.models.gnn import _build_fallback_embeddings

        embeddings = _build_fallback_embeddings(graph)
        embeddings = F.normalize(embeddings.float(), p=2, dim=1)
        embedding_backend = "structural_fallback"
    embeddings = embeddings.to(device)
    norm_degrees, _ = build_node_features(graph, device=device)

    edge_arrays = build_edge_arrays(graph)
    train_states = make_states(graph_data, count=60, candidate_count=16, seed=BASE_SEED + 1)
    val_states = make_states(graph_data, count=12, candidate_count=16, seed=BASE_SEED + 2)
    test_states = make_states(graph_data, count=20, candidate_count=16, seed=BASE_SEED + 3)

    print("building grouped training labels", flush=True)
    train_rows = materialize(train_states, edge_arrays, graph.is_directed(), mc_runs=10, seed_base=BASE_SEED + 10000)
    print("building grouped validation labels", flush=True)
    val_rows = materialize(val_states, edge_arrays, graph.is_directed(), mc_runs=20, seed_base=BASE_SEED + 20000)
    print("building grouped test labels", flush=True)
    test_rows = materialize(test_states, edge_arrays, graph.is_directed(), mc_runs=40, seed_base=BASE_SEED + 30000)

    train_masks, train_cands, train_labels, _ = tensorize(train_rows, graph_data.num_nodes, device)
    val_masks, val_cands, val_labels, _ = tensorize(val_rows, graph_data.num_nodes, device)

    results = []
    for model_seed in [11, 22, 33]:
        set_seed(BASE_SEED + model_seed)
        model = MarginalGainPredictor(embeddings.shape[1], hidden_dim=96).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        best_state = None
        best_val = math.inf
        batch_size = 64
        n = train_labels.shape[0]
        history = []

        for epoch in range(60):
            model.train()
            order = torch.randperm(n, device=device)
            total_loss = 0.0
            batches = 0
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                b = idx.shape[0]
                emb = embeddings.unsqueeze(0).expand(b, -1, -1)
                deg = norm_degrees.unsqueeze(0).expand(b, -1, -1)
                pred = model(emb, deg, train_masks[idx], train_cands[idx])
                loss = F.mse_loss(pred, train_labels[idx])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total_loss += float(loss.item())
                batches += 1

            model.eval()
            with torch.no_grad():
                vp = []
                for start in range(0, val_labels.shape[0], batch_size):
                    sl = slice(start, min(start + batch_size, val_labels.shape[0]))
                    b = val_labels[sl].shape[0]
                    emb = embeddings.unsqueeze(0).expand(b, -1, -1)
                    deg = norm_degrees.unsqueeze(0).expand(b, -1, -1)
                    vp.append(model(emb, deg, val_masks[sl], val_cands[sl]))
                vp = torch.cat(vp, dim=0)
                val_rmse = float(torch.sqrt(F.mse_loss(vp, val_labels)).item())
            history.append({"epoch": epoch + 1, "train_mse": total_loss / max(batches, 1), "val_rmse": val_rmse})
            if val_rmse < best_val:
                best_val = val_rmse
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if (epoch + 1) % 10 == 0:
                print(f"seed={model_seed} epoch={epoch + 1} train_mse={history[-1]['train_mse']:.4f} val_rmse={val_rmse:.4f}", flush=True)

        model.load_state_dict(best_state)
        predictions = predict_rows(model, embeddings, norm_degrees, test_rows, graph_data.num_nodes, device)
        metrics = conditional_metrics(test_rows, predictions, graph)
        metrics["model_seed"] = model_seed
        metrics["best_val_rmse"] = best_val
        metrics["history"] = history
        results.append(metrics)
        print("TEST", json.dumps({k: v for k, v in metrics.items() if k not in ("history", "by_seed_size")}, ensure_ascii=False), flush=True)

    scalar_keys = [
        "mae",
        "rmse",
        "pearson",
        "global_spearman",
        "mean_conditional_spearman",
        "mean_pairwise_accuracy",
        "top1_accuracy",
        "top3_recall",
        "mean_selected_gain_ratio",
        "mean_regret",
        "degree_mean_gain_ratio",
    ]
    aggregate = {}
    for key in scalar_keys:
        vals = [r[key] for r in results]
        aggregate[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": vals}

    report = {
        "dataset": "NetHEPT",
        "graph": {"nodes": graph_data.num_nodes, "edges": graph_data.num_edges},
        "embedding_backend": embedding_backend,
        "device": str(device),
        "protocol": {
            "train_states": len(train_rows),
            "validation_states": len(val_rows),
            "test_states": len(test_rows),
            "candidates_per_state": 16,
            "train_mc": 10,
            "validation_mc": 20,
            "test_mc": 40,
            "training_objective": "MSE on conditional marginal gain Delta(v|S)",
            "test_ranking": "conditional within the same seed set S using common live-edge worlds",
            "limitation": "deterministic seed prefixes can repeat across splits; use the strict experiment for final evidence",
        },
        "runs": results,
        "aggregate": aggregate,
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "test_rows.json").write_text(json.dumps(test_rows, ensure_ascii=False), encoding="utf-8")
    print("FINAL_AGGREGATE", json.dumps(aggregate, ensure_ascii=False), flush=True)
    print(f"saved {OUT / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
