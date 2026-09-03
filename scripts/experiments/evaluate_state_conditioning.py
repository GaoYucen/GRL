from __future__ import annotations

"""Controlled state-conditioning pre-experiment for NetHEPT.

The strict marginal-gain experiment showed excellent unseen-state candidate
ranking but weak sensitivity to the current seed set. This script fixes a
candidate v and varies S with controlled overlap against v's outgoing
neighborhood, then compares the strict baseline with a small contrastive
fine-tuning stage on disjoint candidate nodes.
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import pearsonr, spearmanr

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
)
from evaluate_marginal_strict import make_unique_states

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from grl.baselines import select_high_degree_nodes
from grl.data import load_graph_from_config
from grl.models import MarginalGainPredictor, build_node_features, load_or_create_node2vec_embeddings

OUT = ROOT / "outputs" / "marginal_predictability" / "state_conditioning"
OUT.mkdir(parents=True, exist_ok=True)
STRICT_OUT = ROOT / "outputs" / "marginal_predictability" / "nethept_strict"
OVERLAP_LEVELS = (0, 2, 4, 8)


def make_controlled_states(graph_data, candidates, repeats: int, seed: int):
    graph = graph_data.graph
    nodes = list(range(graph_data.num_nodes))
    rng = random.Random(seed)
    rows = []
    group_id = 0
    for cand in candidates:
        nbrs_all = list(graph.successors(cand)) if graph.is_directed() else list(graph.neighbors(cand))
        nbrs_all = [n for n in nbrs_all if n != cand]
        if len(nbrs_all) < max(OVERLAP_LEVELS):
            continue
        far_pool = [n for n in nodes if n != cand and n not in set(nbrs_all)]
        for rep in range(repeats):
            nbrs = list(nbrs_all)
            rng.shuffle(nbrs)
            far = rng.sample(far_pool, 8)
            overlap_nodes = nbrs[:8]
            for k in OVERLAP_LEVELS:
                seeds = overlap_nodes[:k] + far[: 8 - k]
                rows.append({
                    "candidate": int(cand),
                    "seeds": [int(x) for x in seeds],
                    "overlap": int(k),
                    "group": int(group_id),
                    "repeat": int(rep),
                })
            group_id += 1
    return rows


def label_controlled(rows, edge_arrays, directed: bool, mc_runs: int, seed_base: int):
    out = []
    for i, row in enumerate(rows):
        # Same live-edge worlds for all four overlap levels in one group.
        world_seed = seed_base + 1009 * int(row["group"])
        y = float(label_state(
            *edge_arrays,
            directed,
            row["seeds"],
            [row["candidate"]],
            mc_runs,
            world_seed,
        )[0])
        item = dict(row)
        item["label"] = y
        out.append(item)
        if (i + 1) % 40 == 0 or i + 1 == len(rows):
            print(f"controlled labels {i + 1}/{len(rows)} mc={mc_runs}", flush=True)
    return out


def tensorize_controlled(rows, num_nodes: int, device):
    masks = torch.zeros((len(rows), num_nodes, 1), dtype=torch.float32, device=device)
    candidates = torch.empty((len(rows),), dtype=torch.long, device=device)
    labels = torch.empty((len(rows), 1), dtype=torch.float32, device=device)
    for i, row in enumerate(rows):
        if row["seeds"]:
            masks[i, row["seeds"], 0] = 1.0
        candidates[i] = int(row["candidate"])
        labels[i, 0] = float(row["label"])
    return masks, candidates, labels


def predict_controlled(model, embeddings, norm_degrees, rows, num_nodes: int, device):
    preds = []
    model.eval()
    with torch.no_grad():
        for row in rows:
            mask = torch.zeros((num_nodes, 1), dtype=torch.float32, device=device)
            mask[row["seeds"], 0] = 1.0
            preds.append(float(model(embeddings, norm_degrees, mask, row["candidate"]).item()))
    return preds


def controlled_metrics(rows, preds):
    groups = defaultdict(list)
    for row, pred in zip(rows, preds):
        groups[int(row["group"])].append((int(row["overlap"]), float(row["label"]), float(pred)))

    centered_y = []
    centered_p = []
    group_spearman = []
    group_pairwise = []
    true_ranges = []
    pred_ranges = []
    true_drops = []
    pred_drops = []
    direction = []

    for vals in groups.values():
        vals = sorted(vals)
        k = np.asarray([x[0] for x in vals], dtype=float)
        y = np.asarray([x[1] for x in vals], dtype=float)
        p = np.asarray([x[2] for x in vals], dtype=float)
        centered_y.extend((y - y.mean()).tolist())
        centered_p.extend((p - p.mean()).tolist())
        rho = spearmanr(y, p).statistic if np.std(y) > 0 and np.std(p) > 0 else 0.0
        if not np.isfinite(rho):
            rho = 0.0
        group_spearman.append(float(rho))
        correct = total = 0
        for i in range(len(y)):
            for j in range(i + 1, len(y)):
                dy = y[i] - y[j]
                if abs(dy) < 1e-9:
                    continue
                dp = p[i] - p[j]
                total += 1
                if dp * dy > 0:
                    correct += 1
                elif abs(dp) < 1e-9:
                    correct += 0.5
        group_pairwise.append(correct / total if total else 0.0)
        true_ranges.append(float(y.max() - y.min()))
        pred_ranges.append(float(p.max() - p.min()))
        i0 = int(np.where(k == min(OVERLAP_LEVELS))[0][0])
        i8 = int(np.where(k == max(OVERLAP_LEVELS))[0][0])
        td = float(y[i0] - y[i8])
        pd = float(p[i0] - p[i8])
        true_drops.append(td)
        pred_drops.append(pd)
        if abs(td) > 1e-9:
            direction.append(float(np.sign(td) == np.sign(pd)))

    cy = np.asarray(centered_y, dtype=float)
    cp = np.asarray(centered_p, dtype=float)
    td = np.asarray(true_drops, dtype=float)
    pd = np.asarray(pred_drops, dtype=float)
    centered_pearson = pearsonr(cy, cp).statistic if np.std(cy) > 0 and np.std(cp) > 0 else 0.0
    drop_pearson = pearsonr(td, pd).statistic if np.std(td) > 0 and np.std(pd) > 0 else 0.0
    return {
        "groups": len(groups),
        "states": len(rows),
        "mean_group_spearman": float(np.mean(group_spearman)),
        "mean_group_pairwise_accuracy": float(np.mean(group_pairwise)),
        "centered_pearson": float(centered_pearson),
        "centered_mae": float(np.mean(np.abs(cp - cy))),
        "mean_true_range": float(np.mean(true_ranges)),
        "mean_pred_range": float(np.mean(pred_ranges)),
        "range_ratio": float(np.mean(pred_ranges) / max(1e-12, np.mean(true_ranges))),
        "mean_true_drop_0_to_8": float(np.mean(td)),
        "mean_pred_drop_0_to_8": float(np.mean(pd)),
        "drop_mae": float(np.mean(np.abs(pd - td))),
        "drop_pearson": float(drop_pearson),
        "drop_direction_accuracy": float(np.mean(direction)) if direction else 0.0,
    }


def fine_tune(model, embeddings, norm_degrees, rows, num_nodes: int, device, epochs: int = 25):
    masks, candidates, labels = tensorize_controlled(rows, num_nodes, device)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[int(row["group"])].append(i)
    pair_idx = []
    for idxs in groups.values():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                pair_idx.append((idxs[a], idxs[b]))
    pair_idx = torch.tensor(pair_idx, dtype=torch.long, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    batch_size = 64
    n = len(rows)
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(n, device=device)
        mse_sum = 0.0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            b = len(idx)
            pred = model(
                embeddings.unsqueeze(0).expand(b, -1, -1),
                norm_degrees.unsqueeze(0).expand(b, -1, -1),
                masks[idx],
                candidates[idx],
            )
            loss = F.mse_loss(pred, labels[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            mse_sum += float(loss.item())

        # Explicitly teach changes in Delta(v|S) for the same v across S.
        porder = pair_idx[torch.randperm(len(pair_idx), device=device)]
        diff_sum = 0.0
        for start in range(0, len(porder), batch_size):
            pairs = porder[start:start + batch_size]
            ia, ib = pairs[:, 0], pairs[:, 1]
            b = len(ia)
            pa = model(
                embeddings.unsqueeze(0).expand(b, -1, -1),
                norm_degrees.unsqueeze(0).expand(b, -1, -1),
                masks[ia], candidates[ia],
            )
            pb = model(
                embeddings.unsqueeze(0).expand(b, -1, -1),
                norm_degrees.unsqueeze(0).expand(b, -1, -1),
                masks[ib], candidates[ib],
            )
            target = labels[ia] - labels[ib]
            loss = F.smooth_l1_loss(pa - pb, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            diff_sum += float(loss.item())
        if (epoch + 1) % 5 == 0:
            print(f"state-ft epoch={epoch+1} mse_sum={mse_sum:.4f} diff_sum={diff_sum:.4f}", flush=True)


def main():
    set_seed(BASE_SEED + 9000)
    config = yaml.safe_load((ROOT / "configs" / "gnn_nethept.yaml").read_text())
    graph_data = load_graph_from_config(config)
    graph = graph_data.graph
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device} nodes={graph_data.num_nodes} edges={graph_data.num_edges}", flush=True)

    emb_path = STRICT_OUT / "marginal_node2vec_nethept.pth"
    embeddings = load_or_create_node2vec_embeddings(
        graph, emb_path, dimensions=64, walk_length=10, num_walks=4, window=5, workers=2, quiet=True,
    ).to(device)
    norm_degrees, _ = build_node_features(graph, device=device)
    edge_arrays = build_edge_arrays(graph)

    checkpoint = torch.load(STRICT_OUT / "model.pt", map_location=device, weights_only=False)
    baseline = MarginalGainPredictor(embeddings.shape[1], hidden_dim=96).to(device)
    baseline.load_state_dict(checkpoint["state_dict"])

    degree_rank = select_high_degree_nodes(graph, graph_data.num_nodes)
    eligible = [v for v in degree_rank if graph.out_degree(v) >= 8]
    train_candidates = eligible[:24]
    eval_candidates = eligible[24:36]
    assert set(train_candidates).isdisjoint(eval_candidates)
    print(f"controlled train_candidates={len(train_candidates)} eval_candidates={len(eval_candidates)}", flush=True)

    train_states = make_controlled_states(graph_data, train_candidates, repeats=2, seed=BASE_SEED + 9101)
    eval_states = make_controlled_states(graph_data, eval_candidates, repeats=2, seed=BASE_SEED + 9102)
    train_rows = label_controlled(train_states, edge_arrays, graph.is_directed(), mc_runs=25, seed_base=BASE_SEED + 920000)
    eval_rows = label_controlled(eval_states, edge_arrays, graph.is_directed(), mc_runs=80, seed_base=BASE_SEED + 930000)

    baseline_pred = predict_controlled(baseline, embeddings, norm_degrees, eval_rows, graph_data.num_nodes, device)
    baseline_metrics = controlled_metrics(eval_rows, baseline_pred)
    print("STATE_BASELINE", json.dumps(baseline_metrics), flush=True)

    tuned = MarginalGainPredictor(embeddings.shape[1], hidden_dim=96).to(device)
    tuned.load_state_dict(checkpoint["state_dict"])
    fine_tune(tuned, embeddings, norm_degrees, train_rows, graph_data.num_nodes, device, epochs=25)
    tuned_pred = predict_controlled(tuned, embeddings, norm_degrees, eval_rows, graph_data.num_nodes, device)
    tuned_metrics = controlled_metrics(eval_rows, tuned_pred)
    print("STATE_TUNED", json.dumps(tuned_metrics), flush=True)

    # Small standard candidate-ranking check to detect catastrophic loss of the
    # original within-state capability after the conditioning fine-tune.
    std_states, _ = make_unique_states(graph_data, 12, 16, BASE_SEED + 9401)
    std_rows = materialize(std_states, edge_arrays, graph.is_directed(), mc_runs=60, seed_base=BASE_SEED + 940000)
    std_preds = []
    tuned.eval()
    with torch.no_grad():
        for row in std_rows:
            mask = torch.zeros((graph_data.num_nodes, 1), dtype=torch.float32, device=device)
            mask[row["seeds"], 0] = 1.0
            std_preds.append([
                float(tuned(embeddings, norm_degrees, mask, cand).item())
                for cand in row["candidates"]
            ])
    standard_metrics = conditional_metrics(std_rows, std_preds, graph)
    print("TUNED_STANDARD", json.dumps({k:v for k,v in standard_metrics.items() if k != 'by_seed_size'}), flush=True)

    report = {
        "dataset": "NetHEPT",
        "device": str(device),
        "protocol": {
            "overlap_levels": list(OVERLAP_LEVELS),
            "seed_set_size": 8,
            "train_candidates": len(train_candidates),
            "eval_candidates": len(eval_candidates),
            "candidate_disjoint": True,
            "repeats": 2,
            "train_mc": 25,
            "eval_mc": 80,
            "fine_tune_epochs": 25,
            "fine_tune_lr": 3e-4,
            "losses": ["absolute_mse", "same_candidate_pairwise_delta_smooth_l1"],
        },
        "baseline_controlled": baseline_metrics,
        "tuned_controlled": tuned_metrics,
        "tuned_standard_ranking": standard_metrics,
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save({"state_dict": tuned.state_dict()}, OUT / "model_state_tuned.pt")
    print(f"saved {OUT / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
