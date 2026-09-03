from __future__ import annotations

"""Candidate-conditioned residual ablation for conditional marginal gain.

Keep the strong strict baseline predictor frozen and learn only a state-dependent
correction from explicit candidate--seed relations. This tests whether the
remaining failure is architectural (candidate-independent seed pooling) after
controlled state-sensitive supervision has exposed the right signal.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from evaluate_marginal_predictability import BASE_SEED, build_edge_arrays, conditional_metrics, materialize, set_seed
from evaluate_marginal_strict import make_unique_states
from evaluate_state_conditioning import (
    controlled_metrics,
    label_controlled,
    make_controlled_states,
    predict_controlled,
)

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from grl.baselines import select_high_degree_nodes
from grl.data import load_graph_from_config
from grl.models import MarginalGainPredictor, build_node_features, load_or_create_node2vec_embeddings

STRICT_OUT = ROOT / "outputs" / "marginal_predictability" / "nethept_strict"
OUT = ROOT / "outputs" / "marginal_predictability" / "candidate_conditioned_residual"
OUT.mkdir(parents=True, exist_ok=True)


class CandidateConditionedResidual(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 96):
        super().__init__()
        relation_dim = feature_dim * 4 + 1
        self.rel = nn.Sequential(
            nn.Linear(relation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.cand = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, seed_features, candidate_features, edge_weights, valid_mask):
        # seed_features: [B,K,D], candidate_features: [B,D]
        c = candidate_features.unsqueeze(1).expand(-1, seed_features.shape[1], -1)
        relation = torch.cat([
            seed_features,
            c,
            seed_features * c,
            (seed_features - c).abs(),
            edge_weights,
        ], dim=-1)
        h = self.rel(relation)
        w = valid_mask.unsqueeze(-1)
        pooled = (h * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
        cand = self.cand(candidate_features)
        return self.head(torch.cat([pooled, cand, pooled * cand, (pooled - cand).abs()], dim=-1))


def graph_feature_tensor(embeddings, norm_degrees):
    return torch.cat([embeddings, norm_degrees], dim=-1)


def relational_tensors(rows, features, graph, device):
    max_k = max(max(1, len(r["seeds"])) for r in rows)
    b, d = len(rows), features.shape[1]
    seed_feats = torch.zeros((b, max_k, d), dtype=torch.float32, device=device)
    cand_feats = torch.zeros((b, d), dtype=torch.float32, device=device)
    edge_w = torch.zeros((b, max_k, 1), dtype=torch.float32, device=device)
    valid = torch.zeros((b, max_k), dtype=torch.float32, device=device)
    for i, row in enumerate(rows):
        cand = int(row["candidate"] if "candidate" in row else row["candidates"][0])
        cand_feats[i] = features[cand]
        for j, s in enumerate(row["seeds"]):
            seed_feats[i, j] = features[int(s)]
            valid[i, j] = 1.0
            if graph.has_edge(cand, int(s)):
                edge_w[i, j, 0] = float(graph[cand][int(s)].get("weight", 0.0))
    return seed_feats, cand_feats, edge_w, valid


def baseline_pred_rows(model, embeddings, norm_degrees, rows, num_nodes, device):
    vals = []
    model.eval()
    with torch.no_grad():
        for row in rows:
            cand = int(row["candidate"] if "candidate" in row else row["candidates"][0])
            mask = torch.zeros((num_nodes, 1), dtype=torch.float32, device=device)
            if row["seeds"]:
                mask[row["seeds"], 0] = 1.0
            vals.append(float(model(embeddings, norm_degrees, mask, cand).item()))
    return torch.tensor(vals, dtype=torch.float32, device=device).unsqueeze(-1)


def train_residual(net, tensors, baseline_pred, labels, groups, epochs=40):
    seed_feats, cand_feats, edge_w, valid = tensors
    opt = torch.optim.Adam(net.parameters(), lr=5e-4, weight_decay=1e-5)
    pair_idx = []
    by_group = {}
    for i, g in enumerate(groups):
        by_group.setdefault(int(g), []).append(i)
    for idxs in by_group.values():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                pair_idx.append((idxs[a], idxs[b]))
    pair_idx = torch.tensor(pair_idx, dtype=torch.long, device=labels.device)

    n = len(labels)
    bs = 64
    for epoch in range(epochs):
        order = torch.randperm(n, device=labels.device)
        net.train()
        loss_abs = 0.0
        for start in range(0, n, bs):
            idx = order[start:start+bs]
            corr = net(seed_feats[idx], cand_feats[idx], edge_w[idx], valid[idx])
            pred = baseline_pred[idx] + corr
            loss = F.mse_loss(pred, labels[idx])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
            loss_abs += float(loss.item())

        porder = pair_idx[torch.randperm(len(pair_idx), device=labels.device)]
        loss_diff = 0.0
        for start in range(0, len(porder), bs):
            pair = porder[start:start+bs]
            ia, ib = pair[:,0], pair[:,1]
            ca = net(seed_feats[ia], cand_feats[ia], edge_w[ia], valid[ia])
            cb = net(seed_feats[ib], cand_feats[ib], edge_w[ib], valid[ib])
            pa = baseline_pred[ia] + ca
            pb = baseline_pred[ib] + cb
            target = labels[ia] - labels[ib]
            loss = F.smooth_l1_loss(pa - pb, target)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
            loss_diff += float(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f"residual epoch={epoch+1} abs={loss_abs:.4f} diff={loss_diff:.4f}", flush=True)


def predict_total(net, tensors, baseline_pred):
    net.eval()
    with torch.no_grad():
        return (baseline_pred + net(*tensors)).squeeze(-1).detach().cpu().tolist()


def standard_predictions(net, baseline, embeddings, norm_degrees, features, rows, graph, num_nodes, device):
    out = []
    baseline.eval(); net.eval()
    with torch.no_grad():
        for row in rows:
            mask = torch.zeros((num_nodes, 1), dtype=torch.float32, device=device)
            if row["seeds"]:
                mask[row["seeds"], 0] = 1.0
            preds = []
            for cand in row["candidates"]:
                base = baseline(embeddings, norm_degrees, mask, cand)
                tmp = {"seeds": row["seeds"], "candidate": int(cand)}
                tensors = relational_tensors([tmp], features, graph, device)
                corr = net(*tensors)
                preds.append(float((base + corr).item()))
            out.append(preds)
    return out


def main():
    set_seed(BASE_SEED + 10000)
    config = yaml.safe_load((ROOT / "configs" / "gnn_nethept.yaml").read_text())
    graph_data = load_graph_from_config(config)
    graph = graph_data.graph
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    embeddings = load_or_create_node2vec_embeddings(
        graph, STRICT_OUT / "marginal_node2vec_nethept.pth",
        dimensions=64, walk_length=10, num_walks=4, window=5, workers=2, quiet=True,
    ).to(device)
    norm_degrees, _ = build_node_features(graph, device=device)
    features = graph_feature_tensor(embeddings, norm_degrees)
    edge_arrays = build_edge_arrays(graph)

    ckpt = torch.load(STRICT_OUT / "model.pt", map_location=device, weights_only=False)
    baseline = MarginalGainPredictor(embeddings.shape[1], hidden_dim=96).to(device)
    baseline.load_state_dict(ckpt["state_dict"])
    for p in baseline.parameters():
        p.requires_grad_(False)

    degree_rank = select_high_degree_nodes(graph, graph_data.num_nodes)
    eligible = [v for v in degree_rank if graph.out_degree(v) >= 8]
    train_candidates = eligible[:24]
    eval_candidates = eligible[24:36]
    assert set(train_candidates).isdisjoint(eval_candidates)

    train_states = make_controlled_states(graph_data, train_candidates, repeats=2, seed=BASE_SEED + 9101)
    eval_states = make_controlled_states(graph_data, eval_candidates, repeats=2, seed=BASE_SEED + 9102)
    train_rows = label_controlled(train_states, edge_arrays, graph.is_directed(), 25, BASE_SEED + 920000)
    eval_rows = label_controlled(eval_states, edge_arrays, graph.is_directed(), 80, BASE_SEED + 930000)

    base_eval = predict_controlled(baseline, embeddings, norm_degrees, eval_rows, graph_data.num_nodes, device)
    base_metrics = controlled_metrics(eval_rows, base_eval)

    train_tensors = relational_tensors(train_rows, features, graph, device)
    eval_tensors = relational_tensors(eval_rows, features, graph, device)
    base_train_tensor = baseline_pred_rows(baseline, embeddings, norm_degrees, train_rows, graph_data.num_nodes, device)
    base_eval_tensor = torch.tensor(base_eval, dtype=torch.float32, device=device).unsqueeze(-1)
    labels = torch.tensor([r["label"] for r in train_rows], dtype=torch.float32, device=device).unsqueeze(-1)
    groups = [r["group"] for r in train_rows]

    net = CandidateConditionedResidual(features.shape[1], hidden_dim=96).to(device)
    train_residual(net, train_tensors, base_train_tensor, labels, groups, epochs=40)
    rel_eval = predict_total(net, eval_tensors, base_eval_tensor)
    rel_metrics = controlled_metrics(eval_rows, rel_eval)
    print("REL_BASELINE", json.dumps(base_metrics), flush=True)
    print("REL_CANDIDATE_CONDITIONED", json.dumps(rel_metrics), flush=True)

    std_states, _ = make_unique_states(graph_data, 12, 16, BASE_SEED + 10401)
    std_rows = materialize(std_states, edge_arrays, graph.is_directed(), mc_runs=60, seed_base=BASE_SEED + 1040000)
    std_preds = standard_predictions(net, baseline, embeddings, norm_degrees, features, std_rows, graph, graph_data.num_nodes, device)
    std_metrics = conditional_metrics(std_rows, std_preds, graph)
    print("REL_STANDARD", json.dumps({k:v for k,v in std_metrics.items() if k != "by_seed_size"}), flush=True)

    report = {
        "dataset": "NetHEPT",
        "device": str(device),
        "protocol": {
            "frozen_strict_baseline": True,
            "train_candidates": len(train_candidates),
            "eval_candidates": len(eval_candidates),
            "candidate_disjoint": True,
            "explicit_relation_features": ["seed", "candidate", "product", "abs_difference", "candidate_to_seed_edge_weight"],
            "epochs": 40,
        },
        "baseline_controlled": base_metrics,
        "candidate_conditioned_controlled": rel_metrics,
        "candidate_conditioned_standard_ranking": std_metrics,
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save({"state_dict": net.state_dict()}, OUT / "residual_model.pt")
    print(f"saved {OUT / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
