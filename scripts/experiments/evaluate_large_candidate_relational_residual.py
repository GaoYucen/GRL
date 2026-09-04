from __future__ import annotations

"""Candidate-conditioned relational residual for large-candidate IM proposals.

The state-aware base model is kept frozen.  A lightweight residual explicitly
constructs candidate--seed relation features before pooling, addressing the
candidate-independent seed-pooling bottleneck observed at candidate pools
256/512.  Training uses several disjoint Full-MC trajectories and all available
candidates per state.  If the held-out 512 proposal reaches the rank target,
the script immediately runs the audited-residual end-to-end clean/random pilot.
"""

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

from evaluate_adaptive_certification import build_context
from evaluate_audited_residual_scaleaware import run_level as run_gate_level
from evaluate_large_candidate_hard_negative import fast_scores
from evaluate_trust_calibration_multiseed import run_full_reference
from grl.oracle import BatchedMonteCarloMarginalOracle

OUT = ROOT / "outputs" / "marginal_predictability" / "large_candidate_relational_residual"
OUT.mkdir(parents=True, exist_ok=True)


def mean(xs):
    return float(sum(xs) / len(xs)) if xs else 0.0


class RelationalResidual(nn.Module):
    """Correction that conditions seed aggregation on each candidate."""

    def __init__(self, feature_dim: int, hidden_dim: int = 96):
        super().__init__()
        # seed, candidate, product, abs difference, edge indicator/weight
        rel_dim = feature_dim * 4 + 1
        self.rel = nn.Sequential(
            nn.Linear(rel_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.cand = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # mean relation, max relation, candidate and interactions with mean
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, seed_features, candidate_features, edge_rel, valid):
        # seed_features [C,K,D], candidate_features [C,D]
        c = candidate_features.unsqueeze(1).expand(-1, seed_features.shape[1], -1)
        relation = torch.cat(
            [seed_features, c, seed_features * c, (seed_features - c).abs(), edge_rel],
            dim=-1,
        )
        h = self.rel(relation)
        w = valid.unsqueeze(-1)
        denom = w.sum(dim=1).clamp_min(1.0)
        h_mean = (h * w).sum(dim=1) / denom

        # Max catches a single strongly overlapping selected seed.  Empty seed
        # sets are explicitly zeroed rather than using -inf.
        neg = torch.full_like(h, -1e9)
        masked = torch.where(w > 0, h, neg)
        h_max = masked.max(dim=1).values
        has_seed = (valid.sum(dim=1, keepdim=True) > 0).float()
        h_max = h_max * has_seed

        cand = self.cand(candidate_features)
        x = torch.cat([h_mean, h_max, cand, h_mean * cand, (h_mean - cand).abs()], dim=-1)
        return self.head(x).squeeze(-1)


def feature_tensor(embeddings, norm_degrees):
    return torch.cat([embeddings, norm_degrees], dim=-1)


def relation_batch(features, graph, seeds, candidates, device):
    cands = [int(v) for v in candidates]
    k = max(1, len(seeds))
    c = len(cands)
    d = features.shape[1]
    seed_feats = torch.zeros((c, k, d), dtype=features.dtype, device=device)
    cand_feats = features[torch.as_tensor(cands, dtype=torch.long, device=device)]
    edge_rel = torch.zeros((c, k, 1), dtype=features.dtype, device=device)
    valid = torch.zeros((c, k), dtype=features.dtype, device=device)
    if seeds:
        sidx = torch.as_tensor([int(x) for x in seeds], dtype=torch.long, device=device)
        sf = features[sidx]
        seed_feats[:, : len(seeds), :] = sf.unsqueeze(0).expand(c, -1, -1)
        valid[:, : len(seeds)] = 1.0
        # This is only one relation channel; node2vec product/difference carries
        # broader graph proximity while the edge channel captures direct overlap.
        for i, cand in enumerate(cands):
            for j, s in enumerate(seeds):
                if graph.has_edge(int(cand), int(s)):
                    edge_rel[i, j, 0] = float(graph[int(cand)][int(s)].get("weight", 1.0))
    return seed_feats, cand_feats, edge_rel, valid


def relational_scores(residual, base_model, embeddings, norm_degrees, features, graph, seeds, candidates):
    base = fast_scores(base_model, embeddings, norm_degrees, seeds, candidates)
    tensors = relation_batch(features, graph, seeds, candidates, embeddings.device)
    corr = residual(*tensors)
    return base + corr, base, corr


def collect_full_trajectory(
    graph,
    candidate_pool,
    budget,
    exact_mc,
    exact_seed,
    base_model,
    embeddings,
    norm_degrees,
):
    exact = BatchedMonteCarloMarginalOracle(graph, int(exact_mc), random_seed=int(exact_seed))
    selected = []
    states = []
    base_model.eval()
    for step in range(int(budget)):
        available = [v for v in candidate_pool if v not in set(selected)]
        truth = exact.score(selected, available, step=step)
        with torch.no_grad():
            base = fast_scores(base_model, embeddings, norm_degrees, selected, available)
        base_list = [float(x) for x in base.detach().cpu().tolist()]
        winner = max(available, key=lambda v: (truth[v], -v))
        ranked = [v for _, v in sorted(zip(base_list, available), key=lambda x: (x[0], -x[1]), reverse=True)]
        rank = ranked.index(winner) + 1
        states.append({
            "step": int(step),
            "seeds": [int(x) for x in selected],
            "candidates": [int(v) for v in available],
            "labels": [float(truth[v]) for v in available],
            "winner": int(winner),
            "base_rank": int(rank),
        })
        selected.append(int(winner))
        print(f"collect-rel seed={exact_seed} step={step+1} avail={len(available)} winner={winner} base_rank={rank}", flush=True)
    return states


def train_residual(
    residual,
    base_model,
    embeddings,
    norm_degrees,
    features,
    graph,
    states,
    epochs,
    lr,
    list_weight,
    shape_weight,
    raw_weight,
    corr_weight,
    temperature,
):
    for p in base_model.parameters():
        p.requires_grad_(False)
    base_model.eval()
    opt = torch.optim.Adam(residual.parameters(), lr=float(lr), weight_decay=1e-5)
    rng = random.Random(290901)

    for epoch in range(int(epochs)):
        order = list(range(len(states)))
        rng.shuffle(order)
        totals = {"loss": 0.0, "list": 0.0, "shape": 0.0, "raw": 0.0, "corr": 0.0}
        residual.train()
        for idx in order:
            row = states[idx]
            labels = torch.tensor(row["labels"], dtype=torch.float32, device=embeddings.device)
            pred, base, corr = relational_scores(
                residual, base_model, embeddings, norm_degrees, features, graph,
                row["seeds"], row["candidates"],
            )
            label_std = labels.std(unbiased=False).clamp_min(1.0)
            label_z = (labels - labels.mean()) / label_std
            pred_z = (pred - pred.mean()) / pred.std(unbiased=False).clamp_min(1e-3)

            shape = F.smooth_l1_loss(pred_z, label_z)
            raw = F.smooth_l1_loss((pred - labels) / label_std, torch.zeros_like(pred))
            t = float(temperature)
            target_prob = torch.softmax(label_z / t, dim=0).detach()
            list_loss = -(target_prob * torch.log_softmax(pred_z / t, dim=0)).sum()
            corr_reg = (corr / label_std).pow(2).mean()
            loss = (
                float(list_weight) * list_loss
                + float(shape_weight) * shape
                + float(raw_weight) * raw
                + float(corr_weight) * corr_reg
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(residual.parameters(), 5.0)
            opt.step()
            totals["loss"] += float(loss.item())
            totals["list"] += float(list_loss.item())
            totals["shape"] += float(shape.item())
            totals["raw"] += float(raw.item())
            totals["corr"] += float(corr_reg.item())

        if epoch == 0 or (epoch + 1) % 3 == 0 or epoch + 1 == int(epochs):
            print(
                "rel-ft epoch={} loss={:.4f} list={:.4f} shape={:.4f} raw={:.4f} corr={:.4f}".format(
                    epoch + 1, totals["loss"], totals["list"], totals["shape"], totals["raw"], totals["corr"]
                ), flush=True
            )
    residual.eval()


def rank_diagnostic(
    graph,
    candidate_pool,
    budget,
    exact_mc,
    exact_seed,
    residual,
    base_model,
    embeddings,
    norm_degrees,
    features,
):
    exact = BatchedMonteCarloMarginalOracle(graph, int(exact_mc), random_seed=int(exact_seed))
    selected = []
    rows = []
    for step in range(int(budget)):
        available = [v for v in candidate_pool if v not in set(selected)]
        truth = exact.score(selected, available, step=step)
        with torch.no_grad():
            pred, _, _ = relational_scores(
                residual, base_model, embeddings, norm_degrees, features, graph,
                selected, available,
            )
        vals = [float(x) for x in pred.detach().cpu().tolist()]
        ranked = [v for _, v in sorted(zip(vals, available), key=lambda x: (x[0], -x[1]), reverse=True)]
        winner = max(available, key=lambda v: (truth[v], -v))
        rank = ranked.index(winner) + 1
        top1 = ranked[0]
        rows.append({
            "step": step + 1,
            "winner": int(winner),
            "rank": int(rank),
            "top1": int(top1),
            "top1_regret": float(truth[winner] - truth[top1]),
        })
        selected.append(int(winner))
    ranks = [r["rank"] for r in rows]
    s = {
        "mean_rank": mean(ranks),
        "max_rank": int(max(ranks)),
        "ranks": ranks,
        "mean_top1_regret": mean([r["top1_regret"] for r in rows]),
        "max_top1_regret": float(max(r["top1_regret"] for r in rows)),
    }
    for k in (1, 8, 16, 32, 64, 128):
        s[f"top_{k}_recall"] = mean([float(r <= k) for r in ranks])
    return rows, s


class RelationalLearnedOracle:
    """Adapter so the existing audited gate can use base+relational residual."""

    def __init__(self, residual, base_model, embeddings, norm_degrees, features, graph):
        self.residual = residual
        self.base_model = base_model
        self.embeddings = embeddings
        self.norm_degrees = norm_degrees
        self.features = features
        self.graph = graph
        self.stats = type("Stats", (), {"learned_evaluations": 0})()

    def score(self, selected, candidates, step=None):
        with torch.no_grad():
            pred, _, _ = relational_scores(
                self.residual, self.base_model, self.embeddings, self.norm_degrees,
                self.features, self.graph, selected, candidates,
            )
        vals = pred.detach().cpu().tolist()
        self.stats.learned_evaluations += len(candidates)
        return {int(v): float(x) for v, x in zip(candidates, vals)}


def gate_pilot(graph, pool, budget, eval_mc, learned_oracle_factory):
    # Local implementation mirrors current audited-residual scale-aware logic,
    # while allowing a custom learned oracle adapter.
    from evaluate_audited_residual_gate import audited_residual_greedy
    from evaluate_progressive_mc import ProgressiveMonteCarloOracle
    from evaluate_robustness_stress import CorruptedLearnedOracle
    from grl.diffusion import estimate_spread

    exact_seed = 1280401
    eval_seed = 1290401
    corruption_seed = 1270401
    ref = run_full_reference(graph, pool, budget, eval_mc, exact_seed, eval_seed)
    full_samples = sum(len(pool) - step for step in range(budget)) * 40
    methods = {}
    for alpha in (0.0, 1.0):
        base_learned = learned_oracle_factory()
        learned = CorruptedLearnedOracle(base_learned, alpha=float(alpha), random_seed=corruption_seed)
        exact = ProgressiveMonteCarloOracle(graph, max_mc=40, random_seed=exact_seed)
        seeds, steps = audited_residual_greedy(
            pool, budget, learned, exact,
            audit_top_k=16, audit_sentinels=8, audit_mc=20,
            residual_q=1.0, residual_beta=0.0,
        )
        spread = estimate_spread(graph, seeds, eval_mc, eval_seed)
        fallback = sum(int(x["mode"] == "audited_residual_fallback_full_mc40") for x in steps)
        item = {
            "alpha": alpha,
            "selected_seeds": seeds,
            "final_spread_mean": float(spread["mean"]),
            "quality_ratio_vs_full_mc": float(spread["mean"] / ref["final_spread_mean"]),
            "mc_candidate_samples": int(exact.stats.mc_candidate_samples),
            "sample_fraction_vs_full_mc": float(exact.stats.mc_candidate_samples / full_samples),
            "fallback_steps": int(fallback),
        }
        methods[f"alpha_{alpha:g}"] = item
        print(
            f"REL-GATE alpha={alpha:g} spread={item['final_spread_mean']:.3f} "
            f"ratio={item['quality_ratio_vs_full_mc']:.4f} frac={item['sample_fraction_vs_full_mc']:.3f} "
            f"fallback={fallback}/{budget}", flush=True
        )
    return {"reference": ref, "methods": methods}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, default=10)
    p.add_argument("--train-mc", type=int, default=25)
    p.add_argument("--eval-mc", type=int, default=40)
    p.add_argument("--train-seeds-256", nargs="+", type=int, default=[291101, 291503])
    p.add_argument("--train-seeds-512", nargs="+", type=int, default=[292101, 292503])
    p.add_argument("--eval-seed", type=int, default=260903)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--list-weight", type=float, default=0.7)
    p.add_argument("--shape-weight", type=float, default=1.0)
    p.add_argument("--raw-weight", type=float, default=0.2)
    p.add_argument("--corr-weight", type=float, default=0.01)
    p.add_argument("--temperature", type=float, default=0.4)
    args = p.parse_args()

    gd, graph, device, embeddings, norm_degrees, base_model, full_pool = build_context(512)
    features = feature_tensor(embeddings, norm_degrees)
    print(f"device={device} nodes={gd.num_nodes} edges={gd.num_edges} pool=512", flush=True)

    # Keep the already successful state-aware base frozen.
    for param in base_model.parameters():
        param.requires_grad_(False)
    base_model.eval()

    states = []
    for seed in args.train_seeds_256:
        states.extend(collect_full_trajectory(
            graph, full_pool[:256], args.budget, args.train_mc, seed,
            base_model, embeddings, norm_degrees,
        ))
    for seed in args.train_seeds_512:
        states.extend(collect_full_trajectory(
            graph, full_pool[:512], args.budget, args.train_mc, seed,
            base_model, embeddings, norm_degrees,
        ))
    print(f"relational_training_states={len(states)}", flush=True)

    residual = RelationalResidual(features.shape[1], hidden_dim=96).to(device)
    train_residual(
        residual, base_model, embeddings, norm_degrees, features, graph, states,
        args.epochs, args.lr, args.list_weight, args.shape_weight,
        args.raw_weight, args.corr_weight, args.temperature,
    )

    tuned = {}
    for pool_size in (256, 512):
        rows, s = rank_diagnostic(
            graph, full_pool[:pool_size], args.budget, args.eval_mc, args.eval_seed,
            residual, base_model, embeddings, norm_degrees, features,
        )
        tuned[str(pool_size)] = {"summary": s, "steps": rows}
        print(f"REL-TUNED pool={pool_size} {json.dumps(s, sort_keys=True)}", flush=True)

    s512 = tuned["512"]["summary"]
    proposal_pass = bool(s512["top_64_recall"] >= 0.8 and s512["mean_rank"] < 40.0)
    print(f"REL_PROPOSAL_PASS={proposal_pass}", flush=True)

    gate = None
    if proposal_pass:
        print("=== RELATIONAL END-TO-END GATE PILOT ===", flush=True)
        def factory():
            return RelationalLearnedOracle(
                residual, base_model, embeddings, norm_degrees, features, graph
            )
        gate = gate_pilot(graph, full_pool[:512], args.budget, 1000, factory)
    else:
        print("relational gate pilot skipped because proposal target was not met", flush=True)

    report = {
        "dataset": "NetHEPT",
        "scope": "candidate-conditioned relational residual large-candidate proposal",
        "config": vars(args),
        "tuned": tuned,
        "proposal_pass": proposal_pass,
        "gate_pilot": gate,
    }
    out = OUT / "report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save({"state_dict": residual.state_dict(), "config": vars(args)}, OUT / "relational_residual.pt")
    print(f"saved={out}", flush=True)


if __name__ == "__main__":
    main()
