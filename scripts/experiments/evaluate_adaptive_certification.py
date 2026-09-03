from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from grl.algorithms import adaptive_selective_greedy
from grl.baselines import select_degree_discount_nodes, select_high_degree_nodes
from grl.data import load_graph_from_config
from grl.diffusion import estimate_spread
from grl.models import MarginalGainPredictor, build_node_features, load_or_create_node2vec_embeddings
from grl.oracle import BatchedMonteCarloMarginalOracle, LearnedMarginalOracle


def load_model(path: Path, embedding_dim: int, device) -> MarginalGainPredictor:
    model = MarginalGainPredictor(embedding_dim, hidden_dim=96).to(device)
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def build_context(pool_size: int):
    config = yaml.safe_load((ROOT / "configs" / "gnn_nethept.yaml").read_text())
    graph_data = load_graph_from_config(config)
    graph = graph_data.graph
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cache_candidates = [
        ROOT / "outputs" / "marginal_predictability" / "nethept_strict" / "marginal_node2vec_nethept.pth",
        ROOT / "outputs" / "marginal_predictability" / "nethept_grouped" / "model" / "marginal_node2vec_nethept.pth",
    ]
    cache = next((p for p in cache_candidates if p.exists()), cache_candidates[0])
    embeddings = load_or_create_node2vec_embeddings(
        graph, cache, dimensions=64, walk_length=10, num_walks=4, window=5, workers=2, quiet=True
    ).to(device)
    norm_degrees, _ = build_node_features(graph, device=device)
    state_path = ROOT / "outputs" / "marginal_predictability" / "state_conditioning" / "model_state_tuned.pt"
    model = load_model(state_path, embeddings.shape[1], device)
    degree_rank = select_high_degree_nodes(graph, graph_data.num_nodes)
    dd_rank = select_degree_discount_nodes(graph, min(graph_data.num_nodes, max(1000, pool_size * 4)), 0.01)
    candidate_pool = []
    for node in degree_rank[:pool_size] + dd_rank[:pool_size]:
        if node not in candidate_pool:
            candidate_pool.append(node)
        if len(candidate_pool) >= pool_size:
            break
    return graph_data, graph, device, embeddings, norm_degrees, model, candidate_pool


def shortlist_diagnostics(candidate_pool, budget, learned, exact):
    selected = []
    rows = []
    cutoffs = [1, 4, 8, 16, 32, 64]
    for step in range(int(budget)):
        available = [v for v in candidate_pool if v not in set(selected)]
        pred = learned.score(selected, available, step=step)
        truth = exact.score(selected, available, step=step)
        ranked = sorted(available, key=lambda v: (pred[v], -v), reverse=True)
        true_best = max(available, key=lambda v: (truth[v], -v))
        rank = ranked.index(true_best) + 1
        pred_best = ranked[0]
        rows.append({
            "step": step + 1,
            "seed_prefix": list(selected),
            "true_best": true_best,
            "true_best_gain": float(truth[true_best]),
            "learned_rank_of_true_best": int(rank),
            "learned_top1": pred_best,
            "learned_top1_true_gain": float(truth[pred_best]),
            "learned_top1_regret": float(truth[true_best] - truth[pred_best]),
            "recall": {f"top_{k}": bool(rank <= k) for k in cutoffs},
        })
        selected.append(true_best)
    summary = {
        f"top_{k}_recall": sum(int(row["learned_rank_of_true_best"] <= k) for row in rows) / len(rows)
        for k in cutoffs
    }
    ranks = [row["learned_rank_of_true_best"] for row in rows]
    summary.update({
        "mean_rank": float(sum(ranks) / len(ranks)),
        "max_rank": int(max(ranks)),
        "ranks": ranks,
        "full_mc_seeds": list(selected),
    })
    return rows, summary


def run_adaptive(name, graph, candidate_pool, budget, oracle_mc, eval_mc, model, embeddings, norm_degrees, device, beta):
    learned = LearnedMarginalOracle(model, embeddings, norm_degrees, device)
    exact = BatchedMonteCarloMarginalOracle(graph, oracle_mc, random_seed=260903)
    start = time.perf_counter()
    result = adaptive_selective_greedy(
        candidate_pool,
        budget,
        learned,
        exact,
        initial_m=8,
        batch_m=8,
        residual_beta=beta,
        min_rounds=2,
        max_m=None,
    )
    selection_seconds = time.perf_counter() - start
    spread = estimate_spread(graph, result.selected_seeds, eval_mc, 960903)
    stats = vars(exact.stats).copy()
    stats["learned_evaluations"] = learned.stats.learned_evaluations
    verified_per_step = [int(s["verified"]) for s in result.steps]
    return {
        "name": name,
        "residual_beta": beta,
        "selected_seeds": result.selected_seeds,
        "steps": result.steps,
        "selection_seconds": float(selection_seconds),
        "final_spread_mean": float(spread["mean"]),
        "final_spread_std": float(spread["std"]),
        "oracle_stats": stats,
        "verified_per_step": verified_per_step,
        "mean_verified_per_step": float(sum(verified_per_step) / len(verified_per_step)),
        "max_verified_per_step": int(max(verified_per_step)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-size", type=int, default=128)
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--oracle-mc", type=int, default=40)
    parser.add_argument("--eval-mc", type=int, default=1000)
    parser.add_argument("--betas", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    args = parser.parse_args()

    graph_data, graph, device, embeddings, norm_degrees, model, candidate_pool = build_context(args.pool_size)
    print(f"device={device} nodes={graph_data.num_nodes} edges={graph_data.num_edges}", flush=True)
    print(f"candidate_pool={len(candidate_pool)} budget={args.budget}", flush=True)

    diag_learned = LearnedMarginalOracle(model, embeddings, norm_degrees, device)
    diag_exact = BatchedMonteCarloMarginalOracle(graph, args.oracle_mc, random_seed=260903)
    diag_rows, diag_summary = shortlist_diagnostics(candidate_pool, args.budget, diag_learned, diag_exact)
    print("SHORTLIST_DIAGNOSTICS", json.dumps(diag_summary, sort_keys=True), flush=True)
    for row in diag_rows:
        print(
            f"step={row['step']} true_best={row['true_best']} rank={row['learned_rank_of_true_best']} "
            f"top1_regret={row['learned_top1_regret']:.3f}", flush=True
        )

    full_spread = 444.911
    full_candidate_evals = 1235
    prior = ROOT / "docs" / "results" / "nethept_end_to_end_20260903.json"
    if prior.exists():
        try:
            old = json.loads(prior.read_text())
            if "tradeoff" in old:
                full_spread = float(old["tradeoff"][0]["full_spread"])
                full_candidate_evals = int(old["tradeoff"][0]["full_exact_candidate_evaluations"])
        except Exception:
            pass

    methods = {}
    for beta in args.betas:
        key = f"adaptive_beta_{beta:g}"
        methods[key] = run_adaptive(
            key, graph, candidate_pool, args.budget, args.oracle_mc, args.eval_mc,
            model, embeddings, norm_degrees, device, beta,
        )
        item = methods[key]
        item["quality_ratio_vs_full_mc"] = float(item["final_spread_mean"] / full_spread)
        item["exact_fraction_vs_full_mc"] = float(item["oracle_stats"]["candidate_evaluations"] / full_candidate_evals)
        print(
            f"{key} spread={item['final_spread_mean']:.3f} ratio={item['quality_ratio_vs_full_mc']:.4f} "
            f"exact={item['oracle_stats']['candidate_evaluations']} fraction={item['exact_fraction_vs_full_mc']:.4f} "
            f"mean_verified={item['mean_verified_per_step']:.1f} per_step={item['verified_per_step']}", flush=True
        )

    report = {
        "dataset": "NetHEPT",
        "config": vars(args),
        "reference": {
            "full_mc_spread": full_spread,
            "full_mc_candidate_evaluations": full_candidate_evals,
        },
        "shortlist_diagnostics": {"steps": diag_rows, "summary": diag_summary},
        "adaptive_methods": methods,
    }
    out_dir = ROOT / "outputs" / "end_to_end" / "adaptive_certification"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved={out}", flush=True)


if __name__ == "__main__":
    main()
