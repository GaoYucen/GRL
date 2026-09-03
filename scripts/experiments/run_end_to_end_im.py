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

from grl.algorithms import full_oracle_greedy, learned_greedy, selective_greedy
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


def run_method(name, selector, graph, mc_eval, eval_seed):
    start = time.perf_counter()
    result = selector()
    selection_seconds = time.perf_counter() - start
    spread = estimate_spread(graph, result.selected_seeds, mc_eval, eval_seed)
    return {
        "name": name,
        "selected_seeds": result.selected_seeds,
        "steps": result.steps,
        "selection_seconds": selection_seconds,
        "final_spread_mean": float(spread["mean"]),
        "final_spread_std": float(spread["std"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-size", type=int, default=128)
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--oracle-mc", type=int, default=40)
    parser.add_argument("--eval-mc", type=int, default=1000)
    parser.add_argument("--top-m", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.pool_size = min(args.pool_size, 16)
        args.budget = min(args.budget, 3)
        args.oracle_mc = min(args.oracle_mc, 5)
        args.eval_mc = min(args.eval_mc, 30)
        args.top_m = min(args.top_m, 4)

    config = yaml.safe_load((ROOT / "configs" / "gnn_nethept.yaml").read_text())
    graph_data = load_graph_from_config(config)
    graph = graph_data.graph
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device} nodes={graph_data.num_nodes} edges={graph_data.num_edges}", flush=True)

    cache_candidates = [
        ROOT / "outputs" / "marginal_predictability" / "nethept_strict" / "marginal_node2vec_nethept.pth",
        ROOT / "outputs" / "marginal_predictability" / "nethept_grouped" / "model" / "marginal_node2vec_nethept.pth",
    ]
    cache = next((p for p in cache_candidates if p.exists()), cache_candidates[0])
    embeddings = load_or_create_node2vec_embeddings(
        graph, cache, dimensions=64, walk_length=10, num_walks=4, window=5, workers=2, quiet=True
    ).to(device)
    norm_degrees, _ = build_node_features(graph, device=device)

    strict_path = ROOT / "outputs" / "marginal_predictability" / "nethept_strict" / "model.pt"
    state_path = ROOT / "outputs" / "marginal_predictability" / "state_conditioning" / "model_state_tuned.pt"
    if not strict_path.exists() or not state_path.exists():
        raise FileNotFoundError(f"required checkpoints missing: strict={strict_path.exists()} state={state_path.exists()}")
    strict_model = load_model(strict_path, embeddings.shape[1], device)
    state_model = load_model(state_path, embeddings.shape[1], device)

    degree_rank = select_high_degree_nodes(graph, graph_data.num_nodes)
    dd_rank = select_degree_discount_nodes(graph, min(graph_data.num_nodes, max(1000, args.pool_size * 4)), 0.01)
    candidate_pool = []
    for node in degree_rank[: args.pool_size] + dd_rank[: args.pool_size]:
        if node not in candidate_pool:
            candidate_pool.append(node)
        if len(candidate_pool) >= args.pool_size:
            break
    print(f"candidate_pool={len(candidate_pool)} budget={args.budget} oracle_mc={args.oracle_mc} eval_mc={args.eval_mc}", flush=True)

    methods = {}
    full_mc = BatchedMonteCarloMarginalOracle(graph, args.oracle_mc, random_seed=260903)
    methods["full_mc_greedy"] = run_method(
        "full_mc_greedy",
        lambda: full_oracle_greedy(candidate_pool, args.budget, full_mc),
        graph, args.eval_mc, 960903,
    )
    methods["full_mc_greedy"]["oracle_stats"] = vars(full_mc.stats)

    strict_learned = LearnedMarginalOracle(strict_model, embeddings, norm_degrees, device)
    methods["learned_strict"] = run_method(
        "learned_strict", lambda: learned_greedy(candidate_pool, args.budget, strict_learned), graph, args.eval_mc, 960903
    )
    methods["learned_strict"]["oracle_stats"] = vars(strict_learned.stats)

    state_learned = LearnedMarginalOracle(state_model, embeddings, norm_degrees, device)
    methods["learned_state_aware"] = run_method(
        "learned_state_aware", lambda: learned_greedy(candidate_pool, args.budget, state_learned), graph, args.eval_mc, 960903
    )
    methods["learned_state_aware"]["oracle_stats"] = vars(state_learned.stats)

    selective_learned = LearnedMarginalOracle(state_model, embeddings, norm_degrees, device)
    selective_mc = BatchedMonteCarloMarginalOracle(graph, args.oracle_mc, random_seed=260903)
    methods["selective_state_aware"] = run_method(
        "selective_state_aware",
        lambda: selective_greedy(candidate_pool, args.budget, selective_learned, selective_mc, top_m=args.top_m),
        graph, args.eval_mc, 960903,
    )
    stats = vars(selective_mc.stats).copy()
    stats["learned_evaluations"] = selective_learned.stats.learned_evaluations
    methods["selective_state_aware"]["oracle_stats"] = stats

    degree_seeds = select_high_degree_nodes(graph, args.budget)
    dd_seeds = select_degree_discount_nodes(graph, args.budget, 0.01)
    for name, seeds in [("degree", degree_seeds), ("degree_discount", dd_seeds)]:
        start = time.perf_counter()
        spread = estimate_spread(graph, seeds, args.eval_mc, 960903)
        methods[name] = {
            "name": name,
            "selected_seeds": list(seeds),
            "steps": [],
            "selection_seconds": time.perf_counter() - start,
            "final_spread_mean": float(spread["mean"]),
            "final_spread_std": float(spread["std"]),
            "oracle_stats": {},
        }

    reference = methods["full_mc_greedy"]["final_spread_mean"]
    for item in methods.values():
        item["quality_ratio_vs_full_mc"] = float(item["final_spread_mean"] / reference) if reference else 0.0

    report = {
        "dataset": "NetHEPT",
        "prototype_scope": "fixed candidate pool; exact MC, learned, and selective methods share the same pool",
        "device": str(device),
        "config": vars(args),
        "candidate_pool": candidate_pool,
        "methods": methods,
    }
    out_dir = ROOT / "outputs" / "end_to_end" / ("nethept_smoke" if args.smoke else "nethept_prototype")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=== SUMMARY ===", flush=True)
    for name, item in methods.items():
        s = item.get("oracle_stats", {})
        print(
            f"{name:24s} spread={item['final_spread_mean']:.3f} ratio={item['quality_ratio_vs_full_mc']:.4f} "
            f"time={item['selection_seconds']:.3f}s exact_candidates={s.get('candidate_evaluations', 0)} "
            f"mc_pairs={s.get('mc_candidate_samples', 0)} learned={s.get('learned_evaluations', 0)}",
            flush=True,
        )
    print(f"saved={out}", flush=True)


if __name__ == "__main__":
    main()
