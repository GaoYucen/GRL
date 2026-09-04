from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

from evaluate_adaptive_certification import build_context
from evaluate_audited_residual_gate import audited_residual_greedy
from evaluate_progressive_mc import ProgressiveMonteCarloOracle
from evaluate_trust_calibration_multiseed import run_full_reference
from grl.baselines import select_degree_discount_nodes, select_high_degree_nodes
from grl.diffusion import estimate_spread
from grl.oracle import LearnedMarginalOracle


def generate_rr_sets(graph, count: int, seed: int):
    """Generate reverse-reachable sets for IC by sampling reverse live edges on demand."""
    rng = random.Random(int(seed))
    nodes = list(graph.nodes())
    rr_sets = []
    t0 = time.perf_counter()
    total_size = 0
    for _ in range(int(count)):
        root = nodes[rng.randrange(len(nodes))]
        reached = {int(root)}
        stack = [int(root)]
        while stack:
            v = stack.pop()
            nbrs = graph.predecessors(v) if graph.is_directed() else graph.neighbors(v)
            for u in nbrs:
                u = int(u)
                if u in reached:
                    continue
                data = graph[u][v] if graph.is_directed() else graph[u][v]
                if rng.random() < float(data.get("weight", 0.0)):
                    reached.add(u)
                    stack.append(u)
        rr_sets.append(reached)
        total_size += len(reached)
    return rr_sets, {
        "rr_sets": int(count),
        "generation_seconds": float(time.perf_counter() - t0),
        "mean_rr_size": float(total_size / max(1, count)),
        "total_rr_memberships": int(total_size),
    }


def singleton_rr_ranking(graph, rr_sets):
    counts = defaultdict(int)
    for rr in rr_sets:
        for v in rr:
            counts[int(v)] += 1
    degree = dict(graph.out_degree()) if graph.is_directed() else dict(graph.degree())
    return sorted(
        (int(v) for v in graph.nodes()),
        key=lambda v: (counts[v], degree.get(v, 0), -v),
        reverse=True,
    ), counts


def rr_greedy(rr_sets, all_nodes, budget: int):
    inv = defaultdict(list)
    for i, rr in enumerate(rr_sets):
        for v in rr:
            inv[int(v)].append(i)
    counts = {int(v): len(inv.get(int(v), ())) for v in all_nodes}
    covered = [False] * len(rr_sets)
    selected = []
    marginal_coverages = []
    for _ in range(int(budget)):
        available = (v for v in all_nodes if v not in selected)
        best = max(available, key=lambda v: (counts[int(v)], -int(v)))
        best = int(best)
        gain = int(counts[best])
        selected.append(best)
        marginal_coverages.append(gain)
        for idx in inv.get(best, ()):
            if covered[idx]:
                continue
            covered[idx] = True
            for u in rr_sets[idx]:
                u = int(u)
                if counts.get(u, 0) > 0:
                    counts[u] -= 1
    return selected, marginal_coverages, int(sum(covered))


def evaluate_seed_set(graph, seeds, eval_mc: int, eval_seed: int):
    t0 = time.perf_counter()
    spread = estimate_spread(graph, list(map(int, seeds)), int(eval_mc), int(eval_seed))
    return {
        "selected_seeds": list(map(int, seeds)),
        "final_spread_mean": float(spread["mean"]),
        "final_spread_std": float(spread["std"]),
        "evaluation_seconds": float(time.perf_counter() - t0),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool-size", type=int, default=128)
    p.add_argument("--budget", type=int, default=10)
    p.add_argument("--rr-sets", type=int, default=10000)
    p.add_argument("--rr-seed", type=int, default=1500401)
    p.add_argument("--oracle-seed", type=int, default=1510401)
    p.add_argument("--eval-seed", type=int, default=1520401)
    p.add_argument("--eval-mc", type=int, default=1000)
    p.add_argument("--audit-top-k", type=int, default=16)
    p.add_argument("--audit-sentinels", type=int, default=8)
    p.add_argument("--audit-mc", type=int, default=20)
    args = p.parse_args()

    graph_data, graph, device, embeddings, norm_degrees, model, degree_pool = build_context(args.pool_size)
    nodes = list(map(int, graph.nodes()))
    print(
        f"device={device} nodes={graph_data.num_nodes} edges={graph_data.num_edges} "
        f"rr_sets={args.rr_sets} pool={args.pool_size} budget={args.budget}", flush=True
    )

    rr_sets, rr_stats = generate_rr_sets(graph, args.rr_sets, args.rr_seed)
    rr_rank, rr_counts = singleton_rr_ranking(graph, rr_sets)
    candidate_pool = rr_rank[: int(args.pool_size)]
    degree_pool_set = set(map(int, degree_pool))
    overlap_degree = len(set(candidate_pool) & degree_pool_set)
    print(
        f"RR screening generated in {rr_stats['generation_seconds']:.2f}s "
        f"mean_size={rr_stats['mean_rr_size']:.3f} overlap_degree128={overlap_degree}/{args.pool_size}",
        flush=True,
    )

    rr_seed_set, rr_marginal, rr_covered = rr_greedy(rr_sets, nodes, args.budget)
    rr_baseline = evaluate_seed_set(graph, rr_seed_set, args.eval_mc, args.eval_seed)
    rr_baseline.update({
        "rr_covered": int(rr_covered),
        "rr_coverage_fraction": float(rr_covered / max(1, len(rr_sets))),
        "rr_marginal_coverages": list(map(int, rr_marginal)),
    })
    rr_seed_recall = len(set(rr_seed_set) & set(candidate_pool)) / max(1, len(rr_seed_set))
    print(
        f"FULLGRAPH_RR_GREEDY spread={rr_baseline['final_spread_mean']:.3f} "
        f"screen_recall_of_rr_seeds={rr_seed_recall:.3f} seeds={rr_seed_set}", flush=True
    )

    degree = evaluate_seed_set(graph, select_high_degree_nodes(graph, args.budget), args.eval_mc, args.eval_seed)
    dd = evaluate_seed_set(graph, select_degree_discount_nodes(graph, args.budget, 0.01), args.eval_mc, args.eval_seed)
    print(f"DEGREE spread={degree['final_spread_mean']:.3f}", flush=True)
    print(f"DEGREE_DISCOUNT spread={dd['final_spread_mean']:.3f}", flush=True)

    ref = run_full_reference(
        graph, candidate_pool, args.budget, args.eval_mc, args.oracle_seed, args.eval_seed
    )
    ref_samples = int(ref["oracle_stats"]["mc_candidate_samples"])
    print(
        f"SCREENED_FULL_MC spread={ref['final_spread_mean']:.3f} samples={ref_samples} "
        f"ratio_vs_fullgraph_rr={ref['final_spread_mean']/rr_baseline['final_spread_mean']:.4f}", flush=True
    )

    learned = LearnedMarginalOracle(model, embeddings, norm_degrees, device)
    exact = ProgressiveMonteCarloOracle(graph, max_mc=40, random_seed=args.oracle_seed)
    t0 = time.perf_counter()
    seeds, steps = audited_residual_greedy(
        candidate_pool,
        args.budget,
        learned,
        exact,
        audit_top_k=args.audit_top_k,
        audit_sentinels=args.audit_sentinels,
        audit_mc=args.audit_mc,
        residual_q=1.0,
        residual_beta=0.0,
    )
    selection_seconds = time.perf_counter() - t0
    audited_eval = evaluate_seed_set(graph, seeds, args.eval_mc, args.eval_seed)
    fallback_steps = sum(int(s["mode"] == "audited_residual_fallback_full_mc40") for s in steps)
    audited = {
        **audited_eval,
        "selection_seconds": float(selection_seconds),
        "fallback_steps": int(fallback_steps),
        "oracle_stats": {
            "candidate_evaluations": int(exact.stats.candidate_evaluations),
            "mc_candidate_samples": int(exact.stats.mc_candidate_samples),
            "live_edge_samples": int(exact.stats.live_edge_samples),
            "learned_evaluations": int(learned.stats.learned_evaluations),
        },
        "steps": steps,
    }
    audited["quality_ratio_vs_screened_full_mc"] = float(
        audited["final_spread_mean"] / ref["final_spread_mean"]
    )
    audited["quality_ratio_vs_fullgraph_rr"] = float(
        audited["final_spread_mean"] / rr_baseline["final_spread_mean"]
    )
    audited["sample_fraction_vs_screened_full_mc"] = float(
        audited["oracle_stats"]["mc_candidate_samples"] / max(1, ref_samples)
    )
    print(
        f"AUDITED spread={audited['final_spread_mean']:.3f} "
        f"ratio_screened_ref={audited['quality_ratio_vs_screened_full_mc']:.4f} "
        f"ratio_fullgraph_rr={audited['quality_ratio_vs_fullgraph_rr']:.4f} "
        f"sample_frac={audited['sample_fraction_vs_screened_full_mc']:.3f} "
        f"fallback={fallback_steps}/{args.budget}", flush=True
    )

    report = {
        "dataset": "NetHEPT",
        "scope": "full-graph RR singleton screening to fixed shortlist, then learning-augmented sequential IM",
        "config": vars(args),
        "screening": {
            **rr_stats,
            "candidate_pool": list(map(int, candidate_pool)),
            "candidate_singleton_rr_counts": {str(v): int(rr_counts[v]) for v in candidate_pool},
            "overlap_with_degree_pool": int(overlap_degree),
            "rr_greedy_seed_recall_in_screened_pool": float(rr_seed_recall),
        },
        "fullgraph_baselines": {
            "rr_sketch_greedy": rr_baseline,
            "degree": degree,
            "degree_discount": dd,
        },
        "screened_full_mc_reference": ref,
        "audited_progressive": audited,
    }
    out = ROOT / "outputs" / "end_to_end" / "fullgraph_rr_screening" / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved={out}", flush=True)


if __name__ == "__main__":
    main()
