from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

from evaluate_adaptive_certification import build_context
from evaluate_audited_residual_gate import audited_residual_greedy
from evaluate_fullgraph_rr_screening_closure import (
    evaluate_seed_set,
    generate_rr_sets,
    rr_greedy,
    singleton_rr_ranking,
)
from evaluate_progressive_mc import ProgressiveMonteCarloOracle
from evaluate_trust_calibration_multiseed import run_full_reference
from grl.oracle import LearnedMarginalOracle


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool-sizes", nargs="+", type=int, default=[64, 128, 256])
    p.add_argument("--budget", type=int, default=10)
    p.add_argument("--rr-sets", type=int, default=50000)
    p.add_argument("--eval-mc", type=int, default=1000)
    p.add_argument("--screen-seed", type=int, default=2000401)
    p.add_argument("--baseline-rr-seed", type=int, default=2100401)
    p.add_argument("--oracle-seed", type=int, default=2200401)
    p.add_argument("--eval-seed", type=int, default=2300401)
    args = p.parse_args()

    max_pool = max(args.pool_sizes)
    graph_data, graph, device, embeddings, norm_degrees, model, _ = build_context(max_pool)
    nodes = list(map(int, graph.nodes()))
    print(
        f"device={device} nodes={graph_data.num_nodes} edges={graph_data.num_edges} "
        f"rr_sets={args.rr_sets} pools={args.pool_sizes} budget={args.budget}", flush=True
    )

    screen_rr, screen_stats = generate_rr_sets(graph, args.rr_sets, args.screen_seed)
    ranking, counts = singleton_rr_ranking(graph, screen_rr)
    baseline_rr, baseline_stats = generate_rr_sets(graph, args.rr_sets, args.baseline_rr_seed)
    rr_seeds, rr_marginal, rr_covered = rr_greedy(baseline_rr, nodes, args.budget)
    rr_eval = evaluate_seed_set(graph, rr_seeds, args.eval_mc, args.eval_seed)
    print(
        f"INDEPENDENT_RR_BASELINE spread={rr_eval['final_spread_mean']:.3f} seeds={rr_seeds} "
        f"screen_rr_time={screen_stats['generation_seconds']:.3f}s baseline_rr_time={baseline_stats['generation_seconds']:.3f}s",
        flush=True,
    )

    methods = {}
    for m in sorted(set(args.pool_sizes)):
        pool = list(map(int, ranking[:m]))
        recall = len(set(rr_seeds) & set(pool)) / max(1, len(rr_seeds))
        ref = run_full_reference(
            graph, pool, args.budget, args.eval_mc,
            args.oracle_seed + m * 13, args.eval_seed,
        )
        ref_samples = int(ref["oracle_stats"]["mc_candidate_samples"])

        learned = LearnedMarginalOracle(model, embeddings, norm_degrees, device)
        exact = ProgressiveMonteCarloOracle(
            graph, max_mc=40, random_seed=args.oracle_seed + m * 13
        )
        t0 = time.perf_counter()
        seeds, steps = audited_residual_greedy(
            pool, args.budget, learned, exact,
            audit_top_k=16, audit_sentinels=8, audit_mc=20,
            residual_q=1.0, residual_beta=0.0,
        )
        elapsed = time.perf_counter() - t0
        aud_eval = evaluate_seed_set(graph, seeds, args.eval_mc, args.eval_seed)
        fallback = sum(int(s["mode"] == "audited_residual_fallback_full_mc40") for s in steps)
        aud_samples = int(exact.stats.mc_candidate_samples)

        item = {
            "pool_size": int(m),
            "independent_rr_seed_recall": float(recall),
            "screened_pool": pool,
            "screened_full_mc_spread": float(ref["final_spread_mean"]),
            "screened_full_mc_samples": int(ref_samples),
            "screened_vs_independent_rr_ratio": float(ref["final_spread_mean"] / rr_eval["final_spread_mean"]),
            "audited_spread": float(aud_eval["final_spread_mean"]),
            "audited_vs_screened_ratio": float(aud_eval["final_spread_mean"] / ref["final_spread_mean"]),
            "audited_vs_independent_rr_ratio": float(aud_eval["final_spread_mean"] / rr_eval["final_spread_mean"]),
            "audited_samples": int(aud_samples),
            "audited_sample_fraction": float(aud_samples / max(1, ref_samples)),
            "fallback_steps": int(fallback),
            "selection_seconds": float(elapsed),
            "screened_full_mc_seeds": list(map(int, ref["selected_seeds"])),
            "audited_seeds": list(map(int, seeds)),
        }
        methods[str(m)] = item
        print(
            f"POOL {m} recall={recall:.2f} ref={item['screened_full_mc_spread']:.3f} "
            f"ref/rr={item['screened_vs_independent_rr_ratio']:.4f} "
            f"aud={item['audited_spread']:.3f} aud/ref={item['audited_vs_screened_ratio']:.4f} "
            f"aud/rr={item['audited_vs_independent_rr_ratio']:.4f} "
            f"cost={item['audited_sample_fraction']:.3f} fallback={fallback}/{args.budget}",
            flush=True,
        )

    report = {
        "dataset": "NetHEPT",
        "scope": "full-graph independent RR shortlist budget sensitivity",
        "config": vars(args),
        "screening": {
            "rr_sets": int(args.rr_sets),
            "screening_seconds": float(screen_stats["generation_seconds"]),
            "mean_rr_size": float(screen_stats["mean_rr_size"]),
            "top_singleton_counts": {str(v): int(counts[v]) for v in ranking[:max_pool]},
        },
        "independent_rr_baseline": {
            **rr_eval,
            "rr_seeds": list(map(int, rr_seeds)),
            "rr_marginal_coverages": list(map(int, rr_marginal)),
            "rr_covered": int(rr_covered),
        },
        "methods": methods,
    }
    out = ROOT / "outputs" / "end_to_end" / "fullgraph_rr_pool_sensitivity" / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved={out}", flush=True)


if __name__ == "__main__":
    main()
