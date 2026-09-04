from __future__ import annotations

import argparse
import json
import statistics
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
from grl.baselines import select_degree_discount_nodes, select_high_degree_nodes
from grl.oracle import LearnedMarginalOracle


def mean(xs):
    return float(statistics.mean(xs)) if xs else 0.0


def std(xs):
    return float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0


def agg(records, key):
    vals = [float(r[key]) for r in records]
    return {"mean": mean(vals), "std": std(vals), "values": vals}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool-size", type=int, default=128)
    p.add_argument("--budget", type=int, default=10)
    p.add_argument("--rr-sets", type=int, default=50000)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--eval-mc", type=int, default=1000)
    p.add_argument("--screen-seed", type=int, default=1600401)
    p.add_argument("--baseline-rr-seed", type=int, default=1700401)
    p.add_argument("--oracle-seed", type=int, default=1800401)
    p.add_argument("--eval-seed", type=int, default=1900401)
    args = p.parse_args()

    graph_data, graph, device, embeddings, norm_degrees, model, degree_pool = build_context(args.pool_size)
    nodes = list(map(int, graph.nodes()))
    degree_pool = set(map(int, degree_pool))
    print(
        f"device={device} nodes={graph_data.num_nodes} edges={graph_data.num_edges} "
        f"rr_sets={args.rr_sets} repeats={args.repeats} pool={args.pool_size}", flush=True
    )

    degree_seeds = select_high_degree_nodes(graph, args.budget)
    dd_seeds = select_degree_discount_nodes(graph, args.budget, 0.01)
    records = []
    candidate_sets = []

    for rep in range(args.repeats):
        screen_seed = args.screen_seed + rep * 1009
        baseline_seed = args.baseline_rr_seed + rep * 1013
        oracle_seed = args.oracle_seed + rep * 1019
        eval_seed = args.eval_seed + rep * 1021

        screen_rr, screen_stats = generate_rr_sets(graph, args.rr_sets, screen_seed)
        rr_rank, _ = singleton_rr_ranking(graph, screen_rr)
        pool = list(map(int, rr_rank[: args.pool_size]))
        pool_set = set(pool)
        candidate_sets.append(pool_set)

        baseline_rr, baseline_stats = generate_rr_sets(graph, args.rr_sets, baseline_seed)
        rr_seeds, rr_marginal, rr_covered = rr_greedy(baseline_rr, nodes, args.budget)
        rr_eval = evaluate_seed_set(graph, rr_seeds, args.eval_mc, eval_seed)
        degree_eval = evaluate_seed_set(graph, degree_seeds, args.eval_mc, eval_seed)
        dd_eval = evaluate_seed_set(graph, dd_seeds, args.eval_mc, eval_seed)

        independent_rr_seed_recall = len(set(rr_seeds) & pool_set) / max(1, len(rr_seeds))
        overlap_degree = len(pool_set & degree_pool)

        ref = run_full_reference(graph, pool, args.budget, args.eval_mc, oracle_seed, eval_seed)
        ref_samples = int(ref["oracle_stats"]["mc_candidate_samples"])

        learned = LearnedMarginalOracle(model, embeddings, norm_degrees, device)
        exact = ProgressiveMonteCarloOracle(graph, max_mc=40, random_seed=oracle_seed)
        t0 = time.perf_counter()
        seeds, steps = audited_residual_greedy(
            pool, args.budget, learned, exact,
            audit_top_k=16, audit_sentinels=8, audit_mc=20,
            residual_q=1.0, residual_beta=0.0,
        )
        selection_seconds = time.perf_counter() - t0
        audited_eval = evaluate_seed_set(graph, seeds, args.eval_mc, eval_seed)
        fallback = sum(int(s["mode"] == "audited_residual_fallback_full_mc40") for s in steps)
        audited_samples = int(exact.stats.mc_candidate_samples)

        rec = {
            "repeat": rep,
            "seeds": {
                "screen": screen_seed,
                "baseline_rr": baseline_seed,
                "oracle": oracle_seed,
                "eval": eval_seed,
            },
            "screening_seconds": float(screen_stats["generation_seconds"]),
            "screen_mean_rr_size": float(screen_stats["mean_rr_size"]),
            "baseline_rr_seconds": float(baseline_stats["generation_seconds"]),
            "overlap_with_degree_pool": int(overlap_degree),
            "independent_rr_seed_recall_in_pool": float(independent_rr_seed_recall),
            "candidate_pool": pool,
            "rr_baseline_seeds": list(map(int, rr_seeds)),
            "rr_baseline_spread": float(rr_eval["final_spread_mean"]),
            "degree_spread": float(degree_eval["final_spread_mean"]),
            "degree_discount_spread": float(dd_eval["final_spread_mean"]),
            "screened_full_mc_seeds": list(map(int, ref["selected_seeds"])),
            "screened_full_mc_spread": float(ref["final_spread_mean"]),
            "screened_full_mc_samples": int(ref_samples),
            "screened_vs_rr_ratio": float(ref["final_spread_mean"] / rr_eval["final_spread_mean"]),
            "audited_seeds": list(map(int, seeds)),
            "audited_spread": float(audited_eval["final_spread_mean"]),
            "audited_vs_screened_ratio": float(audited_eval["final_spread_mean"] / ref["final_spread_mean"]),
            "audited_vs_rr_ratio": float(audited_eval["final_spread_mean"] / rr_eval["final_spread_mean"]),
            "audited_samples": audited_samples,
            "audited_sample_fraction": float(audited_samples / max(1, ref_samples)),
            "fallback_steps": int(fallback),
            "selection_seconds": float(selection_seconds),
        }
        records.append(rec)
        print(
            f"REP {rep} recall={independent_rr_seed_recall:.2f} overlap_deg={overlap_degree}/{args.pool_size} "
            f"rr={rec['rr_baseline_spread']:.3f} ref={rec['screened_full_mc_spread']:.3f} "
            f"aud={rec['audited_spread']:.3f} q={rec['audited_vs_screened_ratio']:.4f} "
            f"cost={rec['audited_sample_fraction']:.3f} fallback={fallback}/{args.budget}", flush=True
        )

    jaccards = []
    for i in range(len(candidate_sets)):
        for j in range(i + 1, len(candidate_sets)):
            a, b = candidate_sets[i], candidate_sets[j]
            jaccards.append(len(a & b) / max(1, len(a | b)))

    summary = {
        "independent_rr_seed_recall_in_pool": agg(records, "independent_rr_seed_recall_in_pool"),
        "overlap_with_degree_pool": agg(records, "overlap_with_degree_pool"),
        "rr_baseline_spread": agg(records, "rr_baseline_spread"),
        "screened_full_mc_spread": agg(records, "screened_full_mc_spread"),
        "screened_vs_rr_ratio": agg(records, "screened_vs_rr_ratio"),
        "audited_spread": agg(records, "audited_spread"),
        "audited_vs_screened_ratio": agg(records, "audited_vs_screened_ratio"),
        "audited_vs_rr_ratio": agg(records, "audited_vs_rr_ratio"),
        "audited_sample_fraction": agg(records, "audited_sample_fraction"),
        "fallback_steps": agg(records, "fallback_steps"),
        "candidate_pool_pairwise_jaccard": {"mean": mean(jaccards), "std": std(jaccards), "values": jaccards},
    }
    print("=== AGGREGATE ===", flush=True)
    for k in (
        "independent_rr_seed_recall_in_pool", "rr_baseline_spread", "screened_full_mc_spread",
        "audited_spread", "audited_vs_screened_ratio", "audited_vs_rr_ratio",
        "audited_sample_fraction", "fallback_steps", "candidate_pool_pairwise_jaccard",
    ):
        x = summary[k]
        print(f"{k}: {x['mean']:.4f} +/- {x['std']:.4f}", flush=True)

    report = {
        "dataset": "NetHEPT",
        "scope": "independent RR screening and RR baseline, multi-seed full-graph closure",
        "config": vars(args),
        "records": records,
        "summary": summary,
    }
    out = ROOT / "outputs" / "end_to_end" / "fullgraph_rr_screening_multiseed" / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved={out}", flush=True)


if __name__ == "__main__":
    main()
