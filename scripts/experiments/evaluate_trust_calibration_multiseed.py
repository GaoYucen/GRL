from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

from evaluate_adaptive_certification import build_context
from evaluate_progressive_mc import ProgressiveMonteCarloOracle
from evaluate_robustness_stress import CorruptedLearnedOracle
from evaluate_trust_progressive import trust_progressive_greedy
from grl.diffusion import estimate_spread
from grl.oracle import LearnedMarginalOracle


def _mean(xs):
    return float(sum(xs) / len(xs)) if xs else 0.0


def _std(xs):
    if len(xs) <= 1:
        return 0.0
    m = _mean(xs)
    return float(math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)))


def full_mc_greedy_progressive(candidate_pool, budget, exact_oracle):
    selected = []
    for step in range(int(budget)):
        available = [v for v in candidate_pool if v not in set(selected)]
        if not available:
            break
        means, _ = exact_oracle.score_samples(selected, available, step, 40)
        chosen = max(available, key=lambda v: (means[v], -v))
        selected.append(int(chosen))
    return selected


def run_full_reference(graph, candidate_pool, budget, eval_mc, exact_seed, eval_seed):
    exact = ProgressiveMonteCarloOracle(graph, max_mc=40, random_seed=int(exact_seed))
    t0 = time.perf_counter()
    seeds = full_mc_greedy_progressive(candidate_pool, budget, exact)
    elapsed = time.perf_counter() - t0
    spread = estimate_spread(graph, seeds, eval_mc, int(eval_seed))
    return {
        "selected_seeds": seeds,
        "final_spread_mean": float(spread["mean"]),
        "final_spread_std": float(spread["std"]),
        "selection_seconds": float(elapsed),
        "oracle_stats": {
            "candidate_evaluations": int(exact.stats.candidate_evaluations),
            "mc_candidate_samples": int(exact.stats.mc_candidate_samples),
            "live_edge_samples": int(exact.stats.live_edge_samples),
        },
    }


def run_trust_level(
    graph,
    candidate_pool,
    budget,
    eval_mc,
    model,
    embeddings,
    norm_degrees,
    device,
    alpha,
    tau,
    audit_mc,
    sentinels,
    corruption_seed,
    exact_seed,
    eval_seed,
):
    base = LearnedMarginalOracle(model, embeddings, norm_degrees, device)
    learned = CorruptedLearnedOracle(base, alpha=float(alpha), random_seed=int(corruption_seed))
    exact = ProgressiveMonteCarloOracle(graph, max_mc=40, random_seed=int(exact_seed))
    t0 = time.perf_counter()
    seeds, steps = trust_progressive_greedy(
        candidate_pool,
        budget,
        learned,
        exact,
        trust_tau=float(tau),
        audit_top_k=16,
        audit_sentinels=int(sentinels),
        audit_mc=int(audit_mc),
    )
    elapsed = time.perf_counter() - t0
    spread = estimate_spread(graph, seeds, eval_mc, int(eval_seed))
    modes = Counter(x["mode"] for x in steps)
    fallback_steps = int(modes.get("trust_fallback_full_mc40", 0))
    return {
        "alpha": float(alpha),
        "trust_tau": float(tau),
        "audit_mc": int(audit_mc),
        "audit_sentinels": int(sentinels),
        "corruption_seed": int(corruption_seed),
        "exact_seed": int(exact_seed),
        "eval_seed": int(eval_seed),
        "mean_clean_corrupt_spearman": _mean(learned.correlations),
        "selected_seeds": [int(v) for v in seeds],
        "final_spread_mean": float(spread["mean"]),
        "final_spread_std": float(spread["std"]),
        "selection_seconds": float(elapsed),
        "fallback_steps": fallback_steps,
        "fallback_fraction": float(fallback_steps / max(1, len(steps))),
        "oracle_stats": {
            "candidate_evaluations": int(exact.stats.candidate_evaluations),
            "mc_candidate_samples": int(exact.stats.mc_candidate_samples),
            "live_edge_samples": int(exact.stats.live_edge_samples),
            "learned_evaluations": int(base.stats.learned_evaluations),
        },
        "mode_counts": dict(modes),
        "trust_rho_per_step": [float(x["trust_rho"]) for x in steps],
        "trusted_per_step": [bool(x["trusted"]) for x in steps],
        "sentinel_surprise_per_step": [bool(x["sentinel_surprise"]) for x in steps],
        "verified_per_step": [int(x["verified"]) for x in steps],
        "final_mc_per_step": [int(x["final_mc"]) for x in steps],
    }


def summarize(items, full_samples):
    spreads = [x["final_spread_mean"] for x in items]
    ratios = [x["quality_ratio_vs_full_mc"] for x in items]
    samples = [x["oracle_stats"]["mc_candidate_samples"] for x in items]
    candidates = [x["oracle_stats"]["candidate_evaluations"] for x in items]
    fallbacks = [x["fallback_steps"] for x in items]
    trust = [x["mean_trust_rho"] for x in items]
    pred_rho = [x["mean_clean_corrupt_spearman"] for x in items]
    return {
        "n": len(items),
        "spread_mean": _mean(spreads),
        "spread_std_across_runs": _std(spreads),
        "quality_ratio_mean": _mean(ratios),
        "quality_ratio_std": _std(ratios),
        "candidate_evaluations_mean": _mean(candidates),
        "candidate_evaluations_std": _std(candidates),
        "mc_candidate_samples_mean": _mean(samples),
        "mc_candidate_samples_std": _std(samples),
        "sample_fraction_vs_full_mean": _mean([x / full_samples for x in samples]),
        "fallback_steps_mean": _mean(fallbacks),
        "fallback_steps_std": _std(fallbacks),
        "mean_trust_rho": _mean(trust),
        "mean_clean_corrupt_spearman": _mean(pred_rho),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool-size", type=int, default=128)
    p.add_argument("--budget", type=int, default=10)
    p.add_argument("--eval-mc", type=int, default=1000)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.5, 0.75, 1.0])
    p.add_argument("--tau", type=float, default=0.3)
    p.add_argument("--audit-mc", type=int, default=20)
    p.add_argument("--sentinels", type=int, default=4)
    p.add_argument("--base-corruption-seed", type=int, default=270401)
    p.add_argument("--base-exact-seed", type=int, default=370401)
    p.add_argument("--base-eval-seed", type=int, default=470401)
    args = p.parse_args()

    graph_data, graph, device, embeddings, norm_degrees, model, candidate_pool = build_context(args.pool_size)
    print(
        f"device={device} nodes={graph_data.num_nodes} edges={graph_data.num_edges} "
        f"pool={len(candidate_pool)} budget={args.budget} repeats={args.repeats}",
        flush=True,
    )

    records = []
    references = []
    by_alpha = {float(a): [] for a in args.alphas}
    full_samples = sum(len(candidate_pool) - step for step in range(args.budget)) * 40

    for repeat in range(args.repeats):
        corruption_seed = args.base_corruption_seed + repeat * 1009
        exact_seed = args.base_exact_seed + repeat * 2003
        eval_seed = args.base_eval_seed + repeat * 3001

        ref = run_full_reference(
            graph, candidate_pool, args.budget, args.eval_mc, exact_seed, eval_seed
        )
        ref["repeat"] = repeat
        ref["exact_seed"] = exact_seed
        ref["eval_seed"] = eval_seed
        references.append(ref)
        print(
            f"REF repeat={repeat} spread={ref['final_spread_mean']:.3f} "
            f"samples={ref['oracle_stats']['mc_candidate_samples']} "
            f"time={ref['selection_seconds']:.2f}",
            flush=True,
        )

        for alpha in args.alphas:
            item = run_trust_level(
                graph,
                candidate_pool,
                args.budget,
                args.eval_mc,
                model,
                embeddings,
                norm_degrees,
                device,
                alpha,
                args.tau,
                args.audit_mc,
                args.sentinels,
                corruption_seed,
                exact_seed,
                eval_seed,
            )
            item["repeat"] = repeat
            item["full_mc_spread_same_seed"] = float(ref["final_spread_mean"])
            item["quality_ratio_vs_full_mc"] = float(
                item["final_spread_mean"] / ref["final_spread_mean"]
            )
            item["sample_fraction_vs_full_mc"] = float(
                item["oracle_stats"]["mc_candidate_samples"] / full_samples
            )
            item["mean_trust_rho"] = _mean(item["trust_rho_per_step"])
            records.append(item)
            by_alpha[float(alpha)].append(item)
            print(
                f"RUN repeat={repeat} alpha={alpha:g} "
                f"pred_rho={item['mean_clean_corrupt_spearman']:.3f} "
                f"trust_rho={item['mean_trust_rho']:.3f} "
                f"fallback={item['fallback_steps']}/{args.budget} "
                f"spread={item['final_spread_mean']:.3f} "
                f"ratio={item['quality_ratio_vs_full_mc']:.4f} "
                f"samples={item['oracle_stats']['mc_candidate_samples']} "
                f"sample_frac={item['sample_fraction_vs_full_mc']:.3f}",
                flush=True,
            )

    summaries = {
        f"alpha_{alpha:g}": summarize(items, full_samples)
        for alpha, items in by_alpha.items()
    }
    ref_spreads = [x["final_spread_mean"] for x in references]
    ref_summary = {
        "n": len(references),
        "spread_mean": _mean(ref_spreads),
        "spread_std_across_runs": _std(ref_spreads),
        "mc_candidate_samples": full_samples,
    }

    report = {
        "dataset": "NetHEPT",
        "scope": "multi-seed robustness/calibration pilot on shared 128-candidate pool",
        "config": vars(args),
        "full_mc_reference_summary": ref_summary,
        "summaries": summaries,
        "full_mc_references": references,
        "records": records,
    }
    out = ROOT / "outputs" / "end_to_end" / "trust_calibration_multiseed" / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== AGGREGATE ===", flush=True)
    print(
        f"full_mc spread={ref_summary['spread_mean']:.3f}±{ref_summary['spread_std_across_runs']:.3f}",
        flush=True,
    )
    for key, s in summaries.items():
        print(
            f"{key} pred_rho={s['mean_clean_corrupt_spearman']:.3f} "
            f"trust_rho={s['mean_trust_rho']:.3f} "
            f"fallback={s['fallback_steps_mean']:.2f}±{s['fallback_steps_std']:.2f} "
            f"spread={s['spread_mean']:.3f}±{s['spread_std_across_runs']:.3f} "
            f"ratio={s['quality_ratio_mean']:.4f}±{s['quality_ratio_std']:.4f} "
            f"samples={s['mc_candidate_samples_mean']:.0f}±{s['mc_candidate_samples_std']:.0f} "
            f"sample_frac={s['sample_fraction_vs_full_mean']:.3f}",
            flush=True,
        )
    print(f"saved={out}", flush=True)


if __name__ == "__main__":
    main()
