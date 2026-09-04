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
from evaluate_audited_residual_gate import audited_residual_greedy
from evaluate_progressive_mc import ProgressiveMonteCarloOracle
from evaluate_robustness_stress import CorruptedLearnedOracle
from evaluate_trust_calibration_multiseed import run_full_reference
from grl.diffusion import estimate_spread
from grl.oracle import LearnedMarginalOracle


def mean(xs):
    return float(sum(xs) / len(xs)) if xs else 0.0


def std(xs):
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return float(math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)))


def run_level(
    graph,
    candidate_pool,
    budget,
    eval_mc,
    model,
    embeddings,
    norm_degrees,
    device,
    alpha,
    audit_top_k,
    sentinels,
    audit_mc,
    residual_q,
    residual_beta,
    corruption_seed,
    exact_seed,
    eval_seed,
):
    base = LearnedMarginalOracle(model, embeddings, norm_degrees, device)
    learned = CorruptedLearnedOracle(base, alpha=float(alpha), random_seed=int(corruption_seed))
    exact = ProgressiveMonteCarloOracle(graph, max_mc=40, random_seed=int(exact_seed))

    t0 = time.perf_counter()
    seeds, steps = audited_residual_greedy(
        candidate_pool,
        budget,
        learned,
        exact,
        audit_top_k=int(audit_top_k),
        audit_sentinels=int(sentinels),
        audit_mc=int(audit_mc),
        residual_q=float(residual_q),
        residual_beta=float(residual_beta),
    )
    elapsed = time.perf_counter() - t0
    spread = estimate_spread(graph, seeds, eval_mc, int(eval_seed))
    modes = Counter(x["mode"] for x in steps)

    return {
        "alpha": float(alpha),
        "audit_top_k": int(audit_top_k),
        "sentinels": int(sentinels),
        "residual_q": float(residual_q),
        "residual_beta": float(residual_beta),
        "mean_clean_corrupt_spearman": mean(learned.correlations),
        "selected_seeds": [int(v) for v in seeds],
        "final_spread_mean": float(spread["mean"]),
        "final_spread_std": float(spread["std"]),
        "selection_seconds": float(elapsed),
        "fallback_steps": int(modes.get("audited_residual_fallback_full_mc40", 0)),
        "oracle_stats": {
            "candidate_evaluations": int(exact.stats.candidate_evaluations),
            "mc_candidate_samples": int(exact.stats.mc_candidate_samples),
            "live_edge_samples": int(exact.stats.live_edge_samples),
            "learned_evaluations": int(base.stats.learned_evaluations),
        },
        "steps": steps,
    }


def summarize(items):
    return {
        "n": len(items),
        "quality_ratio_mean": mean([x["quality_ratio_vs_full_mc"] for x in items]),
        "quality_ratio_std": std([x["quality_ratio_vs_full_mc"] for x in items]),
        "sample_fraction_mean": mean([x["sample_fraction_vs_full_mc"] for x in items]),
        "sample_fraction_std": std([x["sample_fraction_vs_full_mc"] for x in items]),
        "fallback_steps_mean": mean([x["fallback_steps"] for x in items]),
        "fallback_steps_std": std([x["fallback_steps"] for x in items]),
        "spread_mean": mean([x["final_spread_mean"] for x in items]),
        "spread_std": std([x["final_spread_mean"] for x in items]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool-size", type=int, required=True)
    p.add_argument("--budget", type=int, default=10)
    p.add_argument("--eval-mc", type=int, default=1000)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--alphas", nargs="+", type=float, default=[0.0, 1.0])
    p.add_argument("--audit-top-k", type=int, required=True)
    p.add_argument("--sentinels", type=int, required=True)
    p.add_argument("--audit-mc", type=int, default=20)
    p.add_argument("--residual-q", type=float, default=1.0)
    p.add_argument("--residual-beta", type=float, default=0.0)
    p.add_argument("--base-corruption-seed", type=int, default=1070401)
    p.add_argument("--base-exact-seed", type=int, default=1080401)
    p.add_argument("--base-eval-seed", type=int, default=1090401)
    args = p.parse_args()

    graph_data, graph, device, embeddings, norm_degrees, model, candidate_pool = build_context(args.pool_size)
    full_samples = sum(len(candidate_pool) - step for step in range(args.budget)) * 40
    print(
        f"device={device} nodes={graph_data.num_nodes} edges={graph_data.num_edges} "
        f"pool={len(candidate_pool)} top={args.audit_top_k} sent={args.sentinels} "
        f"budget={args.budget} repeats={args.repeats}",
        flush=True,
    )

    references = []
    records = []
    by_alpha = {float(a): [] for a in args.alphas}

    for repeat in range(args.repeats):
        corruption_seed = args.base_corruption_seed + repeat * 1009
        exact_seed = args.base_exact_seed + repeat * 2003
        eval_seed = args.base_eval_seed + repeat * 3001

        ref = run_full_reference(graph, candidate_pool, args.budget, args.eval_mc, exact_seed, eval_seed)
        references.append(ref)
        print(
            f"REF repeat={repeat} spread={ref['final_spread_mean']:.3f} "
            f"samples={ref['oracle_stats']['mc_candidate_samples']}",
            flush=True,
        )

        for alpha in args.alphas:
            item = run_level(
                graph,
                candidate_pool,
                args.budget,
                args.eval_mc,
                model,
                embeddings,
                norm_degrees,
                device,
                alpha,
                args.audit_top_k,
                args.sentinels,
                args.audit_mc,
                args.residual_q,
                args.residual_beta,
                corruption_seed,
                exact_seed,
                eval_seed,
            )
            item["repeat"] = int(repeat)
            item["full_mc_spread_same_seed"] = float(ref["final_spread_mean"])
            item["quality_ratio_vs_full_mc"] = float(item["final_spread_mean"] / ref["final_spread_mean"])
            item["sample_fraction_vs_full_mc"] = float(item["oracle_stats"]["mc_candidate_samples"] / full_samples)
            records.append(item)
            by_alpha[float(alpha)].append(item)
            print(
                f"RUN repeat={repeat} alpha={alpha:g} "
                f"rho={item['mean_clean_corrupt_spearman']:.3f} "
                f"fallback={item['fallback_steps']}/{args.budget} "
                f"spread={item['final_spread_mean']:.3f} "
                f"ratio={item['quality_ratio_vs_full_mc']:.4f} "
                f"samples={item['oracle_stats']['mc_candidate_samples']} "
                f"frac={item['sample_fraction_vs_full_mc']:.3f}",
                flush=True,
            )

    summaries = {f"alpha_{a:g}": summarize(xs) for a, xs in by_alpha.items()}
    print("=== AGGREGATE ===", flush=True)
    for key, s in summaries.items():
        print(
            f"{key} ratio={s['quality_ratio_mean']:.4f}±{s['quality_ratio_std']:.4f} "
            f"sample_frac={s['sample_fraction_mean']:.3f}±{s['sample_fraction_std']:.3f} "
            f"fallback={s['fallback_steps_mean']:.2f}±{s['fallback_steps_std']:.2f}",
            flush=True,
        )

    out = (
        ROOT
        / "outputs"
        / "end_to_end"
        / "audited_residual_scaleaware"
        / f"pool{args.pool_size}_top{args.audit_top_k}_sent{args.sentinels}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "dataset": "NetHEPT",
                "scope": "scale-aware audited-residual coverage pilot",
                "config": vars(args),
                "full_mc_candidate_samples": int(full_samples),
                "summaries": summaries,
                "references": references,
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved={out}", flush=True)


if __name__ == "__main__":
    main()
