from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

from evaluate_adaptive_certification import build_context
from evaluate_audited_residual_gate import run_level
from evaluate_trust_calibration_multiseed import run_full_reference


def mean(xs):
    return float(sum(xs) / len(xs)) if xs else 0.0


def std(xs):
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return float(math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)))


def summarize(items, full_samples):
    return {
        "n": len(items),
        "spread_mean": mean([x["final_spread_mean"] for x in items]),
        "spread_std": std([x["final_spread_mean"] for x in items]),
        "quality_ratio_mean": mean([x["quality_ratio_vs_full_mc"] for x in items]),
        "quality_ratio_std": std([x["quality_ratio_vs_full_mc"] for x in items]),
        "fallback_steps_mean": mean([x["fallback_steps"] for x in items]),
        "fallback_steps_std": std([x["fallback_steps"] for x in items]),
        "mc_candidate_samples_mean": mean([x["oracle_stats"]["mc_candidate_samples"] for x in items]),
        "mc_candidate_samples_std": std([x["oracle_stats"]["mc_candidate_samples"] for x in items]),
        "sample_fraction_vs_full_mean": mean([x["oracle_stats"]["mc_candidate_samples"] / full_samples for x in items]),
        "sample_fraction_vs_full_std": std([x["oracle_stats"]["mc_candidate_samples"] / full_samples for x in items]),
        "candidate_evaluations_mean": mean([x["oracle_stats"]["candidate_evaluations"] for x in items]),
        "mean_clean_corrupt_spearman": mean([x["mean_clean_corrupt_spearman"] for x in items]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool-size", type=int, default=128)
    p.add_argument("--budget", type=int, default=10)
    p.add_argument("--eval-mc", type=int, default=1000)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.5, 0.75, 1.0])
    p.add_argument("--residual-q", type=float, default=1.0)
    p.add_argument("--residual-beta", type=float, default=0.0)
    p.add_argument("--audit-mc", type=int, default=20)
    p.add_argument("--sentinels", type=int, default=8)
    p.add_argument("--base-corruption-seed", type=int, default=970401)
    p.add_argument("--base-exact-seed", type=int, default=980401)
    p.add_argument("--base-eval-seed", type=int, default=990401)
    args = p.parse_args()

    graph_data, graph, device, embeddings, norm_degrees, model, candidate_pool = build_context(args.pool_size)
    full_samples = sum(len(candidate_pool) - step for step in range(args.budget)) * 40
    print(
        f"device={device} nodes={graph_data.num_nodes} edges={graph_data.num_edges} "
        f"pool={len(candidate_pool)} budget={args.budget} repeats={args.repeats}", flush=True
    )

    references = []
    records = []
    by_alpha = {float(a): [] for a in args.alphas}

    for repeat in range(args.repeats):
        corruption_seed = args.base_corruption_seed + repeat * 1009
        exact_seed = args.base_exact_seed + repeat * 2003
        eval_seed = args.base_eval_seed + repeat * 3001

        ref = run_full_reference(
            graph, candidate_pool, args.budget, args.eval_mc, exact_seed, eval_seed
        )
        ref.update({"repeat": repeat, "exact_seed": exact_seed, "eval_seed": eval_seed})
        references.append(ref)
        print(
            f"REF repeat={repeat} spread={ref['final_spread_mean']:.3f} "
            f"samples={ref['oracle_stats']['mc_candidate_samples']}", flush=True
        )

        for alpha in args.alphas:
            item = run_level(
                graph, candidate_pool, args.budget, args.eval_mc,
                model, embeddings, norm_degrees, device,
                alpha, args.residual_q, args.residual_beta,
                args.audit_mc, args.sentinels,
                corruption_seed, exact_seed, eval_seed,
            )
            item["repeat"] = repeat
            item["full_mc_spread_same_seed"] = float(ref["final_spread_mean"])
            item["quality_ratio_vs_full_mc"] = float(item["final_spread_mean"] / ref["final_spread_mean"])
            item["sample_fraction_vs_full_mc"] = float(item["oracle_stats"]["mc_candidate_samples"] / full_samples)
            records.append(item)
            by_alpha[float(alpha)].append(item)
            print(
                f"RUN repeat={repeat} alpha={alpha:g} "
                f"pred_rho={item['mean_clean_corrupt_spearman']:.3f} "
                f"fallback={item['fallback_steps']}/{args.budget} "
                f"spread={item['final_spread_mean']:.3f} "
                f"ratio={item['quality_ratio_vs_full_mc']:.4f} "
                f"samples={item['oracle_stats']['mc_candidate_samples']} "
                f"frac={item['sample_fraction_vs_full_mc']:.3f}", flush=True
            )

    ref_spreads = [x["final_spread_mean"] for x in references]
    ref_summary = {
        "n": len(references),
        "spread_mean": mean(ref_spreads),
        "spread_std": std(ref_spreads),
        "mc_candidate_samples": full_samples,
    }
    summaries = {f"alpha_{a:g}": summarize(xs, full_samples) for a, xs in by_alpha.items()}

    out = ROOT / "outputs" / "end_to_end" / "audited_residual_multiseed" / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "dataset": "NetHEPT",
        "scope": "five-seed audited-residual gate validation on shared 128-candidate pool",
        "config": vars(args),
        "full_mc_reference_summary": ref_summary,
        "summaries": summaries,
        "references": references,
        "records": records,
    }, indent=2), encoding="utf-8")

    print("=== AGGREGATE ===", flush=True)
    print(f"full_mc spread={ref_summary['spread_mean']:.3f}±{ref_summary['spread_std']:.3f}", flush=True)
    for key, s in summaries.items():
        print(
            f"{key} pred_rho={s['mean_clean_corrupt_spearman']:.3f} "
            f"fallback={s['fallback_steps_mean']:.2f}±{s['fallback_steps_std']:.2f} "
            f"spread={s['spread_mean']:.3f}±{s['spread_std']:.3f} "
            f"ratio={s['quality_ratio_mean']:.4f}±{s['quality_ratio_std']:.4f} "
            f"samples={s['mc_candidate_samples_mean']:.0f}±{s['mc_candidate_samples_std']:.0f} "
            f"sample_frac={s['sample_fraction_vs_full_mean']:.3f}±{s['sample_fraction_vs_full_std']:.3f}",
            flush=True,
        )
    print(f"saved={out}", flush=True)


if __name__ == "__main__":
    main()
