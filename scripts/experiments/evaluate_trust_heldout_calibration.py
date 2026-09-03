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
from evaluate_progressive_mc import ProgressiveMonteCarloOracle, paired_confidence
from evaluate_robustness_stress import _rank_spearman
from evaluate_trust_gate_stress import sentinel_nodes
from evaluate_trust_progressive import progressive_select_step
from evaluate_trust_calibration_multiseed import run_full_reference, run_trust_level
from grl.oracle import LearnedMarginalOracle


def mean(xs):
    return float(sum(xs) / len(xs)) if xs else 0.0


def std(xs):
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return float(math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)))


def collect_clean_calibration(
    graph,
    candidate_pool,
    budget,
    model,
    embeddings,
    norm_degrees,
    device,
    repeats,
    base_exact_seed,
    audit_mc,
    audit_top_k,
    sentinels,
    significance_z,
):
    """Collect trust labels on independent Full-MC trajectories.

    Calibration uses only the clean predictor. A fast-path decision is labelled
    unsafe only when the MC40 Full-MC winner is significantly better than the
    progressive candidate under common random numbers. This avoids treating
    statistically indistinguishable MC40 alternatives as trust failures.
    """
    records = []
    for repeat in range(int(repeats)):
        exact_seed = int(base_exact_seed) + repeat * 4001
        exact = ProgressiveMonteCarloOracle(graph, max_mc=40, random_seed=exact_seed)
        base = LearnedMarginalOracle(model, embeddings, norm_degrees, device)
        selected = []

        for step in range(int(budget)):
            available = [v for v in candidate_pool if v not in set(selected)]
            learned = base.score(selected, available, step=step)
            ranked = sorted(available, key=lambda v: (learned[v], -v), reverse=True)

            # Generate the reference using all 40 CRN worlds first. Calls below
            # request prefixes of the same cached worlds, so audit/progressive
            # decisions remain exactly those that would have been seen online.
            full_means, full_samples = exact.score_samples(selected, available, step, 40)
            full_winner = max(available, key=lambda v: (full_means[v], -v))

            head = ranked[: min(int(audit_top_k), len(ranked))]
            sentinel_list = sentinel_nodes(ranked, len(head), int(sentinels))
            audit = list(dict.fromkeys(head + sentinel_list))
            audit_means, _ = exact.score_samples(selected, audit, step, int(audit_mc))
            rho = _rank_spearman({v: learned[v] for v in audit}, audit_means)
            best_head = max(head, key=lambda v: (audit_means[v], -v))
            best_sentinel = None
            sentinel_surprise = False
            sentinel_margin = None
            if sentinel_list:
                best_sentinel = max(sentinel_list, key=lambda v: (audit_means[v], -v))
                sentinel_margin = float(audit_means[best_sentinel] - audit_means[best_head])
                sentinel_surprise = bool(sentinel_margin > 0.0)

            fast_choice, _, verified, final_mc, _, stop_reason = progressive_select_step(
                selected,
                ranked,
                learned,
                exact,
                step,
                sample_budgets=(5, 10, 20, 40),
                initial_m=8,
                batch_m=8,
                residual_beta=0.5,
                confidence_z=0.5,
                bootstrap_mc=10,
            )

            full_gain = float(full_means[full_winner])
            fast_gain = float(full_means[fast_choice])
            regret = max(0.0, full_gain - fast_gain)
            if fast_choice == full_winner:
                winner_significantly_better = False
                paired_diff = 0.0
                paired_se = 0.0
            else:
                winner_significantly_better, paired_diff, paired_se = paired_confidence(
                    full_winner, fast_choice, full_samples, float(significance_z)
                )
            fast_safe = not bool(winner_significantly_better)

            records.append({
                "repeat": int(repeat),
                "step": int(step + 1),
                "exact_seed": int(exact_seed),
                "rho": float(rho),
                "sentinel_surprise": bool(sentinel_surprise),
                "sentinel_margin": sentinel_margin,
                "full_winner": int(full_winner),
                "fast_choice": int(fast_choice),
                "fast_matches_full": bool(fast_choice == full_winner),
                "fast_safe": bool(fast_safe),
                "full_gain_mc40": full_gain,
                "fast_gain_mc40": fast_gain,
                "regret_mc40": float(regret),
                "paired_full_minus_fast_mean": float(paired_diff),
                "paired_se": float(paired_se),
                "verified": int(verified),
                "final_mc": int(final_mc),
                "stop_reason": stop_reason,
            })
            selected.append(int(full_winner))

        print(
            f"CAL repeat={repeat} safe={sum(x['fast_safe'] for x in records if x['repeat']==repeat)}/{budget} "
            f"rho_mean={mean([x['rho'] for x in records if x['repeat']==repeat]):.3f}",
            flush=True,
        )
    return records


def score_threshold(records, tau):
    accepted = [r for r in records if r["rho"] >= float(tau) and not r["sentinel_surprise"]]
    unsafe = [r for r in accepted if not r["fast_safe"]]
    safe = [r for r in records if r["fast_safe"]]
    safe_rejected = [r for r in safe if not (r["rho"] >= float(tau) and not r["sentinel_surprise"])]
    return {
        "tau": float(tau),
        "accepted": len(accepted),
        "coverage": float(len(accepted) / max(1, len(records))),
        "unsafe_accepted": len(unsafe),
        "false_trust_rate": float(len(unsafe) / max(1, len(accepted))),
        "safe_rejected": len(safe_rejected),
        "false_distrust_rate_among_safe": float(len(safe_rejected) / max(1, len(safe))),
    }


def choose_threshold(records, target_false_trust, min_accepted):
    taus = [round(-0.10 + 0.025 * i, 3) for i in range(37)]  # -0.10 ... 0.80
    table = [score_threshold(records, tau) for tau in taus]
    feasible = [
        row for row in table
        if row["accepted"] >= int(min_accepted)
        and row["false_trust_rate"] <= float(target_false_trust)
    ]
    if feasible:
        # Maximize clean fast-path coverage; if tied, prefer the more conservative tau.
        best_coverage = max(row["coverage"] for row in feasible)
        candidates = [row for row in feasible if abs(row["coverage"] - best_coverage) < 1e-12]
        chosen = max(candidates, key=lambda row: row["tau"])
    else:
        # Fallback: minimize false trust, then maximize coverage.
        chosen = min(table, key=lambda row: (row["false_trust_rate"], -row["coverage"], -row["tau"]))
    return chosen, table


def aggregate(items, full_samples):
    return {
        "n": len(items),
        "quality_ratio_mean": mean([x["quality_ratio_vs_full_mc"] for x in items]),
        "quality_ratio_std": std([x["quality_ratio_vs_full_mc"] for x in items]),
        "spread_mean": mean([x["final_spread_mean"] for x in items]),
        "mc_samples_mean": mean([x["oracle_stats"]["mc_candidate_samples"] for x in items]),
        "mc_samples_std": std([x["oracle_stats"]["mc_candidate_samples"] for x in items]),
        "sample_fraction_mean": mean([x["oracle_stats"]["mc_candidate_samples"] / full_samples for x in items]),
        "fallback_steps_mean": mean([x["fallback_steps"] for x in items]),
        "fallback_steps_std": std([x["fallback_steps"] for x in items]),
        "trust_rho_mean": mean([x["mean_trust_rho"] for x in items]),
        "predictor_spearman_mean": mean([x["mean_clean_corrupt_spearman"] for x in items]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool-size", type=int, default=128)
    p.add_argument("--budget", type=int, default=10)
    p.add_argument("--eval-mc", type=int, default=1000)
    p.add_argument("--calibration-repeats", type=int, default=5)
    p.add_argument("--evaluation-repeats", type=int, default=5)
    p.add_argument("--audit-mc", type=int, default=20)
    p.add_argument("--audit-top-k", type=int, default=16)
    p.add_argument("--sentinels", type=int, default=4)
    p.add_argument("--target-false-trust", type=float, default=0.10)
    p.add_argument("--min-accepted", type=int, default=10)
    p.add_argument("--significance-z", type=float, default=1.0)
    p.add_argument("--baseline-tau", type=float, default=0.3)
    p.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.5, 0.75, 1.0])
    p.add_argument("--cal-base-exact-seed", type=int, default=570401)
    p.add_argument("--eval-base-corruption-seed", type=int, default=670401)
    p.add_argument("--eval-base-exact-seed", type=int, default=770401)
    p.add_argument("--eval-base-spread-seed", type=int, default=870401)
    args = p.parse_args()

    graph_data, graph, device, embeddings, norm_degrees, model, candidate_pool = build_context(args.pool_size)
    print(
        f"device={device} nodes={graph_data.num_nodes} edges={graph_data.num_edges} "
        f"pool={len(candidate_pool)} budget={args.budget}",
        flush=True,
    )

    calibration = collect_clean_calibration(
        graph,
        candidate_pool,
        args.budget,
        model,
        embeddings,
        norm_degrees,
        device,
        args.calibration_repeats,
        args.cal_base_exact_seed,
        args.audit_mc,
        args.audit_top_k,
        args.sentinels,
        args.significance_z,
    )
    chosen, threshold_table = choose_threshold(
        calibration, args.target_false_trust, args.min_accepted
    )
    calibrated_tau = float(chosen["tau"])
    print(
        "CALIBRATED",
        f"tau={calibrated_tau:.3f}",
        f"coverage={chosen['coverage']:.3f}",
        f"false_trust={chosen['false_trust_rate']:.3f}",
        f"false_distrust={chosen['false_distrust_rate_among_safe']:.3f}",
        f"accepted={chosen['accepted']}",
        flush=True,
    )

    # Fresh evaluation seeds. Compare the calibrated rule against the old tau=0.3
    # on exactly the same references/randomness.
    full_samples = sum(len(candidate_pool) - step for step in range(args.budget)) * 40
    refs = []
    evaluations = {"baseline": {float(a): [] for a in args.alphas},
                   "calibrated": {float(a): [] for a in args.alphas}}

    for repeat in range(args.evaluation_repeats):
        corruption_seed = args.eval_base_corruption_seed + repeat * 1009
        exact_seed = args.eval_base_exact_seed + repeat * 2003
        spread_seed = args.eval_base_spread_seed + repeat * 3001
        ref = run_full_reference(
            graph, candidate_pool, args.budget, args.eval_mc, exact_seed, spread_seed
        )
        ref["repeat"] = int(repeat)
        refs.append(ref)
        print(f"EVAL REF repeat={repeat} spread={ref['final_spread_mean']:.3f}", flush=True)

        for label, tau in [("baseline", args.baseline_tau), ("calibrated", calibrated_tau)]:
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
                    tau,
                    args.audit_mc,
                    args.sentinels,
                    corruption_seed,
                    exact_seed,
                    spread_seed,
                )
                item["repeat"] = int(repeat)
                item["rule"] = label
                item["quality_ratio_vs_full_mc"] = float(
                    item["final_spread_mean"] / ref["final_spread_mean"]
                )
                item["mean_trust_rho"] = mean(item["trust_rho_per_step"])
                evaluations[label][float(alpha)].append(item)
                print(
                    f"EVAL {label} repeat={repeat} tau={tau:.3f} alpha={alpha:g} "
                    f"fallback={item['fallback_steps']}/{args.budget} "
                    f"ratio={item['quality_ratio_vs_full_mc']:.4f} "
                    f"samples={item['oracle_stats']['mc_candidate_samples']} "
                    f"frac={item['oracle_stats']['mc_candidate_samples']/full_samples:.3f}",
                    flush=True,
                )

    summaries = {}
    for label, groups in evaluations.items():
        summaries[label] = {
            f"alpha_{alpha:g}": aggregate(items, full_samples)
            for alpha, items in groups.items()
        }

    report = {
        "dataset": "NetHEPT",
        "scope": "held-out trust calibration pilot",
        "config": vars(args),
        "calibration": {
            "selection_uses_clean_predictor_only": True,
            "label_definition": (
                "fast path is unsafe only if the same-CRN MC40 Full-MC winner "
                "is significantly better than the progressive candidate"
            ),
            "chosen": chosen,
            "threshold_table": threshold_table,
            "records": calibration,
        },
        "evaluation": {
            "calibrated_tau": calibrated_tau,
            "baseline_tau": float(args.baseline_tau),
            "full_mc_references": refs,
            "summaries": summaries,
            "records": {
                label: {f"alpha_{a:g}": items for a, items in groups.items()}
                for label, groups in evaluations.items()
            },
        },
    }
    out = ROOT / "outputs" / "end_to_end" / "trust_heldout_calibration" / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== HELDOUT SUMMARY ===", flush=True)
    print(f"calibrated_tau={calibrated_tau:.3f}", flush=True)
    for label in ["baseline", "calibrated"]:
        for alpha in args.alphas:
            s = summaries[label][f"alpha_{alpha:g}"]
            print(
                f"{label} alpha={alpha:g} ratio={s['quality_ratio_mean']:.4f}±{s['quality_ratio_std']:.4f} "
                f"samples={s['mc_samples_mean']:.0f}±{s['mc_samples_std']:.0f} "
                f"frac={s['sample_fraction_mean']:.3f} "
                f"fallback={s['fallback_steps_mean']:.2f}±{s['fallback_steps_std']:.2f}",
                flush=True,
            )
    print(f"saved={out}", flush=True)


if __name__ == "__main__":
    main()
