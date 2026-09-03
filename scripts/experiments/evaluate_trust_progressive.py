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
from evaluate_progressive_mc import ProgressiveMonteCarloOracle, paired_confidence
from evaluate_robustness_stress import CorruptedLearnedOracle, _rank_spearman
from evaluate_trust_gate_stress import sentinel_nodes
from grl.diffusion import estimate_spread
from grl.oracle import LearnedMarginalOracle


def progressive_select_step(
    selected: list[int],
    ranked: list[int],
    learned: dict[int, float],
    exact: ProgressiveMonteCarloOracle,
    step: int,
    sample_budgets=(5, 10, 20, 40),
    initial_m: int = 8,
    batch_m: int = 8,
    residual_beta: float = 0.5,
    confidence_z: float = 0.5,
    bootstrap_mc: int = 10,
):
    """Select one node using the validated progressive-v3 fast path.

    The exact oracle is shared with the trust audit, so candidate/world samples
    obtained during auditing are reused automatically by the oracle cache.
    """
    budgets = sorted({int(x) for x in sample_budgets if int(x) > 0})
    max_mc = budgets[-1]
    bootstrap_mc = max(1, min(int(bootstrap_mc), max_mc))
    bootstrap_n = max([x for x in budgets if x <= bootstrap_mc] or [budgets[0]])

    target = min(int(initial_m), len(ranked))
    previous_target_winner = None
    rounds = []
    chosen = None
    stop_reason = None
    last_means = None
    final_n = max_mc

    while True:
        verified_nodes = ranked[:target]

        if previous_target_winner is None and target < len(ranked):
            means, _ = exact.score_samples(selected, verified_nodes, step, bootstrap_n)
            winner = max(verified_nodes, key=lambda v: (means[v], -v))
            previous_target_winner = winner
            rounds.append({
                "verified": int(target),
                "mc_budget": int(bootstrap_n),
                "winner": int(winner),
                "stage": "bootstrap",
                "certified": False,
            })
            target = min(len(ranked), target + int(batch_m))
            continue

        winner_at_max = None
        means_at_max = None
        for n in budgets:
            means, samples = exact.score_samples(selected, verified_nodes, step, n)
            last_means = means
            ordered = sorted(verified_nodes, key=lambda v: (means[v], -v), reverse=True)
            winner = ordered[0]
            runner = ordered[1] if len(ordered) > 1 else None
            internal_ok, pair_mean, pair_se = paired_confidence(winner, runner, samples, confidence_z)

            residuals = [means[v] - learned[v] for v in verified_nodes]
            residual_max = max(residuals) if residuals else 0.0
            residual_mean = sum(residuals) / len(residuals) if residuals else 0.0
            residual_var = sum((x - residual_mean) ** 2 for x in residuals) / len(residuals) if residuals else 0.0
            residual_std = math.sqrt(max(0.0, residual_var))
            outsider = ranked[target] if target < len(ranked) else None
            outsider_upper = None if outsider is None else float(
                learned[outsider] + residual_max + float(residual_beta) * residual_std
            )
            stable = previous_target_winner is None or winner == previous_target_winner
            outsider_ok = outsider is None or float(means[winner]) >= float(outsider_upper)
            early_ok = bool(stable and outsider_ok and internal_ok)
            max_mc_ok = bool(stable and outsider_ok and n == max_mc)
            certified = bool(early_ok or max_mc_ok)

            rounds.append({
                "verified": int(target),
                "mc_budget": int(n),
                "winner": int(winner),
                "runner_up": None if runner is None else int(runner),
                "winner_mean": float(means[winner]),
                "runner_mean": None if runner is None else float(means[runner]),
                "paired_diff_mean": float(pair_mean),
                "paired_diff_se": float(pair_se),
                "internal_confident": bool(internal_ok),
                "stable_vs_previous_shortlist": bool(stable),
                "best_unverified": None if outsider is None else int(outsider),
                "best_unverified_upper": outsider_upper,
                "outsider_certified": bool(outsider_ok),
                "residual_std": float(residual_std),
                "certified": bool(certified),
            })

            if certified:
                chosen = winner
                final_n = int(n)
                stop_reason = "progressive_early" if n < max_mc else "progressive_mc40_certified"
                break
            if n == max_mc:
                winner_at_max = winner
                means_at_max = means

        if chosen is not None:
            break

        if target >= len(ranked):
            chosen = winner_at_max if winner_at_max is not None else ordered[0]
            last_means = means_at_max if means_at_max is not None else means
            final_n = max_mc
            stop_reason = "progressive_all_candidates_mc40"
            break

        previous_target_winner = winner_at_max
        target = min(len(ranked), target + int(batch_m))

    return int(chosen), float(last_means[chosen]), int(target), int(final_n), rounds, stop_reason


def trust_progressive_greedy(
    candidate_pool: list[int],
    budget: int,
    learned_oracle,
    exact_oracle: ProgressiveMonteCarloOracle,
    trust_tau: float = 0.3,
    audit_top_k: int = 16,
    audit_sentinels: int = 8,
    audit_mc: int = 10,
    sentinel_margin: float = 0.0,
):
    selected: list[int] = []
    steps = []

    for step in range(int(budget)):
        selected_set = set(selected)
        available = [v for v in candidate_pool if v not in selected_set]
        if not available:
            break

        learned = learned_oracle.score(selected, available, step=step)
        ranked = sorted(available, key=lambda v: (learned[v], -v), reverse=True)

        head = ranked[: min(int(audit_top_k), len(ranked))]
        sentinels = sentinel_nodes(ranked, len(head), int(audit_sentinels))
        audit = list(dict.fromkeys(head + sentinels))
        audit_means, _ = exact_oracle.score_samples(selected, audit, step, int(audit_mc))
        rho = _rank_spearman({v: learned[v] for v in audit}, audit_means)
        best_head = max(head, key=lambda v: (audit_means[v], -v))
        best_sentinel = None
        sentinel_surprise = False
        if sentinels:
            best_sentinel = max(sentinels, key=lambda v: (audit_means[v], -v))
            sentinel_surprise = audit_means[best_sentinel] > audit_means[best_head] + float(sentinel_margin)
        trusted = bool(rho >= float(trust_tau) and not sentinel_surprise)

        before_candidates = int(exact_oracle.stats.candidate_evaluations)
        before_samples = int(exact_oracle.stats.mc_candidate_samples)

        if not trusted:
            full_means, _ = exact_oracle.score_samples(selected, available, step, 40)
            chosen = max(available, key=lambda v: (full_means[v], -v))
            oracle_score = float(full_means[chosen])
            verified = len(available)
            final_mc = 40
            mode = "trust_fallback_full_mc40"
            progressive_rounds = []
            stop_reason = mode
        else:
            chosen, oracle_score, verified, final_mc, progressive_rounds, stop_reason = progressive_select_step(
                selected, ranked, learned, exact_oracle, step,
                sample_budgets=(5, 10, 20, 40),
                initial_m=8, batch_m=8,
                residual_beta=0.5, confidence_z=0.5, bootstrap_mc=10,
            )
            mode = "trusted_progressive"

        after_candidates = int(exact_oracle.stats.candidate_evaluations)
        after_samples = int(exact_oracle.stats.mc_candidate_samples)
        steps.append({
            "step": step + 1,
            "chosen": int(chosen),
            "mode": mode,
            "stop_reason": stop_reason,
            "trust_rho": float(rho),
            "trusted": bool(trusted),
            "sentinel_surprise": bool(sentinel_surprise),
            "audit_size": len(audit),
            "audit_mc": int(audit_mc),
            "best_head": int(best_head),
            "best_sentinel": None if best_sentinel is None else int(best_sentinel),
            "verified": int(verified),
            "final_mc": int(final_mc),
            "oracle_score": float(oracle_score),
            "incremental_candidates_after_audit": after_candidates - before_candidates,
            "incremental_samples_after_audit": after_samples - before_samples,
            "progressive_rounds": progressive_rounds,
        })
        selected.append(int(chosen))

    return selected, steps


def run_level(
    graph, candidate_pool, budget, eval_mc, model, embeddings, norm_degrees, device,
    alpha: float, tau: float, audit_mc: int, sentinels: int,
):
    base = LearnedMarginalOracle(model, embeddings, norm_degrees, device)
    learned = CorruptedLearnedOracle(base, alpha=alpha)
    exact = ProgressiveMonteCarloOracle(graph, max_mc=40, random_seed=260903)
    start = time.perf_counter()
    seeds, steps = trust_progressive_greedy(
        candidate_pool, budget, learned, exact,
        trust_tau=tau, audit_top_k=16,
        audit_sentinels=sentinels, audit_mc=audit_mc,
    )
    elapsed = time.perf_counter() - start
    spread = estimate_spread(graph, seeds, eval_mc, 960903)
    modes = Counter(x["mode"] for x in steps)
    return {
        "alpha": float(alpha),
        "trust_tau": float(tau),
        "audit_mc": int(audit_mc),
        "audit_sentinels": int(sentinels),
        "mean_clean_corrupt_spearman": float(sum(learned.correlations) / len(learned.correlations)),
        "selected_seeds": seeds,
        "final_spread_mean": float(spread["mean"]),
        "final_spread_std": float(spread["std"]),
        "selection_seconds": float(elapsed),
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
        "steps": steps,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool-size", type=int, default=128)
    p.add_argument("--budget", type=int, default=10)
    p.add_argument("--eval-mc", type=int, default=1000)
    p.add_argument("--alphas", nargs="+", type=float, default=[0, 0.75, 1.0])
    p.add_argument("--tau", type=float, default=0.3)
    p.add_argument("--audit-mc-values", nargs="+", type=int, default=[10, 20])
    p.add_argument("--sentinel-values", nargs="+", type=int, default=[4, 8])
    args = p.parse_args()

    graph_data, graph, device, embeddings, norm_degrees, model, candidate_pool = build_context(args.pool_size)
    print(f"device={device} nodes={graph_data.num_nodes} edges={graph_data.num_edges}", flush=True)
    print(f"candidate_pool={len(candidate_pool)} budget={args.budget}", flush=True)

    full_spread = 444.911
    full_candidates = 1235
    full_samples = 1235 * 40
    progressive_clean_samples = 18280
    methods = {}

    for audit_mc in args.audit_mc_values:
        for sentinels in args.sentinel_values:
            for alpha in args.alphas:
                key = f"audit{audit_mc}_sent{sentinels}_alpha{alpha:g}"
                item = run_level(
                    graph, candidate_pool, args.budget, args.eval_mc,
                    model, embeddings, norm_degrees, device,
                    alpha, args.tau, audit_mc, sentinels,
                )
                s = item["oracle_stats"]
                item["quality_ratio_vs_full_mc"] = float(item["final_spread_mean"] / full_spread)
                item["candidate_fraction_vs_full_mc"] = float(s["candidate_evaluations"] / full_candidates)
                item["sample_fraction_vs_full_mc"] = float(s["mc_candidate_samples"] / full_samples)
                item["sample_multiplier_vs_progressive_clean"] = float(s["mc_candidate_samples"] / progressive_clean_samples)
                methods[key] = item
                print(
                    key,
                    f"rho={item['mean_clean_corrupt_spearman']:.3f}",
                    f"spread={item['final_spread_mean']:.3f}",
                    f"ratio={item['quality_ratio_vs_full_mc']:.4f}",
                    f"exact={s['candidate_evaluations']}",
                    f"samples={s['mc_candidate_samples']}",
                    f"sample_full={item['sample_fraction_vs_full_mc']:.3f}",
                    f"sample_x_prog={item['sample_multiplier_vs_progressive_clean']:.3f}",
                    f"modes={item['mode_counts']}",
                    f"trust={[round(x,2) for x in item['trust_rho_per_step']]}",
                    flush=True,
                )

    out = ROOT / "outputs" / "end_to_end" / "trust_progressive" / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "dataset": "NetHEPT",
        "config": vars(args),
        "reference": {
            "full_mc_spread": full_spread,
            "full_mc_candidate_evaluations": full_candidates,
            "full_mc_candidate_samples": full_samples,
            "progressive_v3_clean_samples": progressive_clean_samples,
            "progressive_v3_clean_spread": 443.626,
        },
        "methods": methods,
    }, indent=2))
    print(f"saved={out}", flush=True)


if __name__ == "__main__":
    main()
