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
from evaluate_trust_gate_stress import sentinel_nodes
from evaluate_trust_progressive import progressive_select_step
from evaluate_trust_calibration_multiseed import run_full_reference
from grl.diffusion import estimate_spread
from grl.oracle import LearnedMarginalOracle


def mean(xs):
    return float(sum(xs) / len(xs)) if xs else 0.0


def quantile(xs, q: float):
    ys = sorted(float(x) for x in xs)
    if not ys:
        return 0.0
    if len(ys) == 1:
        return ys[0]
    q = min(1.0, max(0.0, float(q)))
    pos = (len(ys) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ys[lo]
    t = pos - lo
    return ys[lo] * (1.0 - t) + ys[hi] * t


def audited_residual_greedy(
    candidate_pool,
    budget,
    learned_oracle,
    exact_oracle: ProgressiveMonteCarloOracle,
    audit_top_k=16,
    audit_sentinels=8,
    audit_mc=20,
    residual_q=0.9,
    residual_beta=0.0,
):
    selected = []
    steps = []

    for step in range(int(budget)):
        available = [v for v in candidate_pool if v not in set(selected)]
        if not available:
            break

        learned = learned_oracle.score(selected, available, step=step)
        ranked = sorted(available, key=lambda v: (learned[v], -v), reverse=True)
        head = ranked[: min(int(audit_top_k), len(ranked))]
        sentinels = sentinel_nodes(ranked, len(head), int(audit_sentinels))
        audit = list(dict.fromkeys(head + sentinels))
        audit_means, _ = exact_oracle.score_samples(selected, audit, step, int(audit_mc))

        best_head = max(head, key=lambda v: (audit_means[v], -v))
        best_sentinel = None
        sentinel_surprise = False
        if sentinels:
            best_sentinel = max(sentinels, key=lambda v: (audit_means[v], -v))
            sentinel_surprise = bool(audit_means[best_sentinel] > audit_means[best_head])

        residuals = [float(audit_means[v] - learned[v]) for v in audit]
        r_mean = mean(residuals)
        r_var = mean([(x - r_mean) ** 2 for x in residuals]) if residuals else 0.0
        r_std = math.sqrt(max(0.0, r_var))
        r_upper = float(quantile(residuals, residual_q) + float(residual_beta) * r_std)

        outsider = ranked[len(head)] if len(ranked) > len(head) else None
        outsider_upper = None if outsider is None else float(learned[outsider] + r_upper)
        residual_certified = bool(
            outsider is None or float(audit_means[best_head]) >= float(outsider_upper)
        )
        trusted = bool(residual_certified and not sentinel_surprise)

        before_cand = int(exact_oracle.stats.candidate_evaluations)
        before_samples = int(exact_oracle.stats.mc_candidate_samples)

        if trusted:
            chosen, oracle_score, verified, final_mc, rounds, stop_reason = progressive_select_step(
                selected,
                ranked,
                learned,
                exact_oracle,
                step,
                sample_budgets=(5, 10, 20, 40),
                initial_m=8,
                batch_m=8,
                residual_beta=0.5,
                confidence_z=0.5,
                bootstrap_mc=10,
            )
            mode = "trusted_progressive"
        else:
            full_means, _ = exact_oracle.score_samples(selected, available, step, 40)
            chosen = max(available, key=lambda v: (full_means[v], -v))
            oracle_score = float(full_means[chosen])
            verified = len(available)
            final_mc = 40
            rounds = []
            stop_reason = "audited_residual_fallback_full_mc40"
            mode = stop_reason

        after_cand = int(exact_oracle.stats.candidate_evaluations)
        after_samples = int(exact_oracle.stats.mc_candidate_samples)
        steps.append({
            "step": int(step + 1),
            "chosen": int(chosen),
            "mode": mode,
            "trusted": bool(trusted),
            "sentinel_surprise": bool(sentinel_surprise),
            "best_head": int(best_head),
            "best_sentinel": None if best_sentinel is None else int(best_sentinel),
            "residual_q": float(residual_q),
            "residual_quantile": float(quantile(residuals, residual_q)),
            "residual_std": float(r_std),
            "residual_upper": float(r_upper),
            "best_outsider": None if outsider is None else int(outsider),
            "best_outsider_upper": outsider_upper,
            "head_exact_mean": float(audit_means[best_head]),
            "residual_certified": bool(residual_certified),
            "verified": int(verified),
            "final_mc": int(final_mc),
            "oracle_score": float(oracle_score),
            "incremental_candidates_after_audit": after_cand - before_cand,
            "incremental_samples_after_audit": after_samples - before_samples,
            "progressive_rounds": rounds,
            "stop_reason": stop_reason,
        })
        selected.append(int(chosen))

    return selected, steps


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
    residual_q,
    residual_beta,
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
    seeds, steps = audited_residual_greedy(
        candidate_pool,
        budget,
        learned,
        exact,
        audit_top_k=16,
        audit_sentinels=int(sentinels),
        audit_mc=int(audit_mc),
        residual_q=float(residual_q),
        residual_beta=float(residual_beta),
    )
    elapsed = time.perf_counter() - t0
    spread = estimate_spread(graph, seeds, eval_mc, int(eval_seed))
    modes = Counter(x["mode"] for x in steps)
    fallback_steps = int(modes.get("audited_residual_fallback_full_mc40", 0))
    return {
        "alpha": float(alpha),
        "residual_q": float(residual_q),
        "residual_beta": float(residual_beta),
        "corruption_seed": int(corruption_seed),
        "exact_seed": int(exact_seed),
        "eval_seed": int(eval_seed),
        "mean_clean_corrupt_spearman": mean(learned.correlations),
        "selected_seeds": [int(v) for v in seeds],
        "final_spread_mean": float(spread["mean"]),
        "final_spread_std": float(spread["std"]),
        "selection_seconds": float(elapsed),
        "fallback_steps": int(fallback_steps),
        "mode_counts": dict(modes),
        "oracle_stats": {
            "candidate_evaluations": int(exact.stats.candidate_evaluations),
            "mc_candidate_samples": int(exact.stats.mc_candidate_samples),
            "live_edge_samples": int(exact.stats.live_edge_samples),
            "learned_evaluations": int(base.stats.learned_evaluations),
        },
        "trusted_per_step": [bool(x["trusted"]) for x in steps],
        "sentinel_surprise_per_step": [bool(x["sentinel_surprise"]) for x in steps],
        "residual_upper_per_step": [float(x["residual_upper"]) for x in steps],
        "verified_per_step": [int(x["verified"]) for x in steps],
        "steps": steps,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool-size", type=int, default=128)
    p.add_argument("--budget", type=int, default=10)
    p.add_argument("--eval-mc", type=int, default=1000)
    p.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.75, 1.0])
    p.add_argument("--residual-qs", nargs="+", type=float, default=[0.8, 0.9, 1.0])
    p.add_argument("--residual-beta", type=float, default=0.0)
    p.add_argument("--audit-mc", type=int, default=20)
    p.add_argument("--sentinels", type=int, default=8)
    p.add_argument("--corruption-seed", type=int, default=910401)
    p.add_argument("--exact-seed", type=int, default=920401)
    p.add_argument("--eval-seed", type=int, default=930401)
    args = p.parse_args()

    graph_data, graph, device, embeddings, norm_degrees, model, candidate_pool = build_context(args.pool_size)
    print(
        f"device={device} nodes={graph_data.num_nodes} edges={graph_data.num_edges} "
        f"pool={len(candidate_pool)} budget={args.budget}", flush=True
    )

    ref = run_full_reference(
        graph, candidate_pool, args.budget, args.eval_mc, args.exact_seed, args.eval_seed
    )
    full_samples = int(ref["oracle_stats"]["mc_candidate_samples"])
    print(
        f"REF spread={ref['final_spread_mean']:.3f} samples={full_samples} "
        f"time={ref['selection_seconds']:.2f}", flush=True
    )

    methods = {}
    for q in args.residual_qs:
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
                q,
                args.residual_beta,
                args.audit_mc,
                args.sentinels,
                args.corruption_seed,
                args.exact_seed,
                args.eval_seed,
            )
            item["quality_ratio_vs_full_mc"] = float(item["final_spread_mean"] / ref["final_spread_mean"])
            item["sample_fraction_vs_full_mc"] = float(item["oracle_stats"]["mc_candidate_samples"] / full_samples)
            key = f"q{q:g}_alpha{alpha:g}"
            methods[key] = item
            print(
                key,
                f"pred_rho={item['mean_clean_corrupt_spearman']:.3f}",
                f"fallback={item['fallback_steps']}/{args.budget}",
                f"spread={item['final_spread_mean']:.3f}",
                f"ratio={item['quality_ratio_vs_full_mc']:.4f}",
                f"samples={item['oracle_stats']['mc_candidate_samples']}",
                f"frac={item['sample_fraction_vs_full_mc']:.3f}",
                flush=True,
            )

    out = ROOT / "outputs" / "end_to_end" / "audited_residual_gate" / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "dataset": "NetHEPT",
        "scope": "single-seed audited-residual gate pilot",
        "config": vars(args),
        "full_mc_reference": ref,
        "methods": methods,
    }, indent=2), encoding="utf-8")
    print(f"saved={out}", flush=True)


if __name__ == "__main__":
    main()
