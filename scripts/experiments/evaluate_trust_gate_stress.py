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
from evaluate_robustness_stress import CorruptedLearnedOracle, _rank_spearman
from grl.diffusion import estimate_spread
from grl.oracle import LearnedMarginalOracle


def sentinel_nodes(ranked: list[int], top_k: int, count: int) -> list[int]:
    """Deterministic rank-spaced audit nodes outside the predicted head."""
    if count <= 0 or len(ranked) <= top_k:
        return []
    lo = top_k
    hi = len(ranked) - 1
    if count == 1:
        idxs = [hi]
    else:
        idxs = [round(lo + i * (hi - lo) / (count - 1)) for i in range(count)]
    out = []
    seen = set()
    for idx in idxs:
        v = ranked[int(idx)]
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def residual_fast_select(
    selected: list[int],
    ranked: list[int],
    learned: dict[int, float],
    exact: ProgressiveMonteCarloOracle,
    step: int,
    initial_m: int = 16,
    batch_m: int = 8,
    residual_beta: float = 0.5,
):
    """Validated residual-envelope path using MC40, reusing audit samples."""
    target = min(int(initial_m), len(ranked))
    prev_winner = None
    rounds = []
    while True:
        nodes = ranked[:target]
        means, _ = exact.score_samples(selected, nodes, step, 40)
        winner = max(nodes, key=lambda v: (means[v], -v))
        residuals = [means[v] - learned[v] for v in nodes]
        rmax = max(residuals) if residuals else 0.0
        rmean = sum(residuals) / len(residuals) if residuals else 0.0
        rvar = sum((x - rmean) ** 2 for x in residuals) / len(residuals) if residuals else 0.0
        rstd = math.sqrt(max(0.0, rvar))
        outsider = ranked[target] if target < len(ranked) else None
        upper = None if outsider is None else float(learned[outsider] + rmax + residual_beta * rstd)
        stable = prev_winner is not None and winner == prev_winner
        certified = outsider is None or (stable and float(means[winner]) >= float(upper))
        rounds.append({
            "verified": target,
            "winner": int(winner),
            "winner_exact": float(means[winner]),
            "best_unverified": None if outsider is None else int(outsider),
            "best_unverified_upper": upper,
            "stable": bool(stable),
            "residual_std": float(rstd),
            "certified": bool(certified),
        })
        if certified:
            return int(winner), float(means[winner]), target, rounds
        if target >= len(ranked):
            return int(winner), float(means[winner]), target, rounds
        prev_winner = winner
        target = min(len(ranked), target + int(batch_m))


def trust_gated_greedy(
    candidate_pool: list[int],
    budget: int,
    learned_oracle,
    exact_oracle: ProgressiveMonteCarloOracle,
    trust_tau: float = 0.4,
    audit_top_k: int = 16,
    audit_sentinels: int = 8,
    audit_mc: int = 20,
    sentinel_margin: float = 0.0,
):
    selected = []
    steps = []
    for step in range(int(budget)):
        available = [v for v in candidate_pool if v not in set(selected)]
        if not available:
            break
        learned = learned_oracle.score(selected, available, step=step)
        ranked = sorted(available, key=lambda v: (learned[v], -v), reverse=True)

        head = ranked[: min(audit_top_k, len(ranked))]
        sentinels = sentinel_nodes(ranked, len(head), audit_sentinels)
        audit = list(dict.fromkeys(head + sentinels))
        audit_means, _ = exact_oracle.score_samples(selected, audit, step, audit_mc)
        rho = _rank_spearman({v: learned[v] for v in audit}, audit_means)
        best_head = max(head, key=lambda v: (audit_means[v], -v))
        best_sentinel = None
        sentinel_surprise = False
        if sentinels:
            best_sentinel = max(sentinels, key=lambda v: (audit_means[v], -v))
            sentinel_surprise = audit_means[best_sentinel] > audit_means[best_head] + float(sentinel_margin)
        trusted = bool(rho >= float(trust_tau) and not sentinel_surprise)

        if not trusted:
            # Robust fallback: the predictor is ignored for this step.
            full_means, _ = exact_oracle.score_samples(selected, available, step, 40)
            chosen = max(available, key=lambda v: (full_means[v], -v))
            oracle_score = float(full_means[chosen])
            verified = len(available)
            mode = "trust_fallback_full_mc40"
            fast_rounds = []
        else:
            chosen, oracle_score, verified, fast_rounds = residual_fast_select(
                selected, ranked, learned, exact_oracle, step,
                initial_m=audit_top_k, batch_m=8, residual_beta=0.5,
            )
            mode = "trusted_fast_path"

        steps.append({
            "step": step + 1,
            "chosen": int(chosen),
            "mode": mode,
            "trust_rho": float(rho),
            "trusted": bool(trusted),
            "sentinel_surprise": bool(sentinel_surprise),
            "best_head": int(best_head),
            "best_sentinel": None if best_sentinel is None else int(best_sentinel),
            "best_head_audit": float(audit_means[best_head]),
            "best_sentinel_audit": None if best_sentinel is None else float(audit_means[best_sentinel]),
            "verified": int(verified),
            "oracle_score": oracle_score,
            "fast_rounds": fast_rounds,
        })
        selected.append(int(chosen))
    return selected, steps


def run_level(graph, candidate_pool, budget, eval_mc, model, embeddings, norm_degrees, device, alpha, tau):
    base = LearnedMarginalOracle(model, embeddings, norm_degrees, device)
    learned = CorruptedLearnedOracle(base, alpha=alpha)
    exact = ProgressiveMonteCarloOracle(graph, max_mc=40, random_seed=260903)
    start = time.perf_counter()
    seeds, steps = trust_gated_greedy(
        candidate_pool, budget, learned, exact,
        trust_tau=tau, audit_top_k=16, audit_sentinels=8, audit_mc=20,
    )
    elapsed = time.perf_counter() - start
    spread = estimate_spread(graph, seeds, eval_mc, 960903)
    modes = Counter(x["mode"] for x in steps)
    return {
        "alpha": float(alpha),
        "trust_tau": float(tau),
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
        "steps": steps,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool-size", type=int, default=128)
    p.add_argument("--budget", type=int, default=10)
    p.add_argument("--eval-mc", type=int, default=1000)
    p.add_argument("--alphas", nargs="+", type=float, default=[0, 0.5, 0.75, 1.0])
    p.add_argument("--taus", nargs="+", type=float, default=[0.4])
    args = p.parse_args()

    graph_data, graph, device, embeddings, norm_degrees, model, candidate_pool = build_context(args.pool_size)
    print(f"device={device} nodes={graph_data.num_nodes} edges={graph_data.num_edges}", flush=True)
    print(f"candidate_pool={len(candidate_pool)} budget={args.budget}", flush=True)

    full_spread = 444.911
    full_candidates = 1235
    full_samples = 1235 * 40
    methods = {}
    for tau in args.taus:
        for alpha in args.alphas:
            key = f"tau_{tau:g}_alpha_{alpha:g}"
            item = run_level(
                graph, candidate_pool, args.budget, args.eval_mc,
                model, embeddings, norm_degrees, device, alpha, tau,
            )
            s = item["oracle_stats"]
            item["quality_ratio_vs_full_mc"] = float(item["final_spread_mean"] / full_spread)
            item["candidate_fraction_vs_full_mc"] = float(s["candidate_evaluations"] / full_candidates)
            item["sample_fraction_vs_full_mc"] = float(s["mc_candidate_samples"] / full_samples)
            methods[key] = item
            print(
                key,
                f"global_rho={item['mean_clean_corrupt_spearman']:.3f}",
                f"spread={item['final_spread_mean']:.3f}",
                f"ratio={item['quality_ratio_vs_full_mc']:.4f}",
                f"exact={s['candidate_evaluations']}",
                f"samples={s['mc_candidate_samples']}",
                f"cand_frac={item['candidate_fraction_vs_full_mc']:.3f}",
                f"sample_frac={item['sample_fraction_vs_full_mc']:.3f}",
                f"modes={item['mode_counts']}",
                f"trust={[round(x,2) for x in item['trust_rho_per_step']]}",
                f"trusted={item['trusted_per_step']}",
                flush=True,
            )

    out = ROOT / "outputs" / "end_to_end" / "trust_gate_stress" / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "dataset": "NetHEPT",
        "config": vars(args),
        "reference": {
            "full_mc_spread": full_spread,
            "full_mc_candidate_evaluations": full_candidates,
            "full_mc_candidate_samples": full_samples,
        },
        "methods": methods,
    }, indent=2))
    print(f"saved={out}", flush=True)


if __name__ == "__main__":
    main()
