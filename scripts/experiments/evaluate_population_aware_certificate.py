from __future__ import annotations

"""Population-aware audited certificate for large candidate pools.

The previous audited-residual gate used the maximum audited residual as if it
were representative of an arbitrarily large unseen tail.  This script adds an
extreme-value-style correction that grows with the number of unaudited
candidates.  The goal is not to claim a formal probability bound yet; it is a
focused empirical test of the correct scaling behavior:

    more unseen candidates -> larger outsider uncertainty -> more verification.

We compare the original state-aware proposal and the best large-candidate
hard-negative tuned proposal on the same 512-candidate pool.  A small fixed
sweep over the population correction coefficient is used only to identify a
viable quality/cost region before multi-seed formalization.
"""

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

from evaluate_adaptive_certification import build_context, load_model
from evaluate_progressive_mc import ProgressiveMonteCarloOracle
from evaluate_robustness_stress import CorruptedLearnedOracle
from evaluate_trust_gate_stress import sentinel_nodes
from evaluate_trust_progressive import progressive_select_step
from evaluate_trust_calibration_multiseed import run_full_reference
from grl.diffusion import estimate_spread
from grl.models import MarginalGainPredictor
from grl.oracle import LearnedMarginalOracle

OUT = ROOT / "outputs" / "end_to_end" / "population_aware_certificate"
OUT.mkdir(parents=True, exist_ok=True)


def mean(xs):
    return float(sum(xs) / len(xs)) if xs else 0.0


def load_hard_negative_model(embedding_dim, device):
    path = ROOT / "outputs" / "marginal_predictability" / "large_candidate_hard_negative" / "model_hard_negative.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MarginalGainPredictor(embedding_dim, hidden_dim=96).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def population_audited_greedy(
    candidate_pool,
    budget,
    learned_oracle,
    exact_oracle: ProgressiveMonteCarloOracle,
    audit_top_k=16,
    audit_sentinels=8,
    audit_mc=20,
    population_gamma=0.5,
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
        r_std = math.sqrt(max(0.0, mean([(x - r_mean) ** 2 for x in residuals]))) if residuals else 0.0
        r_max = max(residuals) if residuals else 0.0

        n_unseen = max(0, len(ranked) - len(audit))
        # Extreme-value / multiple-comparison heuristic.  It is zero when there
        # is no unseen population and increases roughly as sqrt(log n).
        if n_unseen > 0 and audit:
            multiplicity = math.sqrt(2.0 * math.log1p(float(n_unseen) / max(1.0, float(len(audit)))))
        else:
            multiplicity = 0.0
        population_margin = float(population_gamma) * float(r_std) * float(multiplicity)
        r_upper = float(r_max + population_margin)

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
            stop_reason = "population_fallback_full_mc40"
            mode = stop_reason

        after_cand = int(exact_oracle.stats.candidate_evaluations)
        after_samples = int(exact_oracle.stats.mc_candidate_samples)
        steps.append({
            "step": step + 1,
            "chosen": int(chosen),
            "trusted": bool(trusted),
            "mode": mode,
            "sentinel_surprise": bool(sentinel_surprise),
            "residual_certified": bool(residual_certified),
            "residual_max": float(r_max),
            "residual_std": float(r_std),
            "n_unseen": int(n_unseen),
            "multiplicity": float(multiplicity),
            "population_margin": float(population_margin),
            "residual_upper": float(r_upper),
            "best_outsider": None if outsider is None else int(outsider),
            "best_outsider_upper": outsider_upper,
            "head_exact_mean": float(audit_means[best_head]),
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


def run_method(
    graph,
    candidate_pool,
    budget,
    eval_mc,
    model,
    embeddings,
    norm_degrees,
    device,
    alpha,
    gamma,
    corruption_seed,
    exact_seed,
    eval_seed,
):
    base = LearnedMarginalOracle(model, embeddings, norm_degrees, device)
    learned = CorruptedLearnedOracle(base, alpha=float(alpha), random_seed=int(corruption_seed))
    exact = ProgressiveMonteCarloOracle(graph, max_mc=40, random_seed=int(exact_seed))
    t0 = time.perf_counter()
    seeds, steps = population_audited_greedy(
        candidate_pool,
        budget,
        learned,
        exact,
        audit_top_k=16,
        audit_sentinels=8,
        audit_mc=20,
        population_gamma=float(gamma),
    )
    elapsed = time.perf_counter() - t0
    spread = estimate_spread(graph, seeds, eval_mc, int(eval_seed))
    modes = Counter(x["mode"] for x in steps)
    return {
        "alpha": float(alpha),
        "gamma": float(gamma),
        "selected_seeds": [int(v) for v in seeds],
        "final_spread_mean": float(spread["mean"]),
        "final_spread_std": float(spread["std"]),
        "selection_seconds": float(elapsed),
        "fallback_steps": int(modes.get("population_fallback_full_mc40", 0)),
        "trusted_steps": int(modes.get("trusted_progressive", 0)),
        "oracle_stats": {
            "candidate_evaluations": int(exact.stats.candidate_evaluations),
            "mc_candidate_samples": int(exact.stats.mc_candidate_samples),
            "live_edge_samples": int(exact.stats.live_edge_samples),
        },
        "steps": steps,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool-size", type=int, default=512)
    p.add_argument("--budget", type=int, default=10)
    p.add_argument("--eval-mc", type=int, default=1000)
    p.add_argument("--gammas", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    p.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.75, 1.0])
    p.add_argument("--corruption-seed", type=int, default=1370401)
    p.add_argument("--exact-seed", type=int, default=1380401)
    p.add_argument("--eval-seed", type=int, default=1390401)
    args = p.parse_args()

    gd, graph, device, embeddings, norm_degrees, base_model, candidate_pool = build_context(args.pool_size)
    hard_model = load_hard_negative_model(embeddings.shape[1], device)
    print(f"device={device} nodes={gd.num_nodes} edges={gd.num_edges} pool={len(candidate_pool)}", flush=True)

    ref = run_full_reference(
        graph, candidate_pool, args.budget, args.eval_mc, args.exact_seed, args.eval_seed
    )
    full_samples = int(ref["oracle_stats"]["mc_candidate_samples"])
    print(f"REF spread={ref['final_spread_mean']:.3f} samples={full_samples}", flush=True)

    models = {"state_aware": base_model, "hard_negative": hard_model}
    records = {}
    for model_name, model in models.items():
        for gamma in args.gammas:
            for alpha in args.alphas:
                item = run_method(
                    graph, candidate_pool, args.budget, args.eval_mc,
                    model, embeddings, norm_degrees, device,
                    alpha, gamma,
                    args.corruption_seed, args.exact_seed, args.eval_seed,
                )
                item["quality_ratio_vs_full_mc"] = float(item["final_spread_mean"] / ref["final_spread_mean"])
                item["sample_fraction_vs_full_mc"] = float(item["oracle_stats"]["mc_candidate_samples"] / full_samples)
                key = f"{model_name}_g{gamma:g}_a{alpha:g}"
                records[key] = item
                print(
                    f"RUN {key} ratio={item['quality_ratio_vs_full_mc']:.4f} "
                    f"frac={item['sample_fraction_vs_full_mc']:.3f} "
                    f"fallback={item['fallback_steps']}/{args.budget} spread={item['final_spread_mean']:.3f}",
                    flush=True,
                )

    # Pick the cheapest clean configuration satisfying >=99% quality and whose
    # random endpoint is >=99% quality for the same model/gamma.  This is only
    # a pilot selection rule; multi-seed evaluation must follow.
    viable = []
    for model_name in models:
        for gamma in args.gammas:
            kc = f"{model_name}_g{gamma:g}_a0"
            kr = f"{model_name}_g{gamma:g}_a1"
            if kc in records and kr in records:
                c, r = records[kc], records[kr]
                if c["quality_ratio_vs_full_mc"] >= 0.99 and r["quality_ratio_vs_full_mc"] >= 0.99:
                    viable.append((c["sample_fraction_vs_full_mc"], model_name, gamma, c, r))
    viable.sort(key=lambda x: x[0])
    best = None
    if viable:
        frac, model_name, gamma, c, r = viable[0]
        best = {
            "model": model_name,
            "gamma": float(gamma),
            "clean_quality": float(c["quality_ratio_vs_full_mc"]),
            "clean_sample_fraction": float(c["sample_fraction_vs_full_mc"]),
            "random_quality": float(r["quality_ratio_vs_full_mc"]),
            "random_sample_fraction": float(r["sample_fraction_vs_full_mc"]),
        }
        print("PILOT_VIABLE", json.dumps(best, sort_keys=True), flush=True)
    else:
        print("PILOT_VIABLE none", flush=True)

    out = OUT / "report.json"
    out.write_text(json.dumps({
        "dataset": "NetHEPT",
        "scope": "single-seed 512-candidate population-aware certificate pilot",
        "config": vars(args),
        "full_mc_reference": ref,
        "records": records,
        "best_viable": best,
    }, indent=2), encoding="utf-8")
    print(f"saved={out}", flush=True)


if __name__ == "__main__":
    main()
