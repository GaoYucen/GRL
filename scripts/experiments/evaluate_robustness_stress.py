from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

from evaluate_adaptive_certification import build_context
from evaluate_progressive_mc import ProgressiveMonteCarloOracle, progressive_adaptive_greedy
from grl.diffusion import estimate_spread
from grl.oracle import LearnedMarginalOracle


def _mean(xs):
    return float(sum(xs) / len(xs)) if xs else 0.0


def _std(xs):
    if not xs:
        return 0.0
    m = _mean(xs)
    return float(math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)))


def _rank_spearman(a: dict[int, float], b: dict[int, float]) -> float:
    keys = list(a)
    if len(keys) <= 1:
        return 1.0
    order_a = sorted(keys, key=lambda v: (a[v], -v))
    order_b = sorted(keys, key=lambda v: (b[v], -v))
    ra = {v: i for i, v in enumerate(order_a)}
    rb = {v: i for i, v in enumerate(order_b)}
    xa = [float(ra[v]) for v in keys]
    xb = [float(rb[v]) for v in keys]
    ma, mb = _mean(xa), _mean(xb)
    va = sum((x - ma) ** 2 for x in xa)
    vb = sum((x - mb) ** 2 for x in xb)
    if va <= 0 or vb <= 0:
        return 1.0
    cov = sum((x - ma) * (y - mb) for x, y in zip(xa, xb))
    return float(cov / math.sqrt(va * vb))


class CorruptedLearnedOracle:
    """Deterministically mix clean learned scores with random ranking noise.

    ``alpha=0`` is the original predictor. ``alpha=1`` is a random ranking whose
    score mean/std are matched to the clean predictor at that sequential step.
    Intermediate alpha values smoothly degrade ranking quality without changing
    the score scale, so residual-envelope behavior is not confounded by trivial
    rescaling.
    """

    def __init__(self, base, alpha: float, random_seed: int = 260904):
        self.base = base
        self.alpha = float(alpha)
        self.random_seed = int(random_seed)
        self.correlations: list[float] = []
        self.top8_overlaps: list[float] = []

    @property
    def stats(self):
        return self.base.stats

    def score(self, seeds: list[int], candidates: list[int], step: int = 0) -> dict[int, float]:
        clean = self.base.score(seeds, candidates, step=step)
        if not clean or self.alpha <= 0:
            self.correlations.append(1.0)
            self.top8_overlaps.append(1.0)
            return clean

        vals = [clean[v] for v in candidates]
        mu = _mean(vals)
        sigma = _std(vals)
        if sigma <= 1e-12:
            sigma = 1.0

        clean_z = {v: (clean[v] - mu) / sigma for v in candidates}
        noise_raw = {}
        for v in candidates:
            rng = random.Random(self.random_seed + int(step) * 1_000_003 + int(v) * 9176)
            noise_raw[v] = rng.gauss(0.0, 1.0)
        noise_mu = _mean(list(noise_raw.values()))
        noise_sd = _std(list(noise_raw.values())) or 1.0
        noise_z = {v: (noise_raw[v] - noise_mu) / noise_sd for v in candidates}

        denom = math.sqrt((1.0 - self.alpha) ** 2 + self.alpha ** 2)
        if denom <= 1e-12:
            denom = 1.0
        corrupted = {
            v: float(mu + sigma * (((1.0 - self.alpha) * clean_z[v] + self.alpha * noise_z[v]) / denom))
            for v in candidates
        }
        self.correlations.append(_rank_spearman(clean, corrupted))
        k = min(8, len(candidates))
        clean_top = set(sorted(candidates, key=lambda v: (clean[v], -v), reverse=True)[:k])
        corrupt_top = set(sorted(candidates, key=lambda v: (corrupted[v], -v), reverse=True)[:k])
        self.top8_overlaps.append(float(len(clean_top & corrupt_top) / max(1, k)))
        return corrupted


def run_level(graph, candidate_pool, budget, eval_mc, model, embeddings, norm_degrees, device, alpha):
    base = LearnedMarginalOracle(model, embeddings, norm_degrees, device)
    learned = CorruptedLearnedOracle(base, alpha=alpha)
    exact = ProgressiveMonteCarloOracle(graph, max_mc=40, random_seed=260903)

    start = time.perf_counter()
    seeds, steps = progressive_adaptive_greedy(
        candidate_pool,
        budget,
        learned,
        exact,
        sample_budgets=(5, 10, 20, 40),
        initial_m=8,
        batch_m=8,
        residual_beta=0.5,
        confidence_z=0.5,
        bootstrap_mc=10,
    )
    selection_seconds = time.perf_counter() - start
    spread = estimate_spread(graph, seeds, eval_mc, 960903)

    stop_counts = Counter(x["stop_reason"] for x in steps)
    return {
        "alpha": float(alpha),
        "mean_clean_corrupt_spearman": _mean(learned.correlations),
        "min_clean_corrupt_spearman": min(learned.correlations) if learned.correlations else 1.0,
        "mean_clean_top8_overlap": _mean(learned.top8_overlaps),
        "selected_seeds": [int(v) for v in seeds],
        "final_spread_mean": float(spread["mean"]),
        "final_spread_std": float(spread["std"]),
        "selection_seconds": float(selection_seconds),
        "oracle_stats": {
            "candidate_evaluations": int(exact.stats.candidate_evaluations),
            "mc_candidate_samples": int(exact.stats.mc_candidate_samples),
            "live_edge_samples": int(exact.stats.live_edge_samples),
            "learned_evaluations": int(base.stats.learned_evaluations),
        },
        "verified_per_step": [int(x["verified"]) for x in steps],
        "mc_budget_per_step": [int(x["mc_budget"]) for x in steps],
        "stop_reason_counts": dict(stop_counts),
        "steps": steps,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool-size", type=int, default=128)
    p.add_argument("--budget", type=int, default=10)
    p.add_argument("--eval-mc", type=int, default=1000)
    p.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    args = p.parse_args()

    graph_data, graph, device, embeddings, norm_degrees, model, candidate_pool = build_context(args.pool_size)
    print(f"device={device} nodes={graph_data.num_nodes} edges={graph_data.num_edges}", flush=True)
    print(f"candidate_pool={len(candidate_pool)} budget={args.budget}", flush=True)

    full_spread = 444.911
    full_candidates = 1235
    full_samples = 1235 * 40
    clean_progressive_candidates = 504
    clean_progressive_samples = 18280

    methods = {}
    for alpha in args.alphas:
        key = f"alpha_{alpha:g}"
        item = run_level(
            graph, candidate_pool, args.budget, args.eval_mc,
            model, embeddings, norm_degrees, device, alpha,
        )
        s = item["oracle_stats"]
        item["quality_ratio_vs_full_mc"] = float(item["final_spread_mean"] / full_spread)
        item["candidate_fraction_vs_full_mc"] = float(s["candidate_evaluations"] / full_candidates)
        item["sample_fraction_vs_full_mc"] = float(s["mc_candidate_samples"] / full_samples)
        item["candidate_multiplier_vs_clean"] = float(s["candidate_evaluations"] / clean_progressive_candidates)
        item["sample_multiplier_vs_clean"] = float(s["mc_candidate_samples"] / clean_progressive_samples)
        methods[key] = item
        print(
            f"{key} rank_rho={item['mean_clean_corrupt_spearman']:.3f} top8={item['mean_clean_top8_overlap']:.3f} "
            f"spread={item['final_spread_mean']:.3f} ratio={item['quality_ratio_vs_full_mc']:.4f} "
            f"exact={s['candidate_evaluations']} samples={s['mc_candidate_samples']} worlds={s['live_edge_samples']} "
            f"cand_x={item['candidate_multiplier_vs_clean']:.2f} sample_x={item['sample_multiplier_vs_clean']:.2f} "
            f"time={item['selection_seconds']:.2f} verified={item['verified_per_step']} mc={item['mc_budget_per_step']} "
            f"stops={item['stop_reason_counts']}",
            flush=True,
        )

    report = {
        "dataset": "NetHEPT",
        "protocol": {
            "candidate_pool": len(candidate_pool),
            "budget": args.budget,
            "eval_mc": args.eval_mc,
            "progressive_config": {
                "sample_budgets": [5, 10, 20, 40],
                "initial_m": 8,
                "batch_m": 8,
                "residual_beta": 0.5,
                "confidence_z": 0.5,
                "bootstrap_mc": 10,
            },
            "corruption": "score-scale-preserving mixture of clean standardized scores and deterministic random standardized scores",
        },
        "reference": {
            "full_mc_spread": full_spread,
            "full_mc_candidate_evaluations": full_candidates,
            "full_mc_candidate_samples": full_samples,
            "clean_progressive_candidate_evaluations": clean_progressive_candidates,
            "clean_progressive_candidate_samples": clean_progressive_samples,
        },
        "methods": methods,
    }
    out_dir = ROOT / "outputs" / "end_to_end" / "robustness_stress"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved={out}", flush=True)


if __name__ == "__main__":
    main()
