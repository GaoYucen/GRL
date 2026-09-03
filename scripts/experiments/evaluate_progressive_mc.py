from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

from evaluate_adaptive_certification import build_context
from grl.diffusion import estimate_spread
from grl.oracle import LearnedMarginalOracle


@dataclass
class ProgressiveStats:
    candidate_evaluations: int = 0
    mc_candidate_samples: int = 0
    live_edge_samples: int = 0
    learned_evaluations: int = 0


class ProgressiveMonteCarloOracle:
    """Incremental common-random-number MC oracle.

    Worlds are generated lazily per sequential step. Candidate/world gains are
    cached, so increasing 5->10->20->40 samples only computes newly needed
    candidate-world pairs. Expanding the shortlist evaluates only new candidates
    on worlds that already exist.
    """

    def __init__(self, graph: nx.Graph | nx.DiGraph, max_mc: int = 40, random_seed: int = 260903):
        self.graph = graph
        self.max_mc = int(max_mc)
        self.random_seed = int(random_seed)
        self.stats = ProgressiveStats()
        self._key = None
        self._worlds: list[tuple[nx.Graph | nx.DiGraph, set[int]]] = []
        self._samples: dict[int, list[float]] = {}
        self._seen_candidates: set[int] = set()

    def _reached(self, live_graph, seeds):
        reached = set()
        for seed in seeds:
            if live_graph.is_directed():
                reached.update(nx.descendants(live_graph, seed))
            else:
                reached.update(nx.node_connected_component(live_graph, seed))
            reached.add(seed)
        return reached

    def _reset_if_needed(self, seeds, step):
        key = (int(step), tuple(sorted(int(v) for v in seeds)))
        if key != self._key:
            self._key = key
            self._worlds = []
            self._samples = {}
            self._seen_candidates = set()

    def _ensure_worlds(self, seeds, step, n):
        self._reset_if_needed(seeds, step)
        target = min(int(n), self.max_mc)
        while len(self._worlds) < target:
            offset = len(self._worlds)
            import random
            rng = random.Random(self.random_seed + int(step) * 1_000_003 + offset)
            live_graph = nx.DiGraph() if self.graph.is_directed() else nx.Graph()
            live_graph.add_nodes_from(self.graph.nodes())
            for u, v, data in self.graph.edges(data=True):
                if rng.random() < float(data.get("weight", 0.0)):
                    live_graph.add_edge(u, v)
            self._worlds.append((live_graph, self._reached(live_graph, seeds)))
            self.stats.live_edge_samples += 1

    def score_samples(self, seeds, candidates, step: int, n: int):
        if not candidates:
            return {}, {}
        n = min(int(n), self.max_mc)
        self._ensure_worlds(seeds, step, n)
        for candidate in candidates:
            candidate = int(candidate)
            values = self._samples.setdefault(candidate, [])
            if candidate not in self._seen_candidates:
                self._seen_candidates.add(candidate)
                self.stats.candidate_evaluations += 1
            for idx in range(len(values), n):
                live_graph, base = self._worlds[idx]
                if candidate in base:
                    gain = 0.0
                else:
                    if live_graph.is_directed():
                        cand_reached = set(nx.descendants(live_graph, candidate))
                    else:
                        cand_reached = set(nx.node_connected_component(live_graph, candidate))
                    cand_reached.add(candidate)
                    gain = float(len(cand_reached - base))
                values.append(gain)
                self.stats.mc_candidate_samples += 1
        means = {int(v): float(sum(self._samples[int(v)][:n]) / n) for v in candidates}
        samples = {int(v): list(self._samples[int(v)][:n]) for v in candidates}
        return means, samples


def paired_confidence(best: int, runner: int | None, samples: dict[int, list[float]], z: float):
    if runner is None:
        return True, float("inf"), 0.0
    a = samples[best]
    b = samples[runner]
    n = min(len(a), len(b))
    diffs = [a[i] - b[i] for i in range(n)]
    mean = float(sum(diffs) / n)
    if n <= 1:
        return mean > 0.0, mean, float("inf")
    sd = float(statistics.stdev(diffs))
    se = sd / math.sqrt(n)
    lcb = mean - float(z) * se
    return bool(lcb > 0.0), mean, se


def progressive_adaptive_greedy(
    candidate_pool,
    budget,
    learned_oracle,
    exact_oracle,
    sample_budgets=(5, 10, 20, 40),
    initial_m=8,
    batch_m=8,
    residual_beta=0.5,
    confidence_z=1.0,
    bootstrap_mc=40,
):
    """Two-level adaptive certification.

    Candidate expansion follows the same max-MC residual-envelope/stability rule
    as the validated fixed adaptive method. Progressive MC is only an early-stop
    layer *within* a candidate shortlist. If early sampling is inconclusive, the
    method reaches 40 common-random-number worlds and therefore recovers the same
    shortlist decision logic as fixed adaptive certification.
    """
    selected = []
    steps = []
    budgets = sorted({int(x) for x in sample_budgets if int(x) > 0})
    if not budgets:
        raise ValueError("sample_budgets must be non-empty")
    max_mc = budgets[-1]
    bootstrap_mc = max(1, min(int(bootstrap_mc), max_mc))
    bootstrap_n = max([x for x in budgets if x <= bootstrap_mc] or [budgets[0]])

    for step in range(int(budget)):
        available = [v for v in candidate_pool if v not in set(selected)]
        if not available:
            break
        learned = learned_oracle.score(selected, available, step=step)
        ranked = sorted(available, key=lambda v: (learned[v], -v), reverse=True)
        target = min(int(initial_m), len(ranked))
        previous_target_winner = None
        rounds = []
        chosen = None
        stop_reason = None
        last_means = None
        final_n = max_mc

        while True:
            verified_nodes = ranked[:target]

            # Bootstrap the first shortlist at max MC so the target-to-target
            # stability trajectory is identical to the fixed adaptive baseline.
            if previous_target_winner is None and target < len(ranked):
                means, samples = exact_oracle.score_samples(selected, verified_nodes, step, bootstrap_n)
                ordered = sorted(verified_nodes, key=lambda v: (means[v], -v), reverse=True)
                winner = ordered[0]
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

            expanded = False
            winner_at_max = None
            means_at_max = None
            for n in budgets:
                means, samples = exact_oracle.score_samples(selected, verified_nodes, step, n)
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
                    "residual_max": float(residual_max),
                    "residual_std": float(residual_std),
                    "certified": bool(certified),
                })

                if certified:
                    chosen = winner
                    final_n = int(n)
                    stop_reason = "progressive_early" if n < max_mc else "fallback_mc40_certified"
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
                stop_reason = "all_candidates_mc40"
                break

            # At max MC the current shortlist was not certifiable. Expand exactly
            # as in the fixed adaptive method, but let only the newly added nodes
            # start again from 5 MC samples; old candidate/world values are cached.
            previous_target_winner = winner_at_max
            target = min(len(ranked), target + int(batch_m))
            expanded = True
            if not expanded:
                raise RuntimeError("unreachable")

        selected.append(int(chosen))
        steps.append({
            "step": step + 1,
            "chosen": int(chosen),
            "predicted_score": float(learned[chosen]),
            "oracle_score": float(last_means[chosen]),
            "verified": int(target),
            "mc_budget": int(final_n),
            "stop_reason": stop_reason,
            "rounds": rounds,
            "shortlist": [int(v) for v in ranked[:target]],
        })
    return selected, steps

def run_config(graph, candidate_pool, budget, eval_mc, model, embeddings, norm_degrees, device, z, residual_beta, bootstrap_mc=40):
    learned = LearnedMarginalOracle(model, embeddings, norm_degrees, device)
    exact = ProgressiveMonteCarloOracle(graph, max_mc=40, random_seed=260903)
    start = time.perf_counter()
    seeds, steps = progressive_adaptive_greedy(
        candidate_pool, budget, learned, exact,
        sample_budgets=(5, 10, 20, 40), initial_m=8, batch_m=8,
        residual_beta=residual_beta, confidence_z=z, bootstrap_mc=bootstrap_mc,
    )
    selection_seconds = time.perf_counter() - start
    spread = estimate_spread(graph, seeds, eval_mc, 960903)
    stats = asdict(exact.stats)
    stats["learned_evaluations"] = int(learned.stats.learned_evaluations)
    return {
        "confidence_z": float(z),
        "residual_beta": float(residual_beta),
        "bootstrap_mc": int(bootstrap_mc),
        "selected_seeds": seeds,
        "steps": steps,
        "selection_seconds": float(selection_seconds),
        "final_spread_mean": float(spread["mean"]),
        "final_spread_std": float(spread["std"]),
        "oracle_stats": stats,
        "verified_per_step": [int(x["verified"]) for x in steps],
        "mc_budget_per_step": [int(x["mc_budget"]) for x in steps],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool-size", type=int, default=128)
    p.add_argument("--budget", type=int, default=10)
    p.add_argument("--eval-mc", type=int, default=1000)
    p.add_argument("--z-values", nargs="+", type=float, default=[0.5, 1.0, 1.5])
    p.add_argument("--residual-beta", type=float, default=0.5)
    p.add_argument("--bootstrap-values", nargs="+", type=int, default=[40])
    args = p.parse_args()

    graph_data, graph, device, embeddings, norm_degrees, model, candidate_pool = build_context(args.pool_size)
    print(f"device={device} nodes={graph_data.num_nodes} edges={graph_data.num_edges}", flush=True)
    print(f"candidate_pool={len(candidate_pool)} budget={args.budget}", flush=True)

    full_spread = 444.911
    full_candidate_evals = 1235
    full_mc_candidate_samples = full_candidate_evals * 40
    fixed_adaptive_samples = 512 * 40

    methods = {}
    for bootstrap_mc in args.bootstrap_values:
        for z in args.z_values:
            key = f"progressive_boot_{bootstrap_mc}_z_{z:g}"
            item = run_config(
                graph, candidate_pool, args.budget, args.eval_mc,
                model, embeddings, norm_degrees, device, z, args.residual_beta, bootstrap_mc,
            )
            item["quality_ratio_vs_full_mc"] = float(item["final_spread_mean"] / full_spread)
            item["exact_fraction_vs_full_mc"] = float(item["oracle_stats"]["candidate_evaluations"] / full_candidate_evals)
            item["sample_fraction_vs_full_mc"] = float(item["oracle_stats"]["mc_candidate_samples"] / full_mc_candidate_samples)
            item["sample_fraction_vs_fixed_adaptive"] = float(item["oracle_stats"]["mc_candidate_samples"] / fixed_adaptive_samples)
            methods[key] = item
            print(
                f"{key} spread={item['final_spread_mean']:.3f} ratio={item['quality_ratio_vs_full_mc']:.4f} "
                f"exact={item['oracle_stats']['candidate_evaluations']} samples={item['oracle_stats']['mc_candidate_samples']} "
                f"sample_full={item['sample_fraction_vs_full_mc']:.4f} "
                f"sample_fixed={item['sample_fraction_vs_fixed_adaptive']:.4f} "
                f"worlds={item['oracle_stats']['live_edge_samples']} time={item['selection_seconds']:.2f} "
                f"verified={item['verified_per_step']} mc={item['mc_budget_per_step']}", flush=True
            )

    report = {
        "dataset": "NetHEPT",
        "config": vars(args),
        "reference": {
            "full_mc_spread": full_spread,
            "full_mc_candidate_evaluations": full_candidate_evals,
            "full_mc_candidate_samples": full_mc_candidate_samples,
            "fixed_adaptive_beta_0.5_candidate_evaluations": 512,
            "fixed_adaptive_beta_0.5_mc_candidate_samples": fixed_adaptive_samples,
            "fixed_adaptive_beta_0.5_spread": 443.626,
        },
        "methods": methods,
    }
    out_dir = ROOT / "outputs" / "end_to_end" / "progressive_mc_v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved={out}", flush=True)


if __name__ == "__main__":
    main()
