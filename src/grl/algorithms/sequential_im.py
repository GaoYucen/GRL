from __future__ import annotations

from dataclasses import dataclass, asdict
import math


@dataclass
class SequentialSelectionResult:
    selected_seeds: list[int]
    steps: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


def _available(candidate_pool: list[int], selected: list[int]) -> list[int]:
    chosen = set(selected)
    return [v for v in candidate_pool if v not in chosen]


def full_oracle_greedy(candidate_pool: list[int], budget: int, oracle) -> SequentialSelectionResult:
    selected: list[int] = []
    steps: list[dict] = []
    for step in range(int(budget)):
        available = _available(candidate_pool, selected)
        if not available:
            break
        scores = oracle.score(selected, available, step=step)
        chosen = max(available, key=lambda v: (scores[v], -v))
        steps.append({"step": step + 1, "chosen": chosen, "score": float(scores[chosen]), "verified": len(available)})
        selected.append(chosen)
    return SequentialSelectionResult(selected, steps)


def learned_greedy(candidate_pool: list[int], budget: int, learned_oracle) -> SequentialSelectionResult:
    selected: list[int] = []
    steps: list[dict] = []
    for step in range(int(budget)):
        available = _available(candidate_pool, selected)
        if not available:
            break
        scores = learned_oracle.score(selected, available, step=step)
        chosen = max(available, key=lambda v: (scores[v], -v))
        steps.append({"step": step + 1, "chosen": chosen, "predicted_score": float(scores[chosen]), "verified": 0})
        selected.append(chosen)
    return SequentialSelectionResult(selected, steps)


def selective_greedy(
    candidate_pool: list[int],
    budget: int,
    learned_oracle,
    exact_oracle,
    top_m: int = 8,
) -> SequentialSelectionResult:
    """Prediction-guided greedy with fixed Top-M exact refinement."""
    selected: list[int] = []
    steps: list[dict] = []
    for step in range(int(budget)):
        available = _available(candidate_pool, selected)
        if not available:
            break
        learned = learned_oracle.score(selected, available, step=step)
        shortlist = sorted(available, key=lambda v: (learned[v], -v), reverse=True)[: min(int(top_m), len(available))]
        exact = exact_oracle.score(selected, shortlist, step=step)
        chosen = max(shortlist, key=lambda v: (exact[v], -v))
        steps.append({
            "step": step + 1,
            "chosen": chosen,
            "predicted_score": float(learned[chosen]),
            "oracle_score": float(exact[chosen]),
            "verified": len(shortlist),
            "shortlist": shortlist,
        })
        selected.append(chosen)
    return SequentialSelectionResult(selected, steps)


def adaptive_selective_greedy(
    candidate_pool: list[int],
    budget: int,
    learned_oracle,
    exact_oracle,
    initial_m: int = 8,
    batch_m: int = 8,
    residual_beta: float = 1.0,
    min_rounds: int = 2,
    max_m: int | None = None,
) -> SequentialSelectionResult:
    """Adaptive prediction-guided refinement with safe full-oracle fallback.

    Candidates are ranked once by the learned oracle at each greedy step. Exact
    evaluation starts from a small prefix and expands in batches. A provisional
    winner is accepted only when (1) it is stable across ``min_rounds`` expansion
    rounds and (2) its exact score exceeds an empirical upper envelope for the
    best unverified candidate. The envelope is the learned outsider score plus
    the largest observed exact-minus-learned residual and ``residual_beta``
    residual standard deviations. If this never certifies, the method expands to
    ``max_m``; with max_m=None it can fall back to all remaining candidates.

    This is an operational first certification baseline, not yet a formal
    probabilistic guarantee.
    """
    initial_m = max(1, int(initial_m))
    batch_m = max(1, int(batch_m))
    min_rounds = max(1, int(min_rounds))
    selected: list[int] = []
    steps: list[dict] = []

    for step in range(int(budget)):
        available = _available(candidate_pool, selected)
        if not available:
            break
        learned = learned_oracle.score(selected, available, step=step)
        ranked = sorted(available, key=lambda v: (learned[v], -v), reverse=True)
        cap = len(ranked) if max_m is None else min(len(ranked), max(1, int(max_m)))
        verified: dict[int, float] = {}
        winner_history: list[int] = []
        rounds: list[dict] = []
        target = min(initial_m, cap)
        certified = False
        stop_reason = "max_m"

        while True:
            new_nodes = ranked[len(verified):target]
            if new_nodes:
                exact_batch = exact_oracle.score(selected, new_nodes, step=step)
                verified.update(exact_batch)
            winner = max(verified, key=lambda v: (verified[v], -v))
            winner_history.append(winner)

            residuals = [verified[v] - learned[v] for v in verified]
            residual_max = max(residuals) if residuals else 0.0
            residual_mean = sum(residuals) / len(residuals) if residuals else 0.0
            residual_var = (
                sum((x - residual_mean) ** 2 for x in residuals) / len(residuals)
                if residuals else 0.0
            )
            residual_std = math.sqrt(max(0.0, residual_var))

            outsider = ranked[target] if target < len(ranked) else None
            outsider_upper = None
            if outsider is not None:
                outsider_upper = float(learned[outsider] + residual_max + float(residual_beta) * residual_std)

            stable = (
                len(winner_history) >= min_rounds
                and len(set(winner_history[-min_rounds:])) == 1
            )
            if outsider is None:
                certified = True
                stop_reason = "all_candidates"
            elif stable and float(verified[winner]) >= float(outsider_upper):
                certified = True
                stop_reason = "residual_envelope"

            rounds.append({
                "verified": len(verified),
                "winner": winner,
                "winner_exact": float(verified[winner]),
                "winner_predicted": float(learned[winner]),
                "residual_max": float(residual_max),
                "residual_std": float(residual_std),
                "best_unverified": outsider,
                "best_unverified_predicted": None if outsider is None else float(learned[outsider]),
                "best_unverified_upper": outsider_upper,
                "stable": bool(stable),
                "certified": bool(certified),
            })

            if certified:
                break
            if target >= cap:
                stop_reason = "max_m" if cap < len(ranked) else "all_candidates"
                break
            target = min(cap, target + batch_m)

        chosen = max(verified, key=lambda v: (verified[v], -v))
        steps.append({
            "step": step + 1,
            "chosen": chosen,
            "predicted_score": float(learned[chosen]),
            "oracle_score": float(verified[chosen]),
            "verified": len(verified),
            "certified": bool(certified),
            "stop_reason": stop_reason,
            "rounds": rounds,
            "shortlist": ranked[:len(verified)],
        })
        selected.append(chosen)

    return SequentialSelectionResult(selected, steps)
