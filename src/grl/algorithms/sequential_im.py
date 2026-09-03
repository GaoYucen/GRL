from __future__ import annotations

from dataclasses import dataclass, asdict


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
    """Prediction-guided greedy with fixed Top-M exact refinement.

    This is the first end-to-end baseline. The fixed Top-M gate is intentionally
    simple so it can later be replaced by uncertainty/gap-driven certification.
    """
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
