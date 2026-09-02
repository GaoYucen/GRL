from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .gnn_metrics import kendall_tau, spearman_correlation


@dataclass(frozen=True)
class CandidateRankingGroup:
    """Predictions and oracle gains for one fixed graph/seed state."""

    seed_set: tuple[int, ...]
    candidates: tuple[int, ...]
    predictions: tuple[float, ...]
    targets: tuple[float, ...]

    def __post_init__(self) -> None:
        n = len(self.candidates)
        if n != len(self.predictions) or n != len(self.targets):
            raise ValueError("candidates, predictions, and targets must have equal length")
        if n == 0:
            raise ValueError("a ranking group must contain at least one candidate")


def _top_indices(values: Sequence[float], k: int) -> list[int]:
    return sorted(range(len(values)), key=lambda i: (-values[i], i))[: max(1, min(k, len(values)))]


def evaluate_conditional_rankings(
    groups: Iterable[CandidateRankingGroup],
    top_ks: Sequence[int] = (1, 5, 10),
) -> dict[str, float | int]:
    groups = list(groups)
    if not groups:
        raise ValueError("at least one ranking group is required")

    spearmans: list[float] = []
    kendalls: list[float] = []
    top1_hits = 0
    regrets: list[float] = []
    recall_sums = {int(k): 0.0 for k in top_ks}

    for group in groups:
        pred = list(group.predictions)
        target = list(group.targets)
        pred_order = _top_indices(pred, 1)[0]
        oracle_order = _top_indices(target, 1)[0]
        top1_hits += int(pred_order == oracle_order)
        best = max(target)
        regrets.append(float(best - target[pred_order]))
        spearmans.append(spearman_correlation(pred, target))
        kendalls.append(kendall_tau(pred, target))
        oracle_top = set(_top_indices(target, max(top_ks)))
        for k in top_ks:
            selected = set(_top_indices(pred, int(k)))
            denom = min(int(k), len(target))
            recall_sums[int(k)] += len(selected & oracle_top) / denom

    n = len(groups)
    result: dict[str, float | int] = {
        "groups": n,
        "conditional_spearman": sum(spearmans) / n,
        "conditional_kendall": sum(kendalls) / n,
        "top1_accuracy": top1_hits / n,
        "mean_regret": sum(regrets) / n,
    }
    for k, value in recall_sums.items():
        result[f"recall_at_{k}"] = value / n
    return result
