from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PredictionSummary:
    mean: torch.Tensor
    std: torch.Tensor
    lower_bound: torch.Tensor


def summarize_ensemble(
    predictions: torch.Tensor,
    confidence_scale: float = 1.96,
) -> PredictionSummary:
    """Summarize ensemble predictions for risk-sensitive candidate selection.

    Args:
        predictions: Tensor shaped [ensemble_members, candidates].
        confidence_scale: Multiplier for the empirical uncertainty penalty.
    """
    if predictions.ndim != 2:
        raise ValueError("predictions must have shape [ensemble_members, candidates]")
    if predictions.shape[0] == 0 or predictions.shape[1] == 0:
        raise ValueError("predictions must contain members and candidates")
    if confidence_scale < 0:
        raise ValueError("confidence_scale must be non-negative")
    mean = predictions.mean(dim=0)
    std = predictions.std(dim=0, unbiased=False)
    return PredictionSummary(mean=mean, std=std, lower_bound=mean - confidence_scale * std)


def conservative_order(
    predictions: torch.Tensor,
    confidence_scale: float = 1.96,
) -> torch.Tensor:
    """Return candidate indices ordered by descending conservative score."""
    summary = summarize_ensemble(predictions, confidence_scale)
    return torch.argsort(summary.lower_bound, descending=True)
