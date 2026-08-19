from __future__ import annotations

import math

import numpy as np

from .gnn_metrics import kendall_tau, spearman_correlation


def pairwise_accuracy(predictions: list[float], targets: list[float]) -> float:
    total = correct = 0
    for i in range(len(targets)):
        for j in range(i + 1, len(targets)):
            target_delta = targets[i] - targets[j]
            if target_delta == 0:
                continue
            total += 1
            pred_delta = predictions[i] - predictions[j]
            if pred_delta == 0 or pred_delta * target_delta > 0:
                correct += 0.5 if pred_delta == 0 else 1
    return float(correct / total) if total else 0.0


def top_k_recall(predictions: list[float], targets: list[float], k: int) -> float:
    if not targets:
        return 0.0
    k = min(max(int(k), 1), len(targets))
    pred_top = set(np.argsort(np.asarray(predictions))[-k:])
    true_top = set(np.argsort(np.asarray(targets))[-k:])
    return float(len(pred_top & true_top) / len(true_top))


def regression_ranking_metrics(predictions: list[float], targets: list[float], top_ks=(1, 5, 10)) -> dict[str, float]:
    if not predictions:
        return {"mae": 0.0, "rmse": 0.0, "spearman": 0.0, "kendall": 0.0, "pairwise_accuracy": 0.0}
    errors = np.asarray(predictions, dtype=float) - np.asarray(targets, dtype=float)
    result = {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(math.sqrt(np.mean(errors ** 2))),
        "spearman": spearman_correlation(predictions, targets),
        "kendall": kendall_tau(predictions, targets),
        "pairwise_accuracy": pairwise_accuracy(predictions, targets),
    }
    for k in top_ks:
        result[f"top_{k}_recall"] = top_k_recall(predictions, targets, k)
    return result
