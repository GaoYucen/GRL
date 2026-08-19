from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from grl.models import MarginalGainPredictor, build_node_features, load_or_create_node2vec_embeddings
from grl.training.marginal_dataset import build_marginal_dataset


def _rankdata(values: list[float]) -> list[float]:
    sorted_pairs = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(sorted_pairs):
        j = i
        while j + 1 < len(sorted_pairs) and sorted_pairs[j + 1][1] == sorted_pairs[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[sorted_pairs[k][0]] = avg_rank
        i = j + 1
    return ranks


def _pearson(x: list[float], y: list[float]) -> float:
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    if x_arr.size == 0:
        return 0.0
    if np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return 0.0
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def spearman_correlation(x: list[float], y: list[float]) -> float:
    return _pearson(_rankdata(x), _rankdata(y))


def kendall_tau(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            prod = dx * dy
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
    denom = n * (n - 1) / 2
    return float((concordant - discordant) / denom) if denom else 0.0


def evaluate_trained_gnn(graph_data, config: dict) -> dict:
    device = torch.device(config.get("gnn", {}).get("device", "cpu"))
    model_dir = Path(config.get("gnn", {}).get("model_dir", "param"))
    embedding_path = model_dir / f"marginal_node2vec_{graph_data.name}.pth"
    model_path = model_dir / f"marginal_gain_{graph_data.name}.pth"
    embeddings = load_or_create_node2vec_embeddings(graph_data.graph, embedding_path).to(device)
    norm_degrees, _ = build_node_features(graph_data.graph, device=device)
    checkpoint = torch.load(model_path, map_location=device)
    model = MarginalGainPredictor(embeddings.shape[1], int(config["gnn"].get("hidden_dim", 64))).to(device)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    model.eval()
    test_samples = build_marginal_dataset(graph_data, config)["test"]
    predictions, targets = [], []
    for sample in test_samples:
        mask = torch.zeros((graph_data.num_nodes, 1), dtype=torch.float32, device=device)
        if sample.seed_set:
            mask[sample.seed_set] = 1.0
        with torch.no_grad():
            predictions.append(float(model(embeddings, norm_degrees, mask, sample.candidate).item()))
        targets.append(sample.marginal_gain)
    from .ranking import regression_ranking_metrics
    metrics = regression_ranking_metrics(predictions, targets, top_ks=(1, 5, 10))
    return {"dataset": graph_data.name, "evaluated_samples": len(test_samples), **metrics}


evaluate_marginal_gain_predictor = evaluate_trained_gnn
