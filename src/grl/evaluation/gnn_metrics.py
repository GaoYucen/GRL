from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from grl.baselines import select_degree_discount_nodes, select_high_degree_nodes
from grl.diffusion import estimate_spread
from grl.models import SpreadPredictorGNN, build_node_features, load_or_create_node2vec_embeddings


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
    embedding_path = model_dir / f"node2vec_{graph_data.name}.pth"
    model_path = model_dir / f"gnn_{graph_data.name}.pth"
    embeddings = load_or_create_node2vec_embeddings(graph_data.graph, embedding_path).to(device)
    norm_degrees, _ = build_node_features(graph_data.graph, device=device)
    model = SpreadPredictorGNN(embedding_dim=embeddings.shape[1], hidden_dim=int(config["gnn"].get("hidden_dim", 64))).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    candidate_count = min(int(config["gnn"].get("eval_node_count", 64)), graph_data.num_nodes)
    ranked = sorted(graph_data.graph.out_degree() if graph_data.graph.is_directed() else graph_data.graph.degree(), key=lambda x: (-x[1], x[0]))
    top_nodes = [node for node, _ in ranked[: candidate_count // 2]]
    tail_nodes = [node for node, _ in ranked[-(candidate_count - len(top_nodes)):]]
    nodes = list(dict.fromkeys(top_nodes + tail_nodes))
    mc_eval = int(config["gnn"].get("mc_runs_eval", config["diffusion"].get("mc_runs_eval", 100)))
    base_seed = int(config["experiment"].get("random_seed", 42))

    preds, trues = [], []
    for idx, node in enumerate(nodes):
        mask = torch.zeros((graph_data.num_nodes, 1), dtype=torch.float32, device=device)
        mask[node] = 1.0
        with torch.no_grad():
            pred = model(embeddings, norm_degrees, mask).item()
        true = estimate_spread(graph_data.graph, [node], mc_runs=mc_eval, random_seed=base_seed + idx)["mean"]
        preds.append(pred)
        trues.append(true)

    mae = float(np.mean(np.abs(np.array(preds) - np.array(trues)))) if preds else 0.0
    rmse = float(math.sqrt(np.mean((np.array(preds) - np.array(trues)) ** 2))) if preds else 0.0
    spearman = spearman_correlation(preds, trues)
    kendall = kendall_tau(preds, trues)

    top_k = min(int(config["seed"].get("budget", 10)), len(nodes))
    pred_top = {node for node, _ in sorted(zip(nodes, preds), key=lambda x: x[1], reverse=True)[:top_k]}
    true_top = {node for node, _ in sorted(zip(nodes, trues), key=lambda x: x[1], reverse=True)[:top_k]}
    topk_recall = float(len(pred_top & true_top) / max(len(true_top), 1))

    degree_seeds = select_high_degree_nodes(graph_data.graph, min(int(config["seed"].get("budget", 10)), graph_data.num_nodes))
    degree_discount_seeds = select_degree_discount_nodes(graph_data.graph, min(int(config["seed"].get("budget", 10)), graph_data.num_nodes), float(config["diffusion"].get("probability", 0.01)))
    gnn_ranked_nodes = [node for node, _ in sorted(zip(nodes, preds), key=lambda x: x[1], reverse=True)]
    gnn_seeds = gnn_ranked_nodes[: min(int(config["seed"].get("budget", 10)), len(gnn_ranked_nodes))]

    spread_eval_runs = int(config["diffusion"].get("mc_runs_eval", 100))
    degree_spread = estimate_spread(graph_data.graph, degree_seeds, spread_eval_runs, base_seed)["mean"]
    degree_discount_spread = estimate_spread(graph_data.graph, degree_discount_seeds, spread_eval_runs, base_seed)["mean"]
    gnn_spread = estimate_spread(graph_data.graph, gnn_seeds, spread_eval_runs, base_seed)["mean"]

    return {
        "dataset": graph_data.name,
        "evaluated_nodes": len(nodes),
        "mae": mae,
        "rmse": rmse,
        "spearman": spearman,
        "kendall": kendall,
        "topk_recall": topk_recall,
        "gnn_selected_seeds": gnn_seeds,
        "gnn_spread": gnn_spread,
        "degree_spread": degree_spread,
        "degree_discount_spread": degree_discount_spread,
    }
