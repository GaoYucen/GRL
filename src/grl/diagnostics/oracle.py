from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import torch

from grl.baselines import select_degree_discount_nodes, select_high_degree_nodes
from grl.diffusion import estimate_marginal_gain, estimate_spread
from grl.evaluation.gnn_metrics import evaluate_trained_gnn
from grl.models import (
    MarginalGainPredictor,
    build_node_features,
    load_or_create_node2vec_embeddings,
)
from grl.models.gnn import SpreadPredictorGNN

ScoreCallback = Callable[[list[int], list[int]], dict[int, float]]


def _marginal_gain(graph, seeds: list[int], candidate: int, mc_runs: int, random_seed: int) -> float:
    return float(estimate_marginal_gain(graph, seeds, candidate, mc_runs, random_seed)["mean"])


def _rank_candidates_by_gain(graph, seeds: list[int], candidates: list[int], mc_runs: int, random_seed: int):
    scores = [(node, _marginal_gain(graph, seeds, node, mc_runs, random_seed)) for node in candidates]
    return sorted(scores, key=lambda item: (-item[1], item[0]))


def _load_scorer(graph_data, config: dict, device: torch.device) -> ScoreCallback:
    model_dir = Path(config.get("gnn", {}).get("model_dir", "param"))
    embeddings = load_or_create_node2vec_embeddings(
        graph_data.graph,
        model_dir / f"marginal_node2vec_{graph_data.name}.pth",
    ).to(device)
    norm_degrees, _ = build_node_features(graph_data.graph, device=device)
    marginal_path = model_dir / f"marginal_gain_{graph_data.name}.pth"
    if marginal_path.exists():
        checkpoint = torch.load(marginal_path, map_location=device)
        model = MarginalGainPredictor(embeddings.shape[1], int(config.get("gnn", {}).get("hidden_dim", 64))).to(device)
        model.load_state_dict(checkpoint.get("state_dict", checkpoint))
        model.eval()

        def score(seeds: list[int], candidates: list[int]) -> dict[int, float]:
            mask = torch.zeros((graph_data.num_nodes, 1), dtype=torch.float32, device=device)
            if seeds:
                mask[seeds] = 1.0
            with torch.no_grad():
                return {
                    node: float(model(embeddings, norm_degrees, mask, node).item())
                    for node in candidates
                }

        return score

    # Keep diagnostics runnable before a marginal checkpoint exists. This is a
    # fallback only; new training/evaluation uses MarginalGainPredictor.
    legacy_path = model_dir / f"gnn_{graph_data.name}.pth"
    try:
        legacy_checkpoint = torch.load(legacy_path, map_location=device)
    except (FileNotFoundError, RuntimeError, OSError):
        return lambda seeds, candidates: {
            node: float(graph_data.graph.out_degree(node) if graph_data.graph.is_directed() else graph_data.graph.degree(node))
            for node in candidates
        }
    model = SpreadPredictorGNN(embeddings.shape[1], int(config.get("gnn", {}).get("hidden_dim", 64))).to(device)
    model.load_state_dict(legacy_checkpoint)
    model.eval()

    def legacy_score(seeds: list[int], candidates: list[int]) -> dict[int, float]:
        base_mask = torch.zeros((graph_data.num_nodes, 1), dtype=torch.float32, device=device)
        if seeds:
            base_mask[seeds] = 1.0
        values = {}
        with torch.no_grad():
            for node in candidates:
                candidate_mask = base_mask.clone()
                candidate_mask[node] = 1.0
                values[node] = float(model(embeddings, norm_degrees, candidate_mask).item())
        return values

    return legacy_score


def run_oracle_diagnostics(graph_data, config: dict, scorer: ScoreCallback | None = None) -> dict[str, Any]:
    budget = min(int(config["seed"].get("budget", 10)), graph_data.num_nodes)
    oracle_cfg = config.get("oracle", {})
    mc_runs = int(oracle_cfg.get("mc_runs", 100))
    random_seed = int(config["experiment"].get("random_seed", 42))
    candidate_pool_size = min(int(oracle_cfg.get("candidate_pool_size", max(20, budget * 3))), graph_data.num_nodes)
    max_nodes = min(int(oracle_cfg.get("max_nodes", graph_data.num_nodes)), graph_data.num_nodes)
    step_limit = min(int(oracle_cfg.get("step_limit", budget)), budget)
    device = torch.device(config.get("gnn", {}).get("device", "cpu"))
    scorer = scorer or _load_scorer(graph_data, config, device)

    degree_pool = select_high_degree_nodes(graph_data.graph, candidate_pool_size)
    degree_discount_pool = select_degree_discount_nodes(
        graph_data.graph, candidate_pool_size, float(config["diffusion"].get("probability", 0.01))
    )
    ranked_nodes = sorted(
        graph_data.graph.out_degree() if graph_data.graph.is_directed() else graph_data.graph.degree(),
        key=lambda item: (-item[1], item[0]),
    )
    node_subset = [node for node, _ in ranked_nodes[:max_nodes]]
    selected_seeds: list[int] = []
    steps = []
    for step in range(step_limit):
        start = time.perf_counter()
        available = [node for node in node_subset if node not in selected_seeds]
        degree_candidates = [node for node in degree_pool if node in available][:candidate_pool_size]
        degree_discount_candidates = [node for node in degree_discount_pool if node in available][:candidate_pool_size]
        model_scores = scorer(selected_seeds, available)
        model_candidates = [node for node, _ in sorted(model_scores.items(), key=lambda item: (-item[1], item[0]))[:candidate_pool_size]]
        combined = list(dict.fromkeys(degree_candidates + degree_discount_candidates + model_candidates))
        global_scores = _rank_candidates_by_gain(graph_data.graph, selected_seeds, available, mc_runs, random_seed + step)
        candidate_scores = _rank_candidates_by_gain(graph_data.graph, selected_seeds, combined, mc_runs, random_seed + step)
        if not global_scores or not candidate_scores:
            break
        global_node, global_gain = global_scores[0]
        candidate_node, candidate_gain = candidate_scores[0]
        model_node = max(combined, key=lambda node: (model_scores.get(node, float("-inf")), -node))
        model_gain = next(gain for node, gain in candidate_scores if node == model_node)
        selected_seeds.append(model_node)
        global_ranks = {node: index + 1 for index, (node, _) in enumerate(global_scores)}
        candidate_loss = float(global_gain - candidate_gain)
        ranking_loss = float(candidate_gain - model_gain)
        total_loss = float(global_gain - model_gain)
        step_result = {
            "step": step + 1,
            "seed_set": selected_seeds[:-1],
            "current_seed_set": selected_seeds[:-1],
            "global_oracle_node": global_node,
            "global_oracle_gain": float(global_gain),
            "global_oracle_best_node": global_node,
            "global_oracle_best_gain": float(global_gain),
            "candidate_oracle_node": candidate_node,
            "candidate_oracle_gain": float(candidate_gain),
            "candidate_best_node": candidate_node,
            "candidate_best_gain": float(candidate_gain),
            "model_selected_node": model_node,
            "model_selected_gain": float(model_gain),
            "selected_node": model_node,
            "selected_gain": float(model_gain),
            "candidate_loss": candidate_loss,
            "ranking_loss": ranking_loss,
            "total_loss": total_loss,
            "total_selection_loss": total_loss,
            "candidate_recall_global_best": global_node in combined,
            "candidate_recall_at_k": float(global_node in combined),
            "candidate_pool_size": len(combined),
            "relative_gain": float(model_gain / global_gain) if global_gain else 0.0,
            "selected_node_oracle_rank": global_ranks.get(model_node),
            "degree_candidate_recall_at_k": float(global_node in degree_candidates),
            "degree_discount_candidate_recall_at_k": float(global_node in degree_discount_candidates),
            "model_candidate_recall_at_k": float(global_node in model_candidates),
            "gnn_selected_node": model_node,
            "gnn_selected_gain": float(model_gain),
            "degree_selected_node": degree_candidates[0] if degree_candidates else None,
            "degree_discount_selected_node": degree_discount_candidates[0] if degree_discount_candidates else None,
            "candidate_nodes": combined,
            "candidate_gains": [{"node": node, "gain": gain, "global_oracle_rank": global_ranks.get(node)} for node, gain in candidate_scores],
            "step_runtime": time.perf_counter() - start,
        }
        steps.append(step_result)
    final_spread = estimate_spread(graph_data.graph, selected_seeds, int(config["diffusion"].get("mc_runs_eval", 100)), random_seed)["mean"]
    return {
        "dataset": graph_data.name,
        "budget": budget,
        "step_limit": step_limit,
        "candidate_pool_size": candidate_pool_size,
        "max_nodes": max_nodes,
        "selected_seeds": selected_seeds,
        "final_spread": final_spread,
        "steps": steps,
    }
