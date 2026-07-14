from __future__ import annotations

import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from typing import Any

import torch

from grl.baselines import select_degree_discount_nodes, select_high_degree_nodes
from grl.diffusion import estimate_spread
from grl.evaluation.gnn_metrics import evaluate_trained_gnn
from grl.models import SpreadPredictorGNN, build_node_features, load_or_create_node2vec_embeddings


def _marginal_gain(graph, seeds: list[int], candidate: int, mc_runs: int, random_seed: int) -> float:
    base = estimate_spread(graph, seeds, mc_runs=mc_runs, random_seed=random_seed)["mean"]
    extended = estimate_spread(graph, seeds + [candidate], mc_runs=mc_runs, random_seed=random_seed)["mean"]
    return float(extended - base)


def _rank_candidates_by_gain(graph, seeds: list[int], candidates: list[int], mc_runs: int, random_seed: int) -> list[tuple[int, float]]:
    gains = [
        (node, _marginal_gain(graph, seeds, node, mc_runs, random_seed))
        for node in candidates
    ]
    gains.sort(key=lambda item: (-item[1], item[0]))
    return gains


def run_oracle_diagnostics(graph_data, config: dict) -> dict[str, Any]:
    budget = min(int(config["seed"].get("budget", 10)), graph_data.num_nodes)
    oracle_cfg = config.get("oracle", {})
    mc_runs = int(oracle_cfg.get("mc_runs", 100))
    random_seed = int(config["experiment"].get("random_seed", 42))
    candidate_pool_size = min(int(oracle_cfg.get("candidate_pool_size", max(20, budget * 3))), graph_data.num_nodes)
    max_nodes = min(int(oracle_cfg.get("max_nodes", graph_data.num_nodes)), graph_data.num_nodes)
    step_limit = min(int(oracle_cfg.get("step_limit", budget)), budget)

    degree_pool = select_high_degree_nodes(graph_data.graph, candidate_pool_size)
    degree_discount_pool = select_degree_discount_nodes(graph_data.graph, candidate_pool_size, float(config["diffusion"].get("probability", 0.01)))

    device = torch.device(config.get("gnn", {}).get("device", "cpu"))
    model_dir = config.get("gnn", {}).get("model_dir", "param")
    embeddings = load_or_create_node2vec_embeddings(graph_data.graph, f"{model_dir}/node2vec_{graph_data.name}.pth").to(device)
    norm_degrees, _ = build_node_features(graph_data.graph, device=device)
    model = SpreadPredictorGNN(embedding_dim=embeddings.shape[1], hidden_dim=int(config["gnn"].get("hidden_dim", 64))).to(device)
    model.load_state_dict(torch.load(f"{model_dir}/gnn_{graph_data.name}.pth", map_location=device))
    model.eval()

    ranked_nodes = sorted(
        graph_data.graph.out_degree() if graph_data.graph.is_directed() else graph_data.graph.degree(),
        key=lambda x: (-x[1], x[0]),
    )
    node_subset = [node for node, _ in ranked_nodes[:max_nodes]]
    remaining = list(node_subset)
    selected_seeds: list[int] = []
    steps = []
    for step in range(step_limit):
        step_start = time.perf_counter()
        current_spread = estimate_spread(graph_data.graph, selected_seeds, mc_runs=mc_runs, random_seed=random_seed + step)["mean"]
        available_nodes = [node for node in remaining if node not in selected_seeds]

        degree_candidates = [node for node in degree_pool if node not in selected_seeds and node in remaining][:candidate_pool_size]
        degree_discount_candidates = [
            node
            for node in degree_discount_pool
            if node not in selected_seeds and node in remaining
        ][:candidate_pool_size]

        gnn_candidates_all = []
        with torch.no_grad():
            base_mask = torch.zeros((graph_data.num_nodes, 1), dtype=torch.float32, device=device)
            if selected_seeds:
                base_mask[selected_seeds] = 1.0
            for node in remaining:
                if node in selected_seeds:
                    continue
                cand_mask = base_mask.clone()
                cand_mask[node] = 1.0
                pred = model(embeddings, norm_degrees, cand_mask).item()
                gnn_candidates_all.append((node, pred))
        gnn_candidates_all.sort(key=lambda x: x[1], reverse=True)
        gnn_candidates = [node for node, _ in gnn_candidates_all[:candidate_pool_size]]

        combined_candidates = list(dict.fromkeys(degree_candidates + degree_discount_candidates + gnn_candidates))
        global_oracle_scores = _rank_candidates_by_gain(
            graph_data.graph,
            selected_seeds,
            available_nodes,
            mc_runs,
            random_seed + step,
        )
        candidate_scores = _rank_candidates_by_gain(
            graph_data.graph,
            selected_seeds,
            combined_candidates,
            mc_runs,
            random_seed + step,
        )

        global_oracle_best_node, global_oracle_best_gain = global_oracle_scores[0]
        candidate_best_node, candidate_best_gain = candidate_scores[0]
        selected_node, selected_gain = candidate_best_node, candidate_best_gain
        selected_seeds.append(selected_node)

        global_rank_lookup = {node: idx + 1 for idx, (node, _) in enumerate(global_oracle_scores)}
        candidate_gain_lookup = {node: gain for node, gain in candidate_scores}
        candidate_recall_at_k = float(global_oracle_best_node in combined_candidates)
        candidate_loss = float(global_oracle_best_gain - candidate_best_gain)
        ranking_loss = float(candidate_best_gain - selected_gain)
        relative_gain = float(selected_gain / global_oracle_best_gain) if global_oracle_best_gain else 0.0
        step_runtime = time.perf_counter() - step_start

        steps.append(
            {
                "step": step + 1,
                "current_seed_set": selected_seeds[:-1],
                "current_spread": current_spread,
                "global_oracle_best_node": global_oracle_best_node,
                "global_oracle_best_gain": global_oracle_best_gain,
                "candidate_best_node": candidate_best_node,
                "candidate_best_gain": candidate_best_gain,
                "candidate_recall_at_k": candidate_recall_at_k,
                "candidate_pool_size": len(combined_candidates),
                "degree_candidate_recall_at_k": float(global_oracle_best_node in degree_candidates),
                "gnn_candidate_recall_at_k": float(global_oracle_best_node in gnn_candidates),
                "degree_discount_candidate_recall_at_k": float(global_oracle_best_node in degree_discount_candidates),
                "selected_node": selected_node,
                "selected_gain": selected_gain,
                "relative_gain": relative_gain,
                "candidate_loss": candidate_loss,
                "ranking_loss": ranking_loss,
                "total_selection_loss": float(candidate_loss + ranking_loss),
                "selected_node_oracle_rank": global_rank_lookup.get(selected_node),
                "step_runtime": step_runtime,
                "candidate_nodes": combined_candidates,
                "candidate_gains": [
                    {"node": node, "gain": gain, "global_oracle_rank": global_rank_lookup.get(node)}
                    for node, gain in candidate_scores
                ],
                "degree_selected_node": degree_candidates[0] if degree_candidates else None,
                "degree_selected_gain": candidate_gain_lookup.get(degree_candidates[0]) if degree_candidates else None,
                "degree_selected_oracle_rank": global_rank_lookup.get(degree_candidates[0]) if degree_candidates else None,
                "degree_discount_selected_node": degree_discount_candidates[0] if degree_discount_candidates else None,
                "degree_discount_selected_gain": candidate_gain_lookup.get(degree_discount_candidates[0]) if degree_discount_candidates else None,
                "degree_discount_selected_oracle_rank": global_rank_lookup.get(degree_discount_candidates[0]) if degree_discount_candidates else None,
                "gnn_selected_node": gnn_candidates[0] if gnn_candidates else None,
                "gnn_selected_gain": candidate_gain_lookup.get(gnn_candidates[0]) if gnn_candidates else None,
                "gnn_selected_oracle_rank": global_rank_lookup.get(gnn_candidates[0]) if gnn_candidates else None,
            }
        )

    final_spread = estimate_spread(graph_data.graph, selected_seeds, int(config["diffusion"].get("mc_runs_eval", 100)), random_seed)["mean"]
    gnn_eval = evaluate_trained_gnn(graph_data, config)
    return {
        "dataset": graph_data.name,
        "budget": budget,
        "step_limit": step_limit,
        "candidate_pool_size": candidate_pool_size,
        "max_nodes": max_nodes,
        "selected_seeds": selected_seeds,
        "final_spread": final_spread,
        "steps": steps,
        "gnn_summary": {
            "spearman": gnn_eval["spearman"],
            "kendall": gnn_eval["kendall"],
            "topk_recall": gnn_eval["topk_recall"],
        },
    }
