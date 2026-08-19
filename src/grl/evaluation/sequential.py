from __future__ import annotations

from typing import Callable

from grl.diffusion import estimate_marginal_gain, estimate_spread


def evaluate_sequential_selector(
    graph_data,
    budget: int,
    scorer: Callable[[list[int], list[int]], dict[int, float]],
    mc_runs: int = 100,
    random_seed: int = 42,
) -> dict:
    selected: list[int] = []
    steps = []
    for step in range(min(budget, graph_data.num_nodes)):
        available = [node for node in range(graph_data.num_nodes) if node not in selected]
        if not available:
            break
        scores = scorer(selected, available)
        chosen = max(available, key=lambda node: (scores.get(node, float("-inf")), -node))
        oracle_gains = {
            node: estimate_marginal_gain(graph_data.graph, selected, node, mc_runs, random_seed + step)["mean"]
            for node in available
        }
        oracle_node = max(available, key=lambda node: (oracle_gains[node], -node))
        model_gain = oracle_gains[chosen]
        oracle_gain = oracle_gains[oracle_node]
        selected.append(chosen)
        steps.append({
            "step": step + 1,
            "seed_set": selected[:-1],
            "oracle_node": oracle_node,
            "selected_node": chosen,
            "oracle_gain": float(oracle_gain),
            "selected_gain": float(model_gain),
            "regret": float(oracle_gain - model_gain),
            "ratio": float(model_gain / oracle_gain) if oracle_gain else 0.0,
        })
    final_spread = estimate_spread(graph_data.graph, selected, mc_runs, random_seed)["mean"]
    return {
        "selected_seeds": selected,
        "steps": steps,
        "mean_step_regret": sum(item["regret"] for item in steps) / max(len(steps), 1),
        "cumulative_regret": sum(item["regret"] for item in steps),
        "final_spread": final_spread,
    }
