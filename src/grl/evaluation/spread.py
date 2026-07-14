from __future__ import annotations

import time
from typing import Callable

import networkx as nx

from grl.diffusion import estimate_spread


def evaluate_baseline_method(
    graph: nx.Graph | nx.DiGraph,
    method_name: str,
    selector: Callable[[], list[int]],
    mc_runs: int,
    random_seed: int,
) -> dict:
    selection_start = time.perf_counter()
    seeds = selector()
    selection_time_seconds = time.perf_counter() - selection_start

    evaluation_start = time.perf_counter()
    spread = estimate_spread(graph, seeds, mc_runs=mc_runs, random_seed=random_seed)
    evaluation_time_seconds = time.perf_counter() - evaluation_start

    return {
        "method": method_name,
        "selected_seeds": seeds,
        "spread_mean": spread["mean"],
        "spread_std": spread["std"],
        "selection_time_seconds": selection_time_seconds,
        "evaluation_time_seconds": evaluation_time_seconds,
    }
