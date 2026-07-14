from __future__ import annotations

import random
from statistics import pstdev

import networkx as nx


def run_independent_cascade(
    graph: nx.Graph | nx.DiGraph,
    seeds: list[int],
    rng: random.Random,
) -> int:
    activated = set(seeds)
    frontier = list(seeds)

    while frontier:
        current = frontier.pop(0)
        for neighbor in graph.neighbors(current):
            if neighbor in activated:
                continue
            probability = float(graph[current][neighbor].get("weight", 0.0))
            if rng.random() < probability:
                activated.add(neighbor)
                frontier.append(neighbor)

    return len(activated)


def estimate_spread(
    graph: nx.Graph | nx.DiGraph,
    seeds: list[int],
    mc_runs: int,
    random_seed: int,
) -> dict[str, float]:
    if mc_runs <= 0:
        raise ValueError("mc_runs must be positive")

    spreads = []
    for offset in range(mc_runs):
        rng = random.Random(random_seed + offset)
        spreads.append(run_independent_cascade(graph, seeds, rng))

    mean = sum(spreads) / len(spreads)
    std = pstdev(spreads) if len(spreads) > 1 else 0.0
    return {"mean": float(mean), "std": float(std)}
