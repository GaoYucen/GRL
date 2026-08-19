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


def estimate_marginal_gain(
    graph: nx.Graph | nx.DiGraph,
    seeds: list[int],
    candidate: int,
    mc_runs: int,
    random_seed: int,
) -> dict[str, float]:
    """Estimate Delta(candidate | seeds) with common live-edge samples.

    Sampling each edge once per MC run makes the base and extended seed sets
    use exactly the same random cascade realization, so the marginal label is
    monotone up to floating-point noise.
    """
    if mc_runs <= 0:
        raise ValueError("mc_runs must be positive")

    gains = []
    seed_set = set(seeds)
    for offset in range(mc_runs):
        rng = random.Random(random_seed + offset)
        live_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()
        live_graph.add_nodes_from(graph.nodes())
        for u, v, data in graph.edges(data=True):
            if rng.random() < float(data.get("weight", 0.0)):
                live_graph.add_edge(u, v)

        base_reached = set()
        for seed in seed_set:
            base_reached.update(nx.descendants(live_graph, seed) if live_graph.is_directed() else nx.node_connected_component(live_graph, seed))
            base_reached.add(seed)
        extended_reached = set(base_reached)
        extended_reached.update(nx.descendants(live_graph, candidate) if live_graph.is_directed() else nx.node_connected_component(live_graph, candidate))
        extended_reached.add(candidate)
        gains.append(float(len(extended_reached) - len(base_reached)))

    mean = sum(gains) / len(gains)
    return {"mean": float(mean), "std": float(pstdev(gains) if len(gains) > 1 else 0.0)}
