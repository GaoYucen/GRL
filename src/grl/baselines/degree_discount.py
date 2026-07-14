from __future__ import annotations

import networkx as nx

from .degree import validate_budget


def select_degree_discount_nodes(
    graph: nx.Graph | nx.DiGraph,
    budget: int,
    probability: float,
) -> list[int]:
    validate_budget(graph, budget)
    degree_fn = graph.out_degree if graph.is_directed() else graph.degree
    d = {node: degree for node, degree in degree_fn()}
    dd = d.copy()
    t = {node: 0 for node in graph.nodes()}
    selected: list[int] = []

    for _ in range(budget):
        node = max(dd, key=lambda candidate: (dd[candidate], -candidate))
        selected.append(node)
        dd.pop(node)
        for neighbor in graph.neighbors(node):
            if neighbor in dd:
                t[neighbor] += 1
                dd[neighbor] = d[neighbor] - 2 * t[neighbor] - (d[neighbor] - t[neighbor]) * t[neighbor] * probability

    return selected
