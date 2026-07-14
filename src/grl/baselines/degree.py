from __future__ import annotations

import networkx as nx


def validate_budget(graph: nx.Graph | nx.DiGraph, budget: int) -> None:
    if budget <= 0:
        raise ValueError("budget must be positive")
    if graph.number_of_nodes() == 0:
        raise ValueError("graph is empty")
    if budget > graph.number_of_nodes():
        raise ValueError("budget cannot exceed number of graph nodes")


def select_high_degree_nodes(graph: nx.Graph | nx.DiGraph, budget: int) -> list[int]:
    validate_budget(graph, budget)
    ranked = sorted(graph.out_degree if graph.is_directed() else graph.degree, key=lambda item: (-item[1], item[0]))
    return [node for node, _ in ranked[:budget]]
