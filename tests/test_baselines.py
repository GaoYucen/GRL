import networkx as nx
import pytest

from grl.baselines import select_degree_discount_nodes, select_high_degree_nodes


def build_graph():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3)])
    for u, v in graph.edges():
        graph[u][v]["weight"] = 0.01
    return graph


def test_degree_returns_unique_legal_nodes():
    graph = build_graph()
    seeds = select_high_degree_nodes(graph, 2)
    assert len(seeds) == 2
    assert len(set(seeds)) == 2
    assert all(seed in graph.nodes for seed in seeds)


def test_degree_discount_returns_unique_legal_nodes():
    graph = build_graph()
    seeds = select_degree_discount_nodes(graph, 2, probability=0.01)
    assert len(seeds) == 2
    assert len(set(seeds)) == 2
    assert all(seed in graph.nodes for seed in seeds)


def test_budget_larger_than_graph_raises():
    graph = build_graph()
    with pytest.raises(ValueError):
        select_high_degree_nodes(graph, 10)
