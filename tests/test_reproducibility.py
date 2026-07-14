import networkx as nx

from grl.baselines import select_degree_discount_nodes


def test_degree_discount_is_deterministic():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, weight=0.01)
    graph.add_edge(0, 2, weight=0.01)
    graph.add_edge(2, 3, weight=0.01)

    first = select_degree_discount_nodes(graph, 2, probability=0.01)
    second = select_degree_discount_nodes(graph, 2, probability=0.01)

    assert first == second