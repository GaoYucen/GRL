import networkx as nx

from grl.diffusion import estimate_spread


def test_diffusion_is_reproducible_with_same_seed():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, weight=1.0)
    graph.add_edge(1, 2, weight=1.0)

    first = estimate_spread(graph, [0], mc_runs=5, random_seed=123)
    second = estimate_spread(graph, [0], mc_runs=5, random_seed=123)

    assert first == second
