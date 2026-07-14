from grl.evaluation.gnn_metrics import kendall_tau, spearman_correlation
from grl.models.gnn import _build_fallback_embeddings
import networkx as nx


def test_rank_correlations_basic():
    x = [1.0, 2.0, 3.0, 4.0]
    y = [10.0, 20.0, 30.0, 40.0]
    assert spearman_correlation(x, y) == 1.0
    assert kendall_tau(x, y) == 1.0


def test_fallback_embeddings_shape():
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    emb = _build_fallback_embeddings(graph)
    assert emb.shape == (3, 64)
