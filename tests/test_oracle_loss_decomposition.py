import networkx as nx

from grl.data import GraphData
from grl.diagnostics.oracle import run_oracle_diagnostics


def _data():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, weight=1.0)
    graph.add_edge(1, 3, weight=1.0)
    graph.add_edge(0, 2, weight=1.0)
    return GraphData("toy", "toy.txt", True, graph, 4, 3, 0, 0, 1.5, 2, 1, 4)


def _config(pool_size):
    return {
        "seed": {"budget": 1},
        "oracle": {"mc_runs": 1, "candidate_pool_size": pool_size, "max_nodes": 4, "step_limit": 1},
        "experiment": {"random_seed": 3},
        "diffusion": {"probability": 1.0, "mc_runs_eval": 1},
    }


def test_retrieval_loss_is_separate_from_ranking_loss(monkeypatch):
    monkeypatch.setattr("grl.diagnostics.oracle.select_high_degree_nodes", lambda *args, **kwargs: [1])
    monkeypatch.setattr("grl.diagnostics.oracle.select_degree_discount_nodes", lambda *args, **kwargs: [1])
    result = run_oracle_diagnostics(_data(), _config(1), scorer=lambda seeds, candidates: {1: 0.5})
    step = result["steps"][0]
    assert step["candidate_loss"] > 0
    assert step["ranking_loss"] == 0
    assert step["total_loss"] == step["candidate_loss"] + step["ranking_loss"]


def test_ranking_loss_is_nonzero_when_candidate_oracle_is_misranked(monkeypatch):
    monkeypatch.setattr("grl.diagnostics.oracle.select_high_degree_nodes", lambda *args, **kwargs: [0, 1])
    monkeypatch.setattr("grl.diagnostics.oracle.select_degree_discount_nodes", lambda *args, **kwargs: [])
    result = run_oracle_diagnostics(_data(), _config(2), scorer=lambda seeds, candidates: {0: 0.1, 1: 0.9})
    step = result["steps"][0]
    assert step["candidate_loss"] == 0
    assert step["ranking_loss"] > 0
    assert step["total_loss"] == step["candidate_loss"] + step["ranking_loss"]
