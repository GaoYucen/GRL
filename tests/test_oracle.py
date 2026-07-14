import networkx as nx

from grl.diffusion import estimate_spread
from grl.data.graph_loader import GraphData


def test_marginal_gain_is_non_negative_in_simple_chain():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, weight=1.0)
    graph.add_edge(1, 2, weight=1.0)
    base = estimate_spread(graph, [0], mc_runs=3, random_seed=1)["mean"]
    extended = estimate_spread(graph, [0, 2], mc_runs=3, random_seed=1)["mean"]
    assert extended >= base


def test_oracle_diagnostics_exposes_loss_decomposition(monkeypatch):
    import torch

    from grl.diagnostics.oracle import run_oracle_diagnostics

    graph = nx.DiGraph()
    graph.add_edge(0, 1, weight=1.0)
    graph.add_edge(0, 2, weight=1.0)
    graph.add_edge(3, 4, weight=1.0)

    graph_data = GraphData(
        name="toy",
        graph_path="toy.txt",  # type: ignore[arg-type]
        directed=True,
        graph=graph,
        num_nodes=graph.number_of_nodes(),
        num_edges=graph.number_of_edges(),
        self_loops=0,
        duplicate_edges=0,
        average_degree=1.2,
        max_degree=2,
        connected_components=2,
        largest_component_size=3,
    )

    class DummyModel:
        def to(self, device):
            return self

        def load_state_dict(self, state_dict):
            return self

        def eval(self):
            return self

        def __call__(self, embeddings, norm_degrees, cand_mask):
            selected = torch.where(cand_mask.squeeze(-1) > 0)[0].tolist()
            candidate = selected[-1]
            scores = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.9, 4: 0.0}
            return torch.tensor(scores.get(candidate, 0.0), dtype=torch.float32)

    monkeypatch.setattr("grl.diagnostics.oracle.load_or_create_node2vec_embeddings", lambda *args, **kwargs: torch.zeros((5, 4)))
    monkeypatch.setattr("grl.diagnostics.oracle.build_node_features", lambda *args, **kwargs: (torch.zeros((5, 1)), None))
    monkeypatch.setattr("grl.diagnostics.oracle.SpreadPredictorGNN", lambda *args, **kwargs: DummyModel())
    monkeypatch.setattr("grl.diagnostics.oracle.torch.load", lambda *args, **kwargs: {})
    monkeypatch.setattr("grl.diagnostics.oracle.select_high_degree_nodes", lambda *args, **kwargs: [1])
    monkeypatch.setattr("grl.diagnostics.oracle.select_degree_discount_nodes", lambda *args, **kwargs: [2])
    monkeypatch.setattr(
        "grl.diagnostics.oracle.evaluate_trained_gnn",
        lambda *args, **kwargs: {"spearman": 0.0, "kendall": 0.0, "topk_recall": 0.0},
    )

    config = {
        "seed": {"budget": 1},
        "oracle": {"mc_runs": 2, "candidate_pool_size": 1, "max_nodes": 5, "step_limit": 1},
        "experiment": {"random_seed": 7},
        "diffusion": {"probability": 1.0, "mc_runs_eval": 2},
        "gnn": {"device": "cpu", "model_dir": "param", "hidden_dim": 4},
    }

    results = run_oracle_diagnostics(graph_data, config)
    step = results["steps"][0]

    assert step["global_oracle_best_node"] == 0
    assert step["candidate_best_node"] == 3
    assert step["selected_node"] == 3
    assert step["candidate_recall_at_k"] == 0.0
    assert step["candidate_loss"] > 0.0
    assert step["ranking_loss"] == 0.0
    assert step["total_selection_loss"] == step["candidate_loss"] + step["ranking_loss"]
    assert 0.0 <= step["relative_gain"] <= 1.0
    assert step["gnn_selected_node"] == 3
    assert step["degree_selected_node"] == 1
    assert step["degree_discount_selected_node"] == 2
    assert step["degree_candidate_recall_at_k"] == 0.0
    assert step["degree_discount_candidate_recall_at_k"] == 0.0
    assert "candidate_gains" in step and len(step["candidate_gains"]) >= 1
