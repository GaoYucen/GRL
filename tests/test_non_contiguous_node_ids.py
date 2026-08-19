import networkx as nx

from grl.data import load_graph_from_config
from grl.diffusion import estimate_spread
from grl.models import build_node_features


def test_non_contiguous_ids_are_mapped_to_internal_indices(tmp_path):
    graph_file = tmp_path / "graph.txt"
    graph_file.write_text("10 20 1.0\n20 35 1.0\n35 100 1.0\n", encoding="utf-8")
    data = load_graph_from_config({
        "dataset": {"name": "toy", "graph_path": str(graph_file), "directed": True},
        "diffusion": {"probability": 0.01},
    })
    assert data.idx_to_node == [10, 20, 35, 100]
    assert data.node_to_idx[35] == 2
    assert set(data.graph.nodes()) == {0, 1, 2, 3}
    features, _ = build_node_features(data.graph)
    assert features.shape == (4, 1)
    assert estimate_spread(data.graph, [data.to_index(10)], 2, 1)["mean"] == 4.0
    assert data.restore_nodes([0, 3]) == [10, 100]
