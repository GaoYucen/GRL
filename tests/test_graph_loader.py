import pytest

from grl.data.graph_loader import GraphValidationError, load_graph_from_config


def test_graph_loader_reads_nethept_config():
    config = {
        "dataset": {
            "name": "nethept",
            "graph_path": "data/NetHEPT.txt",
            "directed": True,
            "expected_num_nodes": None,
            "expected_num_edges": None,
        },
        "diffusion": {"probability": 0.01},
    }
    graph_data = load_graph_from_config(config)
    assert graph_data.name == "nethept"
    assert graph_data.num_nodes > 0
    assert graph_data.num_edges > 0


def test_graph_loader_raises_for_wrong_expected_count():
    config = {
        "dataset": {
            "name": "nethept",
            "graph_path": "data/NetHEPT.txt",
            "directed": True,
            "expected_num_nodes": 1,
            "expected_num_edges": None,
        },
        "diffusion": {"probability": 0.01},
    }
    with pytest.raises(GraphValidationError):
        load_graph_from_config(config)
