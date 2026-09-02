import random

import networkx as nx
import pytest

from grl.training.conditional_dataset import ConditionalMarginalGroup, _sample_seed_states


class _GraphData:
    num_nodes = 8
    graph = nx.path_graph(num_nodes)


def test_seed_states_are_unique_and_within_budget():
    states = _sample_seed_states(_GraphData(), budget=4, count=20, rng=random.Random(7))
    assert states
    assert len(states) == len(set(states))
    assert all(len(state) < 8 for state in states)
    assert all(len(state) <= 3 for state in states)


def test_conditional_group_requires_aligned_candidates_and_labels():
    with pytest.raises(ValueError):
        ConditionalMarginalGroup((0,), (1, 2), (1.0,))
