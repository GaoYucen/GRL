from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

from grl.baselines import select_high_degree_nodes
from grl.diffusion import estimate_marginal_gain


@dataclass(frozen=True)
class ConditionalMarginalGroup:
    """Multiple candidate labels for one fixed graph and seed state."""

    seed_set: tuple[int, ...]
    candidates: tuple[int, ...]
    marginal_gains: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("a conditional group needs at least one candidate")
        if len(self.candidates) != len(self.marginal_gains):
            raise ValueError("candidates and marginal_gains must have equal length")


def _sample_seed_states(graph_data, budget: int, count: int, rng: random.Random) -> list[tuple[int, ...]]:
    nodes = list(range(graph_data.num_nodes))
    degree_order = select_high_degree_nodes(graph_data.graph, graph_data.num_nodes)
    states: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    for index in range(max(1, count)):
        size = index % max(1, budget)
        if index % 3 == 0:
            seeds = rng.sample(nodes, size) if size else []
        elif index % 3 == 1:
            seeds = degree_order[:size]
        else:
            shuffled = nodes[:]
            rng.shuffle(shuffled)
            seeds = shuffled[:size]
        state = tuple(sorted(seeds))
        if state not in seen and len(state) < graph_data.num_nodes:
            states.append(state)
            seen.add(state)
    return states


def build_conditional_marginal_dataset(
    graph_data,
    config: dict,
    split: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, list[ConditionalMarginalGroup]]:
    cfg = config.get("gnn", {}) | config.get("marginal_gain", {})
    state_count = int(cfg.get("ranking_states", cfg.get("samples", 128)))
    candidates_per_state = int(cfg.get("candidates_per_state", 16))
    budget = min(int(config.get("seed", {}).get("budget", 10)), graph_data.num_nodes - 1)
    mc_runs = int(cfg.get("mc_runs_train", config.get("diffusion", {}).get("mc_runs_train", 30)))
    base_seed = int(config.get("experiment", {}).get("random_seed", 42))
    rng = random.Random(base_seed)
    nodes = list(range(graph_data.num_nodes))
    groups: list[ConditionalMarginalGroup] = []

    for state_index, seed_set in enumerate(_sample_seed_states(graph_data, budget, state_count, rng)):
        available = [node for node in nodes if node not in seed_set]
        if not available:
            continue
        count = min(max(1, candidates_per_state), len(available))
        candidates = rng.sample(available, count)
        gains = []
        for candidate_index, candidate in enumerate(candidates):
            label_seed = base_seed + state_index * 100000 + candidate_index
            gain = estimate_marginal_gain(
                graph_data.graph,
                list(seed_set),
                candidate,
                mc_runs,
                label_seed,
            )["mean"]
            gains.append(float(max(gain, 0.0)))
        groups.append(ConditionalMarginalGroup(seed_set, tuple(candidates), tuple(gains)))

    rng.shuffle(groups)
    n = len(groups)
    train_end = int(n * split[0])
    valid_end = train_end + int(n * split[1])
    return {
        "train": groups[:train_end],
        "validation": groups[train_end:valid_end],
        "test": groups[valid_end:],
    }
