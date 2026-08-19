from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

from grl.baselines import select_degree_discount_nodes, select_high_degree_nodes
from grl.diffusion import estimate_marginal_gain


@dataclass(frozen=True)
class MarginalGainSample:
    seed_set: list[int]
    candidate: int
    marginal_gain: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _partial_seed_sequences(graph_data, budget: int, probability: float) -> list[list[int]]:
    degree = select_high_degree_nodes(graph_data.graph, min(budget, graph_data.num_nodes))
    discount = select_degree_discount_nodes(graph_data.graph, min(budget, graph_data.num_nodes), probability)
    return [degree[:size] for size in range(len(degree) + 1)] + [discount[:size] for size in range(len(discount) + 1)]


def build_marginal_dataset(
    graph_data,
    config: dict,
    split: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, list[MarginalGainSample]]:
    cfg = config.get("gnn", {}) | config.get("marginal_gain", {})
    total_samples = int(cfg.get("samples", 128))
    budget = min(int(config.get("seed", {}).get("budget", 10)), graph_data.num_nodes)
    mc_runs = int(cfg.get("mc_runs_train", config.get("diffusion", {}).get("mc_runs_train", 30)))
    base_seed = int(config.get("experiment", {}).get("random_seed", 42))
    probability = float(config.get("diffusion", {}).get("probability", 0.01))
    nodes = list(range(graph_data.num_nodes))
    rng = random.Random(base_seed)
    partials = _partial_seed_sequences(graph_data, budget, probability)
    samples: list[MarginalGainSample] = []
    for index in range(total_samples):
        size = index % max(budget, 1)
        if index % 4 == 0:
            seeds = rng.sample(nodes, size) if size else []
        elif index % 4 == 1 and partials:
            seeds = list(partials[index % len(partials)][:size])
        elif index % 4 == 2:
            pool = select_high_degree_nodes(graph_data.graph, graph_data.num_nodes)
            seeds = list(pool[:size])
        else:
            seeds = list(partials[index % len(partials)][:size]) if partials else []
        available = [node for node in nodes if node not in seeds]
        if not available:
            continue
        candidate = available[rng.randrange(len(available))]
        label_seed = base_seed + index
        gain = estimate_marginal_gain(graph_data.graph, seeds, candidate, mc_runs, label_seed)["mean"]
        samples.append(MarginalGainSample(seeds, candidate, float(max(gain, 0.0))))

    rng.shuffle(samples)
    n = len(samples)
    train_end = int(n * split[0])
    valid_end = train_end + int(n * split[1])
    return {"train": samples[:train_end], "validation": samples[train_end:valid_end], "test": samples[valid_end:]}
