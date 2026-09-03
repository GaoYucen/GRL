from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

import networkx as nx
import torch


@dataclass
class OracleStats:
    candidate_evaluations: int = 0
    mc_candidate_samples: int = 0
    live_edge_samples: int = 0
    learned_evaluations: int = 0


class BatchedMonteCarloMarginalOracle:
    """Monte-Carlo marginal oracle that reuses each live-edge world across candidates."""

    def __init__(self, graph: nx.Graph | nx.DiGraph, mc_runs: int = 40, random_seed: int = 42):
        self.graph = graph
        self.mc_runs = int(mc_runs)
        self.random_seed = int(random_seed)
        self.stats = OracleStats()

    def _reached(self, live_graph, seeds: Iterable[int]) -> set[int]:
        reached: set[int] = set()
        for seed in seeds:
            if live_graph.is_directed():
                reached.update(nx.descendants(live_graph, seed))
            else:
                reached.update(nx.node_connected_component(live_graph, seed))
            reached.add(seed)
        return reached

    def score(self, seeds: list[int], candidates: list[int], step: int = 0) -> dict[int, float]:
        if not candidates:
            return {}
        totals = {int(v): 0.0 for v in candidates}
        for offset in range(self.mc_runs):
            rng = random.Random(self.random_seed + step * 1_000_003 + offset)
            live_graph = nx.DiGraph() if self.graph.is_directed() else nx.Graph()
            live_graph.add_nodes_from(self.graph.nodes())
            for u, v, data in self.graph.edges(data=True):
                if rng.random() < float(data.get("weight", 0.0)):
                    live_graph.add_edge(u, v)
            base = self._reached(live_graph, seeds)
            for candidate in candidates:
                if candidate in base:
                    gain = 0
                else:
                    if live_graph.is_directed():
                        cand_reached = set(nx.descendants(live_graph, candidate))
                    else:
                        cand_reached = set(nx.node_connected_component(live_graph, candidate))
                    cand_reached.add(candidate)
                    gain = len(cand_reached - base)
                totals[int(candidate)] += float(gain)
        n = len(candidates)
        self.stats.candidate_evaluations += n
        self.stats.mc_candidate_samples += n * self.mc_runs
        self.stats.live_edge_samples += self.mc_runs
        return {v: total / self.mc_runs for v, total in totals.items()}


class LearnedMarginalOracle:
    """Batched wrapper around a seed-conditioned marginal-gain predictor."""

    def __init__(self, model, embeddings, norm_degrees, device, batch_size: int = 64):
        self.model = model
        self.embeddings = embeddings
        self.norm_degrees = norm_degrees
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.num_nodes = int(embeddings.shape[0])
        self.stats = OracleStats()

    def score(self, seeds: list[int], candidates: list[int], step: int = 0) -> dict[int, float]:
        del step
        if not candidates:
            return {}
        mask = torch.zeros((self.num_nodes, 1), dtype=torch.float32, device=self.device)
        if seeds:
            mask[seeds, 0] = 1.0
        out: dict[int, float] = {}
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(candidates), self.batch_size):
                chunk = candidates[start:start + self.batch_size]
                b = len(chunk)
                candidate_tensor = torch.as_tensor(chunk, dtype=torch.long, device=self.device)
                pred = self.model(
                    self.embeddings.unsqueeze(0).expand(b, -1, -1),
                    self.norm_degrees.unsqueeze(0).expand(b, -1, -1),
                    mask.unsqueeze(0).expand(b, -1, -1),
                    candidate_tensor,
                ).reshape(-1)
                out.update({int(v): float(p) for v, p in zip(chunk, pred.detach().cpu().tolist())})
        self.stats.learned_evaluations += len(candidates)
        return out
