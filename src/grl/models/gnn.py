from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

try:
    from node2vec import Node2Vec
except ImportError:
    Node2Vec = None


def _build_fallback_embeddings(graph: nx.Graph | nx.DiGraph) -> torch.Tensor:
    num_nodes = graph.number_of_nodes()
    out_degrees = dict(graph.out_degree() if graph.is_directed() else graph.degree())
    in_degrees = dict(graph.in_degree()) if graph.is_directed() else out_degrees
    clustering = nx.clustering(graph.to_undirected()) if num_nodes > 0 else {}
    pagerank = nx.pagerank(graph, alpha=0.85) if num_nodes > 0 else {}
    embeddings = np.zeros((num_nodes, 64), dtype=np.float32)
    max_out = max(out_degrees.values(), default=1) or 1
    max_in = max(in_degrees.values(), default=1) or 1
    max_pr = max(pagerank.values(), default=1.0) or 1.0
    for node in graph.nodes():
        node = int(node)
        base = np.array([
            out_degrees.get(node, 0) / max_out,
            in_degrees.get(node, 0) / max_in,
            clustering.get(node, 0.0),
            pagerank.get(node, 0.0) / max_pr,
        ], dtype=np.float32)
        tiled = np.tile(base, 16)
        embeddings[node] = tiled[:64]
    return torch.tensor(embeddings, dtype=torch.float32)


def load_or_create_node2vec_embeddings(
    graph: nx.Graph | nx.DiGraph,
    cache_path: str | Path,
    dimensions: int = 64,
    walk_length: int = 10,
    num_walks: int = 10,
    window: int = 10,
    workers: int = 1,
    quiet: bool = True,
) -> torch.Tensor:
    cache_path = Path(cache_path)
    if cache_path.exists():
        embeddings = torch.load(cache_path, map_location="cpu")
        return F.normalize(embeddings.float(), p=2, dim=1)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if Node2Vec is None:
        embeddings = _build_fallback_embeddings(graph)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        torch.save(embeddings, cache_path)
        return embeddings

    node2vec = Node2Vec(
        graph,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=workers,
        quiet=quiet,
    )
    model = node2vec.fit(window=window, min_count=1, batch_words=4)
    emb_matrix = np.zeros((graph.number_of_nodes(), dimensions), dtype=np.float32)
    for node in graph.nodes():
        key = str(node)
        if key in model.wv:
            emb_matrix[int(node)] = model.wv[key]
    embeddings = torch.tensor(emb_matrix, dtype=torch.float32)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    torch.save(embeddings, cache_path)
    return embeddings


def build_node_features(graph: nx.Graph | nx.DiGraph, device: str | torch.device = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    degree_view = graph.out_degree() if graph.is_directed() else graph.degree()
    degrees = dict(degree_view)
    max_degree = max(degrees.values(), default=1) or 1
    norm_degrees = torch.zeros((graph.number_of_nodes(), 1), dtype=torch.float32, device=device)
    for node in graph.nodes():
        norm_degrees[int(node), 0] = degrees.get(node, 0) / max_degree
    return norm_degrees, torch.tensor([max_degree], dtype=torch.float32, device=device)


class SpreadPredictorGNN(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim + 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, embeddings: torch.Tensor, norm_degrees: torch.Tensor, seed_mask: torch.Tensor) -> torch.Tensor:
        if embeddings.dim() == 3:
            x = torch.cat([embeddings, norm_degrees, seed_mask], dim=2)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            graph_rep = torch.sum(x, dim=1)
        else:
            x = torch.cat([embeddings, norm_degrees, seed_mask], dim=1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            graph_rep = torch.sum(x, dim=0)
        return self.fc_out(graph_rep)
