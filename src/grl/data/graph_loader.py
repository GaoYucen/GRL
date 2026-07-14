from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx


class GraphValidationError(ValueError):
    pass


@dataclass
class GraphData:
    name: str
    graph_path: Path
    directed: bool
    graph: nx.Graph | nx.DiGraph
    num_nodes: int
    num_edges: int
    self_loops: int
    duplicate_edges: int
    average_degree: float
    max_degree: int
    connected_components: int
    largest_component_size: int


def _build_graph(directed: bool) -> nx.Graph | nx.DiGraph:
    return nx.DiGraph() if directed else nx.Graph()


def _parse_graph_file(graph_path: Path, directed: bool, default_probability: float) -> tuple[nx.Graph | nx.DiGraph, int]:
    if not graph_path.exists():
        raise GraphValidationError(f"Graph file does not exist: {graph_path}")

    graph = _build_graph(directed)
    seen_edges: set[tuple[int, int]] = set()
    duplicate_edges = 0

    with graph_path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip().split()
        has_header = len(first_line) == 2 and all(part.lstrip("-").isdigit() for part in first_line)
        if not has_header:
            handle.seek(0)

        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue

            u = int(parts[0])
            v = int(parts[1])
            probability = float(parts[2]) if len(parts) >= 3 else float(default_probability)

            edge_key = (u, v) if directed else tuple(sorted((u, v)))
            if edge_key in seen_edges:
                duplicate_edges += 1
            seen_edges.add(edge_key)
            graph.add_edge(u, v, weight=probability)

    return graph, duplicate_edges


def _component_stats(graph: nx.Graph | nx.DiGraph, directed: bool) -> tuple[int, int]:
    if graph.number_of_nodes() == 0:
        return 0, 0

    if directed:
        components = list(nx.weakly_connected_components(graph))
    else:
        components = list(nx.connected_components(graph))
    largest = max((len(component) for component in components), default=0)
    return len(components), largest


def _validate_expected_counts(graph: nx.Graph | nx.DiGraph, expected_num_nodes: Any, expected_num_edges: Any) -> None:
    if expected_num_nodes is not None and graph.number_of_nodes() != int(expected_num_nodes):
        raise GraphValidationError(
            f"Expected {expected_num_nodes} nodes, got {graph.number_of_nodes()}"
        )
    if expected_num_edges is not None and graph.number_of_edges() != int(expected_num_edges):
        raise GraphValidationError(
            f"Expected {expected_num_edges} edges, got {graph.number_of_edges()}"
        )


def load_graph_from_config(config: dict[str, Any]) -> GraphData:
    dataset = config["dataset"]
    diffusion = config.get("diffusion", {})
    name = str(dataset["name"])
    graph_path = Path(str(dataset["graph_path"]))
    directed = bool(dataset.get("directed", True))
    default_probability = float(diffusion.get("probability", 0.01))

    graph, duplicate_edges = _parse_graph_file(graph_path, directed, default_probability)
    _validate_expected_counts(
        graph,
        dataset.get("expected_num_nodes"),
        dataset.get("expected_num_edges"),
    )

    degrees = [degree for _, degree in graph.degree()]
    average_degree = float(sum(degrees) / len(degrees)) if degrees else 0.0
    max_degree = max(degrees, default=0)
    components, largest_component = _component_stats(graph, directed)

    return GraphData(
        name=name,
        graph_path=graph_path,
        directed=directed,
        graph=graph,
        num_nodes=graph.number_of_nodes(),
        num_edges=graph.number_of_edges(),
        self_loops=nx.number_of_selfloops(graph),
        duplicate_edges=duplicate_edges,
        average_degree=average_degree,
        max_degree=max_degree,
        connected_components=components,
        largest_component_size=largest_component,
    )
