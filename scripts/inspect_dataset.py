import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from grl.data import load_graph_from_config
from grl.utils import load_yaml_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    graph_data = load_graph_from_config(config)

    print(f"Dataset name: {graph_data.name}")
    print(f"Graph path: {graph_data.graph_path}")
    print(f"Directed: {graph_data.directed}")
    print(f"Number of nodes: {graph_data.num_nodes}")
    print(f"Number of edges: {graph_data.num_edges}")
    print(f"Connected components: {graph_data.connected_components}")
    print(f"Largest component size: {graph_data.largest_component_size}")
    print(f"Average degree: {graph_data.average_degree:.4f}")
    print(f"Maximum degree: {graph_data.max_degree}")
    print(f"Self loops: {graph_data.self_loops}")
    print(f"Duplicate edges: {graph_data.duplicate_edges}")


if __name__ == "__main__":
    main()
