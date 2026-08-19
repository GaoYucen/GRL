import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from grl.data import load_graph_from_config
from grl.training import build_marginal_dataset
from grl.utils import load_yaml_config, set_random_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    set_random_seed(int(config["experiment"]["random_seed"]))
    graph_data = load_graph_from_config(config)
    splits = build_marginal_dataset(graph_data, config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({key: [sample.to_dict() for sample in value] for key, value in splits.items()}, indent=2), encoding="utf-8")
    print(f"Marginal-gain dataset written to: {output}")


if __name__ == "__main__":
    main()
