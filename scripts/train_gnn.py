import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from grl.data import load_graph_from_config
from grl.training import GNNTrainer
from grl.utils import load_yaml_config, set_random_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    set_random_seed(int(config["experiment"]["random_seed"]))
    graph_data = load_graph_from_config(config)
    output_dir = Path(config["experiment"]["output_dir"]) / datetime.now().strftime("%Y%m%d_%H%M%S") / "gnn_train"
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = GNNTrainer(graph_data, config)
    metrics, artifacts = trainer.train(output_dir)

    with (output_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with (output_dir / "artifacts.json").open("w", encoding="utf-8") as f:
        json.dump({"model_path": str(artifacts.model_path), "embedding_path": str(artifacts.embedding_path)}, f, ensure_ascii=False, indent=2)

    print(f"GNN training finished. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
