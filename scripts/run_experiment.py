import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines import IMBaselines


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_output_dir(base_dir: str, dataset_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(base_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def run_method(solver: IMBaselines, method_name: str, budget: int, mc_eval: int):
    if method_name == "degree":
        seeds = solver.high_degree(budget)
    elif method_name == "degree_discount":
        seeds = solver.degree_discount_ic(budget)
    else:
        raise ValueError(f"Unsupported baseline method: {method_name}")

    spread = solver.evaluate(seeds, mc=mc_eval)
    return {
        "method": method_name,
        "selected_seeds": list(seeds),
        "spread": float(spread),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]["name"]
    graph_path = config["dataset"]["graph_path"]
    budget = int(config["seed"]["budget"])
    mc_eval = int(config["diffusion"]["mc_runs_eval"])
    methods = config.get("baselines", {}).get("methods", ["degree", "degree_discount"])
    random_seed = int(config["experiment"]["random_seed"])
    output_base = config["experiment"]["output_dir"]

    set_random_seed(random_seed)
    out_dir = ensure_output_dir(output_base, dataset_name)

    with (out_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    solver = IMBaselines(graph_path)
    results = []
    for method in methods:
        results.append(run_method(solver, method, budget, mc_eval))

    metrics = {
        "dataset": dataset_name,
        "graph_path": graph_path,
        "budget": budget,
        "mc_runs_eval": mc_eval,
        "random_seed": random_seed,
        "results": [
            {"method": item["method"], "spread": item["spread"]}
            for item in results
        ],
    }

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with (out_dir / "selected_seeds.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with (out_dir / "run.log").open("w", encoding="utf-8") as f:
        f.write(f"config={config_path}\n")
        f.write(f"dataset={dataset_name}\n")
        f.write(f"budget={budget}\n")
        for item in results:
            f.write(f"{item['method']}: spread={item['spread']:.4f}, seeds={item['selected_seeds']}\n")

    print(f"Experiment finished. Outputs saved to: {out_dir}")
    for item in results:
        print(f"[{item['method']}] spread={item['spread']:.4f} seeds={item['selected_seeds']}")


if __name__ == "__main__":
    main()
