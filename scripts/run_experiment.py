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

from grl.baselines import select_degree_discount_nodes, select_high_degree_nodes
from grl.data import load_graph_from_config
from grl.evaluation import evaluate_baseline_method
from grl.utils import load_yaml_config, set_random_seed


def ensure_output_dir(base_dir: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(base_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def run_method(graph_data, method_name: str, budget: int, mc_eval: int, probability: float, random_seed: int):
    if method_name == "degree":
        selector = lambda: select_high_degree_nodes(graph_data.graph, budget)
    elif method_name == "degree_discount":
        selector = lambda: select_degree_discount_nodes(graph_data.graph, budget, probability)
    else:
        raise ValueError(f"Unsupported baseline method: {method_name}")
    return evaluate_baseline_method(
        graph=graph_data.graph,
        method_name=method_name,
        selector=selector,
        mc_runs=mc_eval,
        random_seed=random_seed,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_yaml_config(config_path)

    dataset_name = config["dataset"]["name"]
    graph_path = config["dataset"]["graph_path"]
    budget = int(config["seed"]["budget"])
    mc_eval = int(config["diffusion"]["mc_runs_eval"])
    probability = float(config["diffusion"]["probability"])
    methods = config.get("baselines", {}).get("methods", ["degree", "degree_discount"])
    random_seed = int(config["experiment"]["random_seed"])
    output_base = config["experiment"]["output_dir"]

    set_random_seed(random_seed)
    out_dir = ensure_output_dir(output_base)

    with (out_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    graph_data = load_graph_from_config(config)
    results = []
    for method in methods:
        results.append(run_method(graph_data, method, budget, mc_eval, probability, random_seed))

    metrics = {
        "dataset": dataset_name,
        "graph_path": graph_path,
        "directed": graph_data.directed,
        "num_nodes": graph_data.num_nodes,
        "num_edges": graph_data.num_edges,
        "budget": budget,
        "diffusion_model": "independent_cascade",
        "probability": probability,
        "mc_runs_eval": mc_eval,
        "random_seed": random_seed,
        "results": results,
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
            f.write(
                f"{item['method']}: spread_mean={item['spread_mean']:.4f}, "
                f"spread_std={item['spread_std']:.4f}, seeds={item['selected_seeds']}\n"
            )

    print(f"Experiment finished. Outputs saved to: {out_dir}")
    for item in results:
        print(
            f"[{item['method']}] spread_mean={item['spread_mean']:.4f} "
            f"spread_std={item['spread_std']:.4f} seeds={item['selected_seeds']}"
        )


if __name__ == "__main__":
    main()
