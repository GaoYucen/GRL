from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from grl.baselines import select_degree_discount_nodes, select_high_degree_nodes
from grl.data import load_graph_from_config
from grl.diagnostics import run_oracle_diagnostics
from grl.diffusion import estimate_spread
from grl.evaluation.gnn_metrics import evaluate_marginal_gain_predictor
from grl.evaluation.sequential import evaluate_sequential_selector
from grl.evaluation.spread import evaluate_baseline_method
from grl.models import build_node_features, load_or_create_node2vec_embeddings, MarginalGainPredictor
from grl.training import MarginalGainTrainer, build_marginal_dataset
from grl.utils import build_run_metadata, load_yaml_config, set_random_seed


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def _build_scorer(graph_data, config, model_path):
    import torch

    device = torch.device(config["gnn"].get("device", "cpu"))
    model_dir = Path(config["gnn"]["model_dir"])
    embeddings = load_or_create_node2vec_embeddings(
        graph_data.graph,
        model_dir / f"marginal_node2vec_{graph_data.name}.pth",
        dimensions=int(config["gnn"]["embedding_dim"]),
        walk_length=int(config["gnn"]["walk_length"]),
        num_walks=int(config["gnn"]["num_walks"]),
        window=int(config["gnn"]["window"]),
        workers=int(config["gnn"]["workers"]),
    ).to(device)
    degrees, _ = build_node_features(graph_data.graph, device=device)
    checkpoint = torch.load(model_path, map_location=device)
    model = MarginalGainPredictor(embeddings.shape[1], int(config["gnn"]["hidden_dim"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    def scorer(seeds, candidates):
        mask = torch.zeros((graph_data.num_nodes, 1), dtype=torch.float32, device=device)
        if seeds:
            mask[seeds] = 1.0
        with torch.no_grad():
            return {node: float(model(embeddings, degrees, mask, node).item()) for node in candidates}

    return scorer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke/network_science_first_round.yaml")
    args = parser.parse_args()
    started = time.perf_counter()
    config = load_yaml_config(args.config)
    set_random_seed(int(config["experiment"]["random_seed"]))
    graph_data = load_graph_from_config(config)
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")

    results = {
        "metadata": build_run_metadata(args.config, config, PROJECT_ROOT, "marginal_gain_predictor_v1"),
        "graph": {
            "dataset": graph_data.name,
            "num_nodes": graph_data.num_nodes,
            "num_edges": graph_data.num_edges,
            "node_indexing": "contiguous_internal_indices",
        },
    }

    baseline_results = []
    budget = int(config["seed"]["budget"])
    probability = float(config["diffusion"]["probability"])
    mc_eval = int(config["diffusion"]["mc_runs_eval"])
    for method in config["baselines"]["methods"]:
        if method == "degree":
            selector = lambda: select_high_degree_nodes(graph_data.graph, budget)
        else:
            selector = lambda: select_degree_discount_nodes(graph_data.graph, budget, probability)
        baseline_results.append(evaluate_baseline_method(graph_data.graph, method, selector, mc_eval, config["experiment"]["random_seed"]))
    results["baselines"] = baseline_results

    splits = build_marginal_dataset(graph_data, config)
    all_samples = [sample for values in splits.values() for sample in values]
    results["dataset"] = {
        "split_sizes": {key: len(value) for key, value in splits.items()},
        "seed_set_sizes": sorted({len(sample.seed_set) for sample in all_samples}),
        "min_marginal_gain": min((sample.marginal_gain for sample in all_samples), default=0.0),
        "max_marginal_gain": max((sample.marginal_gain for sample in all_samples), default=0.0),
        "all_labels_finite": all(math.isfinite(sample.marginal_gain) for sample in all_samples),
        "all_labels_nonnegative": all(sample.marginal_gain >= 0.0 for sample in all_samples),
    }

    trainer = MarginalGainTrainer(graph_data, config)
    train_metrics, artifacts = trainer.train(output_dir / "training")
    results["training"] = train_metrics
    results["artifacts"] = {"model_path": str(artifacts.model_path), "embedding_path": str(artifacts.embedding_path), "dataset_path": str(artifacts.dataset_path)}
    results["predictor"] = evaluate_marginal_gain_predictor(graph_data, config)

    scorer = _build_scorer(graph_data, config, artifacts.model_path)
    results["sequential"] = evaluate_sequential_selector(graph_data, budget, scorer, mc_eval, config["experiment"]["random_seed"])
    results["oracle"] = run_oracle_diagnostics(graph_data, config, scorer=scorer)
    steps = results["oracle"]["steps"]
    results["validation"] = {
        "all_metrics_finite": all(math.isfinite(float(value)) for value in [
            results["predictor"].get("mae", 0.0),
            results["predictor"].get("rmse", 0.0),
            results["predictor"].get("spearman", 0.0),
            results["predictor"].get("kendall", 0.0),
        ]),
        "oracle_loss_decomposition_holds": all(abs(step["total_loss"] - step["candidate_loss"] - step["ranking_loss"]) < 1e-8 for step in steps),
        "oracle_steps": len(steps),
        "different_seed_sets_change_scores": len({tuple(step["seed_set"]) for step in steps}) == len(steps),
    }
    results["elapsed_seconds"] = time.perf_counter() - started
    result_path = output_dir / "first_round_results.json"
    result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"result_path": str(result_path), "elapsed_seconds": results["elapsed_seconds"], "validation": results["validation"], "predictor": results["predictor"], "baselines": results["baselines"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
