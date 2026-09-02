import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from grl.data import load_graph_from_config
from grl.evaluation.conditional_ranking import CandidateRankingGroup, evaluate_conditional_rankings
from grl.models import MarginalGainPredictor, build_node_features, load_or_create_node2vec_embeddings
from grl.training.conditional_dataset import build_conditional_marginal_dataset
from grl.utils import load_yaml_config, set_random_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--embedding-path", default=None)
    parser.add_argument("--ranking-states", type=int, default=None)
    parser.add_argument("--candidates-per-state", type=int, default=None)
    parser.add_argument("--mc-runs", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    gnn_config = config.setdefault("gnn", {})
    if args.ranking_states is not None:
        gnn_config["ranking_states"] = args.ranking_states
    if args.candidates_per_state is not None:
        gnn_config["candidates_per_state"] = args.candidates_per_state
    if args.mc_runs is not None:
        gnn_config["mc_runs_train"] = args.mc_runs

    set_random_seed(int(config["experiment"]["random_seed"]))
    graph_data = load_graph_from_config(config)
    device = torch.device(config.get("gnn", {}).get("device", "cpu"))
    model_dir = Path(config.get("gnn", {}).get("model_dir", "param"))
    embedding_path = Path(args.embedding_path) if args.embedding_path else model_dir / f"marginal_node2vec_{graph_data.name}.pth"
    model_path = Path(args.model_path) if args.model_path else model_dir / f"marginal_gain_{graph_data.name}.pth"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Marginal predictor checkpoint not found: {model_path}. "
            "Train it first or pass --model-path."
        )

    embeddings = load_or_create_node2vec_embeddings(graph_data.graph, embedding_path).to(device)
    norm_degrees, _ = build_node_features(graph_data.graph, device=device)
    checkpoint = torch.load(model_path, map_location=device)
    model = MarginalGainPredictor(
        embeddings.shape[1],
        int(config["gnn"].get("hidden_dim", 64)),
    ).to(device)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    model.eval()

    ranking_groups = []
    groups = build_conditional_marginal_dataset(graph_data, config)[args.split]
    with torch.no_grad():
        for group in groups:
            seed_mask = torch.zeros((graph_data.num_nodes, 1), dtype=torch.float32, device=device)
            if group.seed_set:
                seed_mask[list(group.seed_set)] = 1.0
            predictions = []
            for candidate in group.candidates:
                predictions.append(float(model(embeddings, norm_degrees, seed_mask, candidate).item()))
            ranking_groups.append(
                CandidateRankingGroup(
                    group.seed_set,
                    group.candidates,
                    tuple(predictions),
                    group.marginal_gains,
                )
            )

    candidates_per_group = int(gnn_config.get("candidates_per_state", 16))
    metrics = {
        "dataset": graph_data.name,
        "split": args.split,
        "groups": len(ranking_groups),
        "candidates_per_group": candidates_per_group,
        "model_path": str(model_path),
        "metrics": evaluate_conditional_rankings(ranking_groups),
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
