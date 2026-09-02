from __future__ import annotations

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
from grl.evaluation.conditional_ranking import (
    CandidateRankingGroup,
    evaluate_conditional_rankings,
)
from grl.evaluation.gnn_metrics import spearman_correlation
from grl.evaluation.uncertainty import summarize_ensemble
from grl.models import (
    MarginalGainPredictor,
    build_node_features,
    load_or_create_node2vec_embeddings,
)
from grl.training.conditional_dataset import build_conditional_marginal_dataset
from grl.utils import load_yaml_config, set_random_seed


def _load_model(
    model_path: Path,
    embeddings: torch.Tensor,
    norm_degrees: torch.Tensor,
    hidden_dim: int,
    device: torch.device,
) -> MarginalGainPredictor:
    checkpoint = torch.load(model_path, map_location=device)
    model = MarginalGainPredictor(embeddings.shape[1], hidden_dim).to(device)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a small ensemble and uncertainty-error calibration."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-paths", nargs="+", required=True)
    parser.add_argument("--embedding-path", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--ranking-states", type=int, default=None)
    parser.add_argument("--candidates-per-state", type=int, default=None)
    parser.add_argument("--confidence-scale", type=float, default=1.96)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    gnn_config = config.setdefault("gnn", {})
    if args.ranking_states is not None:
        gnn_config["ranking_states"] = args.ranking_states
    if args.candidates_per_state is not None:
        gnn_config["candidates_per_state"] = args.candidates_per_state
    set_random_seed(int(config["experiment"]["random_seed"]))

    graph_data = load_graph_from_config(config)
    device = torch.device(gnn_config.get("device", "cpu"))
    embeddings = load_or_create_node2vec_embeddings(
        graph_data.graph, Path(args.embedding_path)
    ).to(device)
    norm_degrees, _ = build_node_features(graph_data.graph, device=device)
    models = [
        _load_model(
            Path(model_path),
            embeddings,
            norm_degrees,
            int(gnn_config.get("hidden_dim", 64)),
            device,
        )
        for model_path in args.model_paths
    ]

    groups = build_conditional_marginal_dataset(graph_data, config)[args.split]
    mean_groups: list[CandidateRankingGroup] = []
    conservative_groups: list[CandidateRankingGroup] = []
    uncertainties: list[float] = []
    errors: list[float] = []
    covered = 0
    total = 0

    with torch.no_grad():
        for group in groups:
            seed_mask = torch.zeros(
                (graph_data.num_nodes, 1), dtype=torch.float32, device=device
            )
            if group.seed_set:
                seed_mask[list(group.seed_set)] = 1.0
            member_predictions = []
            for model in models:
                member_predictions.append(
                    [
                        float(
                            model(
                                embeddings,
                                norm_degrees,
                                seed_mask,
                                candidate,
                            ).item()
                        )
                        for candidate in group.candidates
                    ]
                )
            prediction_tensor = torch.tensor(member_predictions, device=device)
            summary = summarize_ensemble(prediction_tensor, args.confidence_scale)
            mean_predictions = tuple(float(value) for value in summary.mean.cpu())
            conservative_predictions = tuple(
                float(value) for value in summary.lower_bound.cpu()
            )
            mean_groups.append(
                CandidateRankingGroup(
                    group.seed_set,
                    group.candidates,
                    mean_predictions,
                    group.marginal_gains,
                )
            )
            conservative_groups.append(
                CandidateRankingGroup(
                    group.seed_set,
                    group.candidates,
                    conservative_predictions,
                    group.marginal_gains,
                )
            )
            target_tensor = torch.tensor(group.marginal_gains, device=device)
            absolute_errors = (summary.mean - target_tensor).abs()
            uncertainties.extend(float(value) for value in summary.std.cpu())
            errors.extend(float(value) for value in absolute_errors.cpu())
            covered += int(
                (absolute_errors <= args.confidence_scale * summary.std).sum().item()
            )
            total += len(group.candidates)

    if not groups:
        raise ValueError(f"split {args.split!r} produced no ranking groups")

    metrics = {
        "dataset": graph_data.name,
        "split": args.split,
        "ensemble_members": len(models),
        "groups": len(groups),
        "candidates_per_group": len(groups[0].candidates),
        "confidence_scale": args.confidence_scale,
        "mean_prediction_metrics": evaluate_conditional_rankings(mean_groups),
        "conservative_prediction_metrics": evaluate_conditional_rankings(
            conservative_groups
        ),
        "mean_absolute_error": sum(errors) / len(errors),
        "mean_uncertainty_std": sum(uncertainties) / len(uncertainties),
        "uncertainty_error_spearman": spearman_correlation(uncertainties, errors),
        "empirical_interval_coverage": covered / total,
        "evaluated_candidates": total,
        "model_paths": [str(path) for path in args.model_paths],
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
