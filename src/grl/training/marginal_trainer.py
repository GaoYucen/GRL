from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

from grl.models import MarginalGainPredictor, build_node_features, load_or_create_node2vec_embeddings
from .marginal_dataset import MarginalGainSample, build_marginal_dataset


@dataclass
class MarginalGainArtifacts:
    model_path: Path
    embedding_path: Path
    dataset_path: Path


class MarginalGainTrainer:
    def __init__(self, graph_data, config: dict):
        self.graph_data = graph_data
        self.config = config
        self.device = torch.device(config.get("gnn", {}).get("device", "cpu"))
        self.model_dir = Path(config.get("gnn", {}).get("model_dir", "param"))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_path = self.model_dir / f"marginal_node2vec_{graph_data.name}.pth"
        self.model_path = self.model_dir / f"marginal_gain_{graph_data.name}.pth"
        self.embeddings = load_or_create_node2vec_embeddings(
            graph_data.graph,
            self.embedding_path,
            dimensions=int(config.get("gnn", {}).get("embedding_dim", 64)),
            walk_length=int(config.get("gnn", {}).get("walk_length", 10)),
            num_walks=int(config.get("gnn", {}).get("num_walks", 10)),
            window=int(config.get("gnn", {}).get("window", 10)),
            workers=int(config.get("gnn", {}).get("workers", 1)),
        ).to(self.device)
        self.norm_degrees, _ = build_node_features(graph_data.graph, self.device)
        self.model = MarginalGainPredictor(
            self.embeddings.shape[1],
            int(config.get("gnn", {}).get("hidden_dim", 64)),
        ).to(self.device)

    def _tensor_batch(self, samples: list[MarginalGainSample]):
        masks = torch.zeros((len(samples), self.graph_data.num_nodes, 1), device=self.device)
        candidates = []
        labels = []
        for row, sample in enumerate(samples):
            if sample.seed_set:
                masks[row, sample.seed_set, 0] = 1.0
            candidates.append(sample.candidate)
            labels.append(sample.marginal_gain)
        return masks, torch.tensor(candidates, device=self.device), torch.tensor(labels, device=self.device).unsqueeze(-1)

    def train(self, output_dir: str | Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        splits = build_marginal_dataset(self.graph_data, self.config)
        dataset_path = output_dir / "marginal_dataset.json"
        import json
        dataset_path.write_text(json.dumps({key: [item.to_dict() for item in value] for key, value in splits.items()}, indent=2), encoding="utf-8")
        batch_size = int(self.config.get("gnn", {}).get("batch_size", 16))
        epochs = int(self.config.get("gnn", {}).get("epochs", 3))
        learning_rate = float(self.config.get("gnn", {}).get("learning_rate", 1e-3))
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        losses = []
        start = time.perf_counter()
        for _ in range(epochs):
            random.shuffle(splits["train"])
            epoch_loss = 0.0
            count = 0
            for begin in range(0, len(splits["train"]), batch_size):
                masks, candidates, labels = self._tensor_batch(splits["train"][begin:begin + batch_size])
                emb = self.embeddings.unsqueeze(0).expand(masks.shape[0], -1, -1)
                deg = self.norm_degrees.unsqueeze(0).expand(masks.shape[0], -1, -1)
                prediction = self.model(emb, deg, masks, candidates)
                loss = F.mse_loss(prediction, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                count += 1
            losses.append(epoch_loss / max(count, 1))
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "model_name": self.model.model_name,
            "model_version": self.model.model_version,
            "dataset": self.graph_data.name,
            "num_nodes": self.graph_data.num_nodes,
            "embedding_dim": self.embeddings.shape[1],
            "config": self.config,
        }
        torch.save(checkpoint, self.model_path)
        metrics = {
            "dataset": self.graph_data.name,
            "model_name": self.model.model_name,
            "model_version": self.model.model_version,
            "train_samples": len(splits["train"]),
            "validation_samples": len(splits["validation"]),
            "test_samples": len(splits["test"]),
            "epochs": epochs,
            "epoch_losses": losses,
            "train_time_seconds": time.perf_counter() - start,
        }
        return metrics, MarginalGainArtifacts(self.model_path, self.embedding_path, dataset_path)
