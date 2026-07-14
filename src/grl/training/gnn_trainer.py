from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn.functional as F
import torch.optim as optim

from grl.baselines import select_degree_discount_nodes
from grl.diffusion import estimate_spread
from grl.models import SpreadPredictorGNN, build_node_features, load_or_create_node2vec_embeddings


@dataclass
class GNNArtifacts:
    model_path: Path
    embedding_path: Path
    output_dir: Path


class GNNTrainer:
    def __init__(self, graph_data, config: dict):
        self.graph_data = graph_data
        self.config = config
        self.device = torch.device(config.get("gnn", {}).get("device", "cpu"))
        self.model_dir = Path(config.get("gnn", {}).get("model_dir", "param"))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_path = self.model_dir / f"node2vec_{self.graph_data.name}.pth"
        self.model_path = self.model_dir / f"gnn_{self.graph_data.name}.pth"

        self.embeddings = load_or_create_node2vec_embeddings(
            self.graph_data.graph,
            cache_path=self.embedding_path,
            dimensions=int(config["gnn"].get("embedding_dim", 64)),
            walk_length=int(config["gnn"].get("walk_length", 10)),
            num_walks=int(config["gnn"].get("num_walks", 10)),
            window=int(config["gnn"].get("window", 10)),
            workers=int(config["gnn"].get("workers", 1)),
            quiet=True,
        ).to(self.device)
        self.norm_degrees, _ = build_node_features(self.graph_data.graph, device=self.device)
        self.model = SpreadPredictorGNN(
            embedding_dim=self.embeddings.shape[1],
            hidden_dim=int(config["gnn"].get("hidden_dim", 64)),
        ).to(self.device)

    def _sample_seed_sets(self, total_samples: int, max_budget: int) -> list[list[int]]:
        nodes = list(self.graph_data.graph.nodes())
        top_k = min(len(nodes), int(self.config["gnn"].get("top_degree_pool", 1000)))
        ranked = sorted(self.graph_data.graph.out_degree() if self.graph_data.graph.is_directed() else self.graph_data.graph.degree(), key=lambda x: (-x[1], x[0]))
        top_nodes = [node for node, _ in ranked[:top_k]]
        greedy_seq = select_degree_discount_nodes(
            self.graph_data.graph,
            budget=min(max_budget, self.graph_data.num_nodes),
            probability=float(self.config["diffusion"].get("probability", 0.01)),
        )
        samples: list[list[int]] = []
        for idx in range(total_samples):
            branch = idx % 4
            if branch == 0:
                samples.append([random.choice(nodes)])
            elif branch == 1:
                k = random.randint(1, max_budget)
                samples.append(random.sample(nodes, k))
            elif branch == 2:
                k = random.randint(1, min(max_budget, len(top_nodes)))
                samples.append(random.sample(top_nodes, k))
            else:
                k = random.randint(1, min(max_budget, len(greedy_seq)))
                samples.append(greedy_seq[:k])
        return samples

    def build_dataset(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        total_samples = int(self.config["gnn"].get("samples", 128))
        max_budget = int(self.config["seed"].get("budget", 10))
        mc_runs = int(self.config["gnn"].get("mc_runs_train", 50))
        base_seed = int(self.config["experiment"].get("random_seed", 42))
        dataset = []
        for idx, seeds in enumerate(self._sample_seed_sets(total_samples, max_budget)):
            mask = torch.zeros((self.graph_data.num_nodes, 1), dtype=torch.float32)
            mask[seeds] = 1.0
            spread = estimate_spread(self.graph_data.graph, seeds, mc_runs=mc_runs, random_seed=base_seed + idx)["mean"]
            dataset.append((mask, torch.tensor([spread], dtype=torch.float32)))
        return dataset

    def train(self, output_dir: str | Path) -> tuple[dict, GNNArtifacts]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset = self.build_dataset()
        batch_size = int(self.config["gnn"].get("batch_size", 16))
        epochs = int(self.config["gnn"].get("epochs", 3))
        learning_rate = float(self.config["gnn"].get("learning_rate", 1e-3))
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model.train()

        start = time.perf_counter()
        epoch_losses = []
        for _ in range(epochs):
            random.shuffle(dataset)
            total_loss = 0.0
            batch_count = 0
            for begin in range(0, len(dataset), batch_size):
                batch = dataset[begin: begin + batch_size]
                masks = torch.stack([item[0] for item in batch]).to(self.device)
                labels = torch.stack([item[1] for item in batch]).to(self.device)
                batch_embs = self.embeddings.unsqueeze(0).expand(masks.shape[0], -1, -1)
                batch_degs = self.norm_degrees.unsqueeze(0).expand(masks.shape[0], -1, -1)
                preds = self.model(batch_embs, batch_degs, masks)
                loss = F.mse_loss(preds, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batch_count += 1
            epoch_losses.append(total_loss / max(batch_count, 1))
        elapsed = time.perf_counter() - start
        torch.save(self.model.state_dict(), self.model_path)

        metrics = {
            "dataset": self.graph_data.name,
            "train_samples": len(dataset),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "train_time_seconds": elapsed,
            "epoch_losses": epoch_losses,
        }
        return metrics, GNNArtifacts(self.model_path, self.embedding_path, output_dir)
