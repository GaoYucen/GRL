from __future__ import annotations

import torch
import torch.nn as nn


class SetEncoder(nn.Module):
    """Encode a variable-size seed set with a nonlinear permutation-invariant map."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.set_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, node_features: torch.Tensor, seed_mask: torch.Tensor) -> torch.Tensor:
        encoded = self.node_mlp(node_features)
        pooled = (encoded * seed_mask).sum(dim=-2)
        return self.set_mlp(pooled)


class MarginalGainPredictor(nn.Module):
    """Predict conditional influence gain for a candidate under the current seed set."""

    model_name = "marginal_gain_predictor"
    model_version = "1"

    def __init__(self, embedding_dim: int, hidden_dim: int = 64, structural_dim: int = 1):
        super().__init__()
        input_dim = embedding_dim + structural_dim
        self.seed_encoder = SetEncoder(input_dim, hidden_dim)
        self.candidate_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        norm_degrees: torch.Tensor,
        seed_mask: torch.Tensor,
        candidate_indices: int | torch.Tensor,
    ) -> torch.Tensor:
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
        if norm_degrees.dim() == 2:
            norm_degrees = norm_degrees.unsqueeze(0)
        if seed_mask.dim() == 2:
            seed_mask = seed_mask.unsqueeze(0)
        batch_size = embeddings.shape[0]
        candidates = torch.as_tensor(candidate_indices, device=embeddings.device, dtype=torch.long)
        if candidates.dim() == 0:
            candidates = candidates.expand(batch_size)
        candidate_features = torch.cat([embeddings, norm_degrees], dim=-1)
        seed_repr = self.seed_encoder(candidate_features, seed_mask)
        row = torch.arange(batch_size, device=embeddings.device)
        candidate_repr = self.candidate_encoder(candidate_features[row, candidates])
        interaction = seed_repr * candidate_repr
        difference = (seed_repr - candidate_repr).abs()
        output = self.head(torch.cat([seed_repr, candidate_repr, interaction, difference], dim=-1))
        return output
