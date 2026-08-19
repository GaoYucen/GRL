import torch

from grl.models import MarginalGainPredictor


def test_marginal_predictor_forward_shape_and_seed_conditioning():
    model = MarginalGainPredictor(embedding_dim=4, hidden_dim=8)
    embeddings = torch.randn(5, 4)
    degrees = torch.rand(5, 1)
    empty = torch.zeros(5, 1)
    seeded = empty.clone()
    seeded[0] = 1.0
    first = model(embeddings, degrees, empty, 1)
    second = model(embeddings, degrees, seeded, 1)
    assert first.shape == (1, 1)
    assert second.shape == (1, 1)
    assert not torch.allclose(first, second)
