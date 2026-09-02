import pytest
import torch

from grl.evaluation.uncertainty import conservative_order, summarize_ensemble


def test_ensemble_summary_has_zero_uncertainty_for_identical_members():
    summary = summarize_ensemble(torch.tensor([[1.0, 2.0], [1.0, 2.0]]))
    assert torch.allclose(summary.mean, torch.tensor([1.0, 2.0]))
    assert torch.allclose(summary.std, torch.zeros(2))
    assert torch.allclose(summary.lower_bound, summary.mean)


def test_conservative_order_penalizes_high_variance_candidate():
    predictions = torch.tensor([[10.0, 9.0], [0.0, 9.0]])
    assert conservative_order(predictions, confidence_scale=1.0).tolist() == [1, 0]


@pytest.mark.parametrize(
    "predictions",
    [torch.tensor([1.0, 2.0]), torch.empty((0, 2)), torch.empty((2, 0))],
)
def test_summary_rejects_invalid_shapes(predictions):
    with pytest.raises(ValueError):
        summarize_ensemble(predictions)


def test_summary_rejects_negative_confidence_scale():
    with pytest.raises(ValueError):
        summarize_ensemble(torch.ones((2, 2)), confidence_scale=-1.0)
