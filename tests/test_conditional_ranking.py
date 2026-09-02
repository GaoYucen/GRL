import pytest

from grl.evaluation.conditional_ranking import (
    CandidateRankingGroup,
    evaluate_conditional_rankings,
)


def test_conditional_ranking_is_computed_within_each_seed_state():
    groups = [
        CandidateRankingGroup((0,), (1, 2, 3), (0.9, 0.2, 0.1), (3.0, 1.0, 0.5)),
        CandidateRankingGroup((2,), (0, 1, 3), (0.2, 0.8, 0.1), (0.4, 2.0, 0.2)),
    ]
    metrics = evaluate_conditional_rankings(groups, top_ks=(1, 2))
    assert metrics["groups"] == 2
    assert metrics["conditional_spearman"] == pytest.approx(1.0)
    assert metrics["conditional_kendall"] == pytest.approx(1.0)
    assert metrics["top1_accuracy"] == pytest.approx(1.0)
    assert metrics["mean_regret"] == pytest.approx(0.0)
    assert metrics["recall_at_1"] == pytest.approx(1.0)
    assert metrics["recall_at_2"] == pytest.approx(1.0)


def test_seed_state_groups_are_not_merged():
    groups = [
        CandidateRankingGroup((), (0, 1), (0.9, 0.1), (1.0, 0.0)),
        CandidateRankingGroup((0,), (1, 2), (0.1, 0.9), (0.0, 1.0)),
    ]
    metrics = evaluate_conditional_rankings(groups, top_ks=(1,))
    assert metrics["conditional_spearman"] == pytest.approx(1.0)
    assert metrics["top1_accuracy"] == pytest.approx(1.0)


def test_invalid_group_shapes_are_rejected():
    with pytest.raises(ValueError):
        CandidateRankingGroup((), (0, 1), (1.0,), (1.0, 0.0))
