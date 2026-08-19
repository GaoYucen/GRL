from grl.evaluation.ranking import pairwise_accuracy, regression_ranking_metrics, top_k_recall


def test_ranking_metrics():
    predictions = [0.1, 0.8, 0.4]
    targets = [1.0, 3.0, 2.0]
    metrics = regression_ranking_metrics(predictions, targets, top_ks=(1, 2))
    assert metrics["pairwise_accuracy"] == 1.0
    assert metrics["top_1_recall"] == 1.0
    assert top_k_recall(predictions, targets, 2) == 1.0
    assert pairwise_accuracy(predictions, targets) == 1.0
