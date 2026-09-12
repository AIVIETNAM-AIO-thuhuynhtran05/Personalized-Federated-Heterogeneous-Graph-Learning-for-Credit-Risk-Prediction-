# Kiểm tra cách tính PR-AUC khác average precision và cách biểu diễn metric không xác định.
import pytest
from src.evaluation.metrics import auc_metrics


def test_pr_auc_is_trapezoidal_and_distinct_from_average_precision():
    metrics = auc_metrics([0, 1], [0.9, 0.1])
    assert metrics["roc_auc"] == metrics["auc"] == 0.0
    assert metrics["pr_auc"] == pytest.approx(0.25)
    assert metrics["average_precision"] == pytest.approx(0.5)


def test_perfect_ranking():
    metrics = auc_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert metrics["roc_auc"] == metrics["pr_auc"] == 1.0


@pytest.mark.parametrize("labels,scores", [([], []), ([0, 0], [0.1, 0.2]), ([1, 1], [0.8, 0.9])])
def test_degenerate_sets_are_explicitly_unscored(labels, scores):
    metrics = auc_metrics(labels, scores)
    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None
