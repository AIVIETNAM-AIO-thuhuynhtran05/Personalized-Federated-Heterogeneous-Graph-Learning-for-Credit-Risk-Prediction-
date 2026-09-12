"""AUC is undefined for empty or single-class local test sets."""
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, average_precision_score


def auc_metrics(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("Labels and scores must be equal-length 1D arrays")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Expected binary labels and finite prediction scores")
    roc_auc = pr_auc = average_precision = None
    if len(np.unique(labels)) == 2:
        roc_auc = float(roc_auc_score(labels, scores))
        precision, recall, _ = precision_recall_curve(labels, scores)
        pr_auc = float(auc(recall, precision))
        average_precision = float(average_precision_score(labels, scores))
    # Keep auc as a backward-compatible alias; PR-AUC uses trapezoidal integration.
    return {"auc": roc_auc, "roc_auc": roc_auc, "pr_auc": pr_auc,
            "average_precision": average_precision,
            "test_count": len(labels), "test_positive": int(labels.sum())}
