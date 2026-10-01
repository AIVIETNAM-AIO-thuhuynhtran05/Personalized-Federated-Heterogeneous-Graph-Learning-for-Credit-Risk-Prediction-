import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             log_loss, precision_recall_curve, roc_auc_score, roc_curve)


def select_threshold(y_true, y_prob, criterion: str = "f1") -> float:
    """Chọn threshold trên tập Validation. Không bao giờ gọi hàm này với tập Test."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    if criterion == "f1":
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
        return float(thresholds[np.argmax(f1[:-1])])
    if criterion == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        return float(thresholds[np.argmax(tpr - fpr)])
    raise ValueError(f"Unknown threshold criterion: {criterion}")


def ks_statistic(y_true, y_prob) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def evaluate(y_true, y_prob, threshold: float) -> dict:
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        # Không phụ thuộc threshold
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "ks": ks_statistic(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "log_loss": log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7)),
        # Tại threshold đã chọn trên Validation
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "n": int(len(y_true)), "positive_rate": float(y_true.mean()),
    }
