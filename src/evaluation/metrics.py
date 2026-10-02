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


def evaluate_by_client(df, prob_col: str, thresholds: dict) -> dict:
    """df: SK_ID_CURR, client, TARGET, prob_col (chỉ các dòng Test).
    thresholds: {client: threshold} (đã chọn trên Val). Trả về metric từng client + pooled:
    metric không phụ thuộc threshold tính trên toàn bộ dự đoán gộp; precision/recall/F1 pooled
    tính từ tổng ma trận nhầm lẫn của các client (mỗi client dùng threshold của mình)."""
    per = {int(k): evaluate(g["TARGET"], g[prob_col], thresholds[k]) for k, g in df.groupby("client")}
    pooled = evaluate(df["TARGET"], df[prob_col], 0.5)
    tp, fp, tn, fn = (sum(m[x] for m in per.values()) for x in ["tp", "fp", "tn", "fn"])
    precision, recall, specificity = tp / max(tp + fp, 1), tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    pooled.update({"threshold": None, "precision": precision, "recall": recall,
                   "f1": 2 * precision * recall / max(precision + recall, 1e-12),
                   "specificity": specificity, "balanced_accuracy": (recall + specificity) / 2,
                   "tp": tp, "fp": fp, "tn": tn, "fn": fn})
    weights = [m["n"] for m in per.values()]
    pooled["client_avg_roc_auc"] = float(np.average([m["roc_auc"] for m in per.values()], weights=weights))
    pooled["client_worst_roc_auc"] = float(min(m["roc_auc"] for m in per.values()))
    return {"per_client": per, "pooled": pooled}
