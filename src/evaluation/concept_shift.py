"""Đo concept shift giữa các client: P(y | x) có khác nhau giữa các tổ chức không.

Ba phép đo bổ sung cho nhau:
  1. Ma trận chéo: mô hình Local-only của client i dự đoán Test của client j.
  2. Hiệu chỉnh theo client: y ~ logit(p) với hệ số chung / hệ số chặn riêng / độ dốc riêng
     theo client, kiểm định tỷ số hợp lý (LRT).
  3. Kiểm định tương tác: thêm mã client vào mô hình bảng có giúp dự đoán không.
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ---------- 1. Ma trận chéo ----------
def cross_client_summary(matrix: pd.DataFrame) -> pd.DataFrame:
    """matrix[i, j] = AUC của mô hình client i trên Test của client j.
    own: mô hình của chính client j; foreign_mean / foreign_best: mô hình của client khác;
    drop_mean: trung bình AUC_i(j) - AUC_i(i) qua i != j (mô hình bị giảm bao nhiêu khi sang client khác)."""
    rows = {}
    for j in matrix.columns:
        others = [i for i in matrix.index if i != j]
        rows[j] = {
            "own": matrix.loc[j, j],
            "foreign_mean": matrix.loc[others, j].mean(),
            "foreign_best": matrix.loc[others, j].max(),
            "own_minus_foreign_mean": matrix.loc[j, j] - matrix.loc[others, j].mean(),
            "drop_when_applied_here": float(np.mean([matrix.loc[i, j] - matrix.loc[i, i] for i in others])),
        }
    return pd.DataFrame(rows).T


# ---------- 2. Hiệu chỉnh theo client ----------
def _fit_ll(X, y) -> float:
    m = LogisticRegression(penalty=None, max_iter=2000, fit_intercept=False).fit(X, y)
    return -log_loss(y, m.predict_proba(X)[:, 1], normalize=False)


def calibration_shift_test(y, prob, client) -> dict:
    """M0: y ~ a + b*s (chung)   M1: y ~ a_k + b*s (hệ số chặn riêng: lệch mức rủi ro nền)
    M2: y ~ a_k + b_k*s (độ dốc riêng: cùng điểm số nhưng quan hệ với rủi ro khác nhau).
    s = logit(p). LRT M0->M1 và M1->M2."""
    y, s, client = np.asarray(y), _logit(prob), np.asarray(client)
    ks = np.unique(client)
    D = (client[:, None] == ks[None, :]).astype(float)
    X0 = np.column_stack([np.ones_like(s), s])
    X1 = np.column_stack([D, s])
    X2 = np.column_stack([D, D * s[:, None]])
    ll = [_fit_ll(X, y) for X in (X0, X1, X2)]
    k = len(ks)
    lrt01, lrt12 = 2 * (ll[1] - ll[0]), 2 * (ll[2] - ll[1])
    per = {}
    for c in ks:
        mask = client == c
        m = LogisticRegression(penalty=None, max_iter=2000).fit(s[mask, None], y[mask])
        per[int(c)] = {"intercept": float(m.intercept_[0]), "slope": float(m.coef_[0, 0]),
                       "observed_rate": float(y[mask].mean()),
                       "mean_predicted": float(np.mean(prob[mask])), "n": int(mask.sum())}
    n = len(y)
    return {
        "lrt_intercepts": {"stat": lrt01, "df": k - 1, "p_value": float(chi2.sf(lrt01, k - 1))},
        "lrt_slopes": {"stat": lrt12, "df": k - 1, "p_value": float(chi2.sf(lrt12, k - 1))},
        # Cỡ hiệu ứng: log-loss giảm bao nhiêu trên mỗi khách hàng khi cho phép riêng theo client
        "logloss_gain_per_sample_intercepts": (ll[1] - ll[0]) / n,
        "logloss_gain_per_sample_slopes": (ll[2] - ll[1]) / n,
        "per_client": per,
    }


def bootstrap_slopes(y, prob, client, rounds: int, rng) -> dict:
    """Khoảng tin cậy 95% của độ dốc hiệu chỉnh từng client."""
    y, s, client = np.asarray(y), _logit(prob), np.asarray(client)
    out = {}
    for c in np.unique(client):
        idx = np.flatnonzero(client == c)
        vals = []
        for _ in range(rounds):
            b = rng.choice(idx, len(idx))
            if y[b].min() == y[b].max():
                continue
            vals.append(LogisticRegression(penalty=None, max_iter=1000).fit(s[b, None], y[b]).coef_[0, 0])
        out[int(c)] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
    return out


# ---------- 3. Kiểm định tương tác ----------
def per_client_auc(y, prob, client) -> dict:
    y, prob, client = map(np.asarray, (y, prob, client))
    return {int(c): float(roc_auc_score(y[client == c], prob[client == c])) for c in np.unique(client)}


def paired_bootstrap_by_client(y, a, b, client, rounds: int, rng) -> dict:
    """ΔAUC = AUC(a) - AUC(b) theo từng client, CI 95% paired bootstrap."""
    y, a, b, client = map(np.asarray, (y, a, b, client))
    out = {}
    for c in np.unique(client):
        idx = np.flatnonzero(client == c)
        diffs = []
        for _ in range(rounds):
            i = rng.choice(idx, len(idx))
            if y[i].min() == y[i].max():
                continue
            diffs.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
        d = roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx])
        out[int(c)] = {"delta_auc": float(d), "ci95": [float(np.percentile(diffs, 2.5)),
                                                        float(np.percentile(diffs, 97.5))]}
    return out
