"""Encoder thống nhất giữa các client mà không gom dữ liệu thô về server.

Hai vòng trao đổi thống kê tổng hợp (tính trên node thuộc khách hàng Train của từng client):
  Vòng 1: client gửi quantile clip / median / cờ có-missing / tần suất category
          -> server lấy trung bình có trọng số (theo số giá trị không thiếu), hợp tập category.
  Vòng 2: client gửi n, sum, sum of squares của giá trị đã clip + impute
          -> server tính mean/std toàn cục (chính xác, như pooled variance).
Quantile/median gộp bằng trung bình có trọng số là xấp xỉ của quantile toàn cục.
Kết quả: mọi client có cùng không gian đặc trưng (cùng số chiều, cùng ý nghĩa) -> FedAvg được.
"""
import warnings

import numpy as np
import pandas as pd

OTHER, MISSING = "__OTHER__", "__MISSING__"


class FederatedTabularEncoder:
    def __init__(self, num_cols, cat_cols, clip_quantiles=(0.01, 0.99), min_frequency=0.005):
        self.num_cols = list(num_cols)
        self.cat_cols = list(cat_cols)
        self.clip_quantiles = clip_quantiles
        self.min_frequency = min_frequency

    def _num(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.num_cols].to_numpy(np.float64)
        X[~np.isfinite(X)] = np.nan
        return X

    # ---------- Vòng 1 ----------
    def local_stats_round1(self, df: pd.DataFrame, max_rows: int, rng: np.random.Generator) -> dict:
        X = self._num(df)
        sample = X if len(X) <= max_rows else X[rng.choice(len(X), max_rows, replace=False)]
        lo, hi = self.clip_quantiles
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # cột toàn NaN ở client này
            q = np.nanquantile(sample, [lo, 0.5, hi], axis=0)
        return {
            "n": len(df),
            "nonnull": (~np.isnan(X)).sum(axis=0),
            "quantiles": q,
            "has_missing": np.isnan(X).any(axis=0),
            "cat_counts": {c: df[c].value_counts(dropna=True).to_dict() for c in self.cat_cols},
        }

    def aggregate_round1(self, stats: list[dict]) -> None:
        w = np.stack([s["nonnull"] for s in stats]).astype(float)          # K x d
        q = np.stack([s["quantiles"] for s in stats])                       # K x 3 x d
        w3 = np.where(np.isnan(q), 0.0, w[:, None, :])
        tot = w3.sum(axis=0)
        agg = np.where(tot > 0, np.nansum(np.nan_to_num(q) * w3, axis=0) / np.maximum(tot, 1), np.nan)
        self.lo_ = np.where(np.isnan(agg[0]), -np.inf, agg[0])
        self.median_ = np.where(np.isnan(agg[1]), 0.0, agg[1])
        self.hi_ = np.where(np.isnan(agg[2]), np.inf, agg[2])
        self.missing_mask_ = np.any(np.stack([s["has_missing"] for s in stats]), axis=0)

        total_n = sum(s["n"] for s in stats)
        self.vocab_ = {}
        for c in self.cat_cols:
            counts: dict = {}
            for s in stats:
                for v, cnt in s["cat_counts"][c].items():
                    counts[v] = counts.get(v, 0) + cnt
            self.vocab_[c] = sorted(v for v, cnt in counts.items() if cnt / total_n >= self.min_frequency)

    # ---------- Vòng 2 ----------
    def _clip_impute(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = self._num(df)
        M = np.isnan(X[:, self.missing_mask_])
        X = np.clip(X, self.lo_, self.hi_)
        X = np.where(np.isnan(X), self.median_, X)
        return X, M

    def local_stats_round2(self, df: pd.DataFrame, chunk: int = 1_000_000) -> dict:
        s = np.zeros(len(self.num_cols))
        ss = np.zeros(len(self.num_cols))
        for i in range(0, len(df), chunk):
            X, _ = self._clip_impute(df.iloc[i:i + chunk])
            s += X.sum(axis=0)
            ss += (X ** 2).sum(axis=0)
        return {"n": len(df), "sum": s, "sumsq": ss}

    def aggregate_round2(self, stats: list[dict]) -> None:
        n = sum(st["n"] for st in stats)
        self.mean_ = sum(st["sum"] for st in stats) / n
        var = sum(st["sumsq"] for st in stats) / n - self.mean_ ** 2
        std = np.sqrt(np.maximum(var, 0))
        self.std_ = np.where(std < 1e-8, 1.0, std)

    # ---------- Áp dụng ----------
    def _transform_chunk(self, df: pd.DataFrame) -> np.ndarray:
        X, M = self._clip_impute(df)
        parts = [((X - self.mean_) / self.std_).astype(np.float32), M.astype(np.float32)]
        for c in self.cat_cols:
            vocab = self.vocab_[c]
            codes = pd.Categorical(df[c], categories=vocab).codes.astype(np.int64)
            codes = np.where(codes >= 0, codes, np.where(df[c].isna().values, len(vocab) + 1, len(vocab)))
            oh = np.zeros((len(df), len(vocab) + 2), dtype=np.float32)
            oh[np.arange(len(df)), codes] = 1
            parts.append(oh)
        return np.concatenate(parts, axis=1)

    def transform(self, df: pd.DataFrame, chunk: int = 1_000_000) -> np.ndarray:
        if len(df) == 0:
            return np.zeros((0, len(self.feature_names_out())), dtype=np.float32)
        return np.concatenate([self._transform_chunk(df.iloc[i:i + chunk]) for i in range(0, len(df), chunk)])

    def feature_names_out(self) -> list[str]:
        names = list(self.num_cols)
        names += [f"{c}_missingindicator" for c, m in zip(self.num_cols, self.missing_mask_) if m]
        for c in self.cat_cols:
            names += [f"{c}_{v}" for v in self.vocab_[c]] + [f"{c}_{OTHER}", f"{c}_{MISSING}"]
        return names
