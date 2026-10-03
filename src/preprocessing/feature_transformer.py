"""Bộ biến đổi đặc trưng. Mọi thống kê (quantile, median, mean/std, danh mục)
chỉ được fit trên tập Train rồi áp cho Validation/Test."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ID_COLS = ["SK_ID_CURR"]
TARGET = "TARGET"


def split_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    feats = [c for c in df.columns if c not in ID_COLS + [TARGET]]
    cat_cols = [c for c in feats if df[c].dtype == object or isinstance(df[c].dtype, pd.CategoricalDtype)]
    num_cols = [c for c in feats if c not in cat_cols]
    return num_cols, cat_cols


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Winsorize mỗi cột số theo quantile học từ Train; inf -> NaN."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        X = np.where(np.isfinite(X), X, np.nan)
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        X = np.where(np.isfinite(X), X, np.nan)
        return np.clip(X, self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def make_linear_preprocessor(num_cols, cat_cols, clip_quantiles=(0.01, 0.99),
                             onehot_min_frequency=0.005) -> ColumnTransformer:
    """Cho Logistic Regression: clip -> impute median (+ cờ missing) -> chuẩn hóa; one-hot cho cột phân loại."""
    num_pipe = Pipeline([
        ("clip", QuantileClipper(*clip_quantiles)),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=onehot_min_frequency, sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        verbose_feature_names_out=False,
    )


class CategoryEncoder(BaseEstimator, TransformerMixin):
    """Cho LightGBM: giữ nguyên cột số (inf -> NaN), cột phân loại -> pandas category
    với danh mục học từ Train (giá trị mới ở Val/Test -> NaN)."""

    def __init__(self, cat_cols: list[str]):
        self.cat_cols = cat_cols

    def fit(self, X: pd.DataFrame, y=None):
        self.categories_ = {c: sorted(X[c].dropna().unique()) for c in self.cat_cols}
        self.columns_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.columns_].copy()
        num_cols = [c for c in self.columns_ if c not in self.cat_cols]
        X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan)
        for c in self.cat_cols:
            X[c] = pd.Categorical(X[c], categories=self.categories_[c])
        return X
