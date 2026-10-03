"""Centralized tabular baselines: Logistic Regression và LightGBM.
Encoder fit trên Train; Validation chỉ dùng để chọn siêu tham số / early stopping."""
import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from src.preprocessing.feature_transformer import CategoryEncoder, make_linear_preprocessor

log = logging.getLogger(__name__)


def train_logistic_regression(X_train: pd.DataFrame, y_train, X_val: pd.DataFrame, y_val,
                              num_cols, cat_cols, cfg: dict, seed: int) -> tuple[Pipeline, dict]:
    prep = make_linear_preprocessor(num_cols, cat_cols, cfg["clip_quantiles"],
                                    cfg["onehot_min_frequency"])
    Xtr = prep.fit_transform(X_train)
    Xva = prep.transform(X_val)
    log.info("LR: %d features after encoding", Xtr.shape[1])

    best_model, best_auc, history = None, -np.inf, {}
    for C in cfg["C_grid"]:
        model = LogisticRegression(C=C, max_iter=cfg["max_iter"], random_state=seed)
        model.fit(Xtr, y_train)
        auc = roc_auc_score(y_val, model.predict_proba(Xva)[:, 1])
        history[str(C)] = auc
        log.info("LR: C=%g  val AUC=%.5f", C, auc)
        if auc > best_auc:
            best_model, best_auc = model, auc

    info = {"best_C": best_model.C, "val_auc_by_C": history, "n_features": int(Xtr.shape[1])}
    return Pipeline([("prep", prep), ("model", best_model)]), info


def train_lightgbm(X_train: pd.DataFrame, y_train, X_val: pd.DataFrame, y_val,
                   cat_cols, cfg: dict, seed: int) -> tuple[Pipeline, dict]:
    prep = CategoryEncoder(cat_cols).fit(X_train)
    Xtr, Xva = prep.transform(X_train), prep.transform(X_val)

    model = lgb.LGBMClassifier(**cfg["params"], random_state=seed)
    model.fit(
        Xtr, y_train,
        eval_set=[(Xva, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(cfg["early_stopping_rounds"], verbose=False),
                   lgb.log_evaluation(500)],
    )
    importance = pd.Series(model.booster_.feature_importance("gain"), index=Xtr.columns)
    info = {
        "best_iteration": int(model.best_iteration_),
        "n_features": int(Xtr.shape[1]),
        "top_features_gain": importance.sort_values(ascending=False).head(30).round(1).to_dict(),
    }
    return Pipeline([("prep", prep), ("model", model)]), info
