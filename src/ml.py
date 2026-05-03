"""ML modelleri: LogReg + XGBoost + CatBoost (sklearn fallback).

Outcome: 0=W, 1=D, 2=L (home perspektifi).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier  # type: ignore
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier  # type: ignore
    HAS_CB = True
except ImportError:
    HAS_CB = False


FEATURES = [
    "elo_diff", "rank_diff", "fifa_pts_diff", "home_advantage",
    "form_diff", "gd_avg_diff",
    "attack_diff", "defense_diff",
    "pi_diff",
    "xg_for_diff", "xg_against_diff",
]


def fit_logreg(X: pd.DataFrame, y: np.ndarray) -> Pipeline:
    cols = [c for c in FEATURES if c in X.columns]
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, multi_class="multinomial",
                                    C=1.0)),
    ])
    pipe.fit(X[cols].fillna(0.0), y)
    pipe.feature_names_in_ = cols  # type: ignore
    return pipe


def fit_xgb(X: pd.DataFrame, y: np.ndarray, n_estimators: int = 300):
    cols = [c for c in FEATURES if c in X.columns]
    Xc = X[cols].fillna(0.0).to_numpy()
    if HAS_XGB:
        clf = XGBClassifier(
            n_estimators=n_estimators, max_depth=5, learning_rate=0.05,
            objective="multi:softprob", num_class=3, eval_metric="mlogloss",
            tree_method="hist",
        )
    else:
        clf = HistGradientBoostingClassifier(
            max_iter=n_estimators, max_depth=5, learning_rate=0.05,
        )
    clf.fit(Xc, y)
    clf._cols = cols  # type: ignore
    return clf


def fit_catboost(X: pd.DataFrame, y: np.ndarray, n_estimators: int = 300):
    cols = [c for c in FEATURES if c in X.columns]
    Xc = X[cols].fillna(0.0).to_numpy()
    if HAS_CB:
        clf = CatBoostClassifier(
            iterations=n_estimators, depth=6, learning_rate=0.05,
            loss_function="MultiClass", eval_metric="MultiClass",
            verbose=0,
        )
    else:
        clf = HistGradientBoostingClassifier(
            max_iter=n_estimators, max_depth=6, learning_rate=0.05,
        )
    clf.fit(Xc, y)
    clf._cols = cols  # type: ignore
    return clf


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    cols = getattr(model, "_cols", None) or list(getattr(
        model, "feature_names_in_", FEATURES))
    cols = [c for c in cols if c in X.columns]
    Xc = X[cols].fillna(0.0)
    return model.predict_proba(Xc.to_numpy() if hasattr(Xc, "to_numpy") else Xc)


def ensemble(probs_list: list[np.ndarray], weights: list[float] | None = None
             ) -> np.ndarray:
    """Ağırlıklı ortalama, her satırda normalize."""
    if weights is None:
        weights = [1.0] * len(probs_list)
    w = np.array(weights, dtype=float)
    w /= w.sum()
    out = np.zeros_like(probs_list[0])
    for p, wi in zip(probs_list, w):
        out += wi * p
    out /= out.sum(axis=1, keepdims=True)
    return out
