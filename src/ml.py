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
    "squad_value_diff", "squad_age_diff",
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


def _make_xgb(n_estimators: int = 300, max_depth: int = 5,
              learning_rate: float = 0.05, subsample: float = 1.0,
              colsample_bytree: float = 1.0):
    """XGBClassifier (yoksa HistGradientBoosting fallback) — ortak kurucu."""
    if HAS_XGB:
        return XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            subsample=subsample, colsample_bytree=colsample_bytree,
            objective="multi:softprob", num_class=3, eval_metric="mlogloss",
            tree_method="hist",
        )
    return HistGradientBoostingClassifier(
        max_iter=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
    )


def fit_xgb(X: pd.DataFrame, y: np.ndarray, n_estimators: int = 300):
    cols = [c for c in FEATURES if c in X.columns]
    Xc = X[cols].fillna(0.0).to_numpy()
    clf = _make_xgb(n_estimators=n_estimators)
    clf.fit(Xc, y)
    clf._cols = cols  # type: ignore
    return clf


def fit_xgb_tuned(X: pd.DataFrame, y: np.ndarray, cv: int = 5,
                  trials: int = 40, seed: int = 0, n_jobs: int = 1):
    """Hiperparametre araması: Optuna (RPS K-fold) varsa, yoksa RandomizedSearchCV.

    Döndürür: refit edilmiş model. `model._tune_info` = {method, best_params, cv_metric}.
    """
    from src.eval import rps as _rps
    cols = [c for c in FEATURES if c in X.columns]
    Xc = X[cols].fillna(0.0).to_numpy()
    info = {"method": None, "best_params": None, "cv_metric": None}

    try:
        import optuna
        from sklearn.model_selection import StratifiedKFold
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = dict(
                n_estimators=trial.suggest_int("n_estimators", 150, 500, step=50),
                max_depth=trial.suggest_int("max_depth", 3, 7),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                subsample=trial.suggest_float("subsample", 0.7, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
            )
            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
            scores = []
            for tr, va in skf.split(Xc, y):
                m = _make_xgb(**params)
                m.fit(Xc[tr], y[tr])
                scores.append(_rps(m.predict_proba(Xc[va]), y[va]))
            return float(np.mean(scores))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=trials, show_progress_bar=False)
        best = study.best_params
        clf = _make_xgb(**best)
        clf.fit(Xc, y)
        info.update(method="optuna", best_params=best, cv_metric=study.best_value)
    except ImportError:
        from sklearn.model_selection import RandomizedSearchCV
        if HAS_XGB:
            dist = dict(
                n_estimators=[150, 200, 300, 400, 500],
                max_depth=[3, 4, 5, 6, 7],
                learning_rate=[0.01, 0.03, 0.05, 0.08, 0.1],
                subsample=[0.7, 0.85, 1.0],
                colsample_bytree=[0.7, 0.85, 1.0],
            )
        else:
            dist = dict(
                max_iter=[150, 200, 300, 400, 500],
                max_depth=[3, 4, 5, 6, 7],
                learning_rate=[0.01, 0.03, 0.05, 0.08, 0.1],
            )
        rs = RandomizedSearchCV(
            _make_xgb(), dist, n_iter=trials, scoring="neg_log_loss",
            cv=cv, random_state=seed, n_jobs=n_jobs,
        )
        rs.fit(Xc, y)
        clf = rs.best_estimator_
        info.update(method="randomized_search", best_params=rs.best_params_,
                    cv_metric=-float(rs.best_score_))

    clf._cols = cols  # type: ignore
    clf._tune_info = info  # type: ignore
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
