"""Geçmiş turnuva holdout backtest."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .eval import report
from .elo import EloRatings, HistoricalElo
from .poisson import DixonColes


OUTCOME_MAP = {"W": 0, "D": 1, "L": 2}


def outcomes_from_df(df: pd.DataFrame) -> np.ndarray:
    diff = df["home_score"].to_numpy() - df["away_score"].to_numpy()
    out = np.where(diff > 0, 0, np.where(diff == 0, 1, 2))
    return out


def evaluate_predictor(proba_fn, test_df: pd.DataFrame, date_aware: bool = False) -> dict:
    """proba_fn(home, away, neutral[, date]) -> (pw, pd, pl).
    date_aware=True ise her satır için r.date pozisyonel arg olarak geçilir.
    """
    probs = []
    for r in test_df.itertuples(index=False):
        if date_aware:
            probs.append(proba_fn(r.home_team, r.away_team, bool(r.neutral), r.date))
        else:
            probs.append(proba_fn(r.home_team, r.away_team, bool(r.neutral)))
    probs = np.array(probs)
    y = outcomes_from_df(test_df)
    return {**report(probs, y), "probs": probs}


def split_by_tournament(df: pd.DataFrame, holdout_tournament: str = "FIFA World Cup",
                        holdout_year: int = 2022) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["year"] = df["date"].dt.year
    test_mask = (df["tournament"] == holdout_tournament) & (df["year"] == holdout_year)
    train = df[~test_mask].drop(columns="year").reset_index(drop=True)
    test = df[test_mask].drop(columns="year").reset_index(drop=True)
    return train, test


def backtest_elo(df: pd.DataFrame, holdout_year: int = 2022) -> dict:
    train, test = split_by_tournament(df, holdout_year=holdout_year)
    elo = HistoricalElo()
    return evaluate_predictor(elo.predict_proba, test, date_aware=True)


def backtest_dc(df: pd.DataFrame, holdout_year: int = 2022,
                since: str = "2014-01-01") -> dict:
    train, test = split_by_tournament(df, holdout_year=holdout_year)
    train = train[train["date"] >= pd.Timestamp(since)]
    dc = DixonColes().fit(train)
    return evaluate_predictor(dc.predict_proba, test)


def split_by_tournaments(
    df: pd.DataFrame,
    holdouts: list[tuple[str, int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Multi-tournament holdout split.

    holdouts: [(tournament_name, year), ...] — bunları test seti yap, geri kalanlar train.
    """
    df = df.copy()
    df["year"] = df["date"].dt.year
    test_mask = pd.Series(False, index=df.index)
    for tournament, year in holdouts:
        test_mask |= (df["tournament"] == tournament) & (df["year"] == year)
    train = df[~test_mask].drop(columns="year").reset_index(drop=True)
    test = df[test_mask].drop(columns="year").reset_index(drop=True)
    return train, test


def backtest_market_baseline() -> dict:
    """Bahis verisi yoksa pas. Placeholder."""
    return {"note": "implement when betting odds available"}
