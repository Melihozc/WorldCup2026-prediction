"""Pi-ratings (Constantinou & Fenton 2012).

Separate home/away ratings updated by goal difference.
No look-ahead: build_pi_features processes chronologically.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class PiRatings:
    gamma: float = 0.036   # learning rate from original paper
    c: float = 3.0         # expected_diff = (rh - ra) / c
    home_rating: dict[str, float] = field(default_factory=dict)
    away_rating: dict[str, float] = field(default_factory=dict)

    def _expected(self, home: str, away: str) -> float:
        return (self.home_rating.get(home, 0.0) - self.away_rating.get(away, 0.0)) / self.c

    def update(self, home: str, away: str, home_score: int, away_score: int) -> None:
        actual = float(home_score - away_score)
        error = actual - self._expected(home, away)
        self.home_rating[home] = self.home_rating.get(home, 0.0) + self.gamma * error
        self.away_rating[away] = self.away_rating.get(away, 0.0) - self.gamma * error

    def fit(self, matches: pd.DataFrame) -> "PiRatings":
        """matches must be sorted by date ascending."""
        for row in matches.itertuples(index=False):
            self.update(row.home_team, row.away_team,
                        int(row.home_score), int(row.away_score))
        return self

    def diff(self, home: str, away: str) -> float:
        """Feature: pi_home(home) - pi_away(away)."""
        return self.home_rating.get(home, 0.0) - self.away_rating.get(away, 0.0)

    def get(self, team: str) -> tuple[float, float]:
        return self.home_rating.get(team, 0.0), self.away_rating.get(team, 0.0)


def build_pi_features(df: pd.DataFrame,
                       gamma: float = 0.036,
                       c: float = 3.0) -> pd.Series:
    """Walk-forward pi_diff for every row. df must be sorted by date ascending."""
    pi = PiRatings(gamma=gamma, c=c)
    diffs = []
    for row in df.itertuples(index=False):
        diffs.append(pi.diff(row.home_team, row.away_team))
        pi.update(row.home_team, row.away_team,
                  int(row.home_score), int(row.away_score))
    return pd.Series(diffs, index=df.index, name="pi_diff")
