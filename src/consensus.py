"""Consensus-Hybrid (C) — market consensus as a team-strength signal.

Zeileis paradigm: de-vig multi-bookmaker outright odds into a consensus
champion distribution, then invert it through tournament simulation to recover
draw-difficulty-corrected team abilities, blend with historical (Dixon-Coles)
and squad-value abilities, and temperature-calibrate the champion distribution
to the market. Fixes the M+ overconfidence (top-3 mass 57% vs market 38%).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import poisson

from .market import implied_probs, load_outright_odds

ROOT = Path(__file__).resolve().parents[1]
ODDS_DIR = ROOT / "data" / "raw" / "odds"
_EPS = 1e-6


def load_consensus_odds(paths: list | None = None) -> pd.DataFrame:
    """Multiple bookmakers' outright CSVs → consensus de-vigged champ probs.

    Each book de-vigged separately (market.implied_probs), then averaged on the
    logit scale (Leitner-Zeileis-Hornik 2010) and renormalized to sum 1.
    Single book → that book's de-vigged probs. Returns [team, market_prob] desc.
    """
    if paths is None:
        paths = sorted(ODDS_DIR.glob("wc2026_outright_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No outright CSV in {ODDS_DIR}")
    logits: dict[str, list] = {}
    for p in paths:
        odds = load_outright_odds(Path(p))
        probs, _ = implied_probs(odds)
        for _, r in probs.iterrows():
            q = min(max(float(r["market_prob"]), _EPS), 1 - _EPS)
            logits.setdefault(str(r["team"]).strip(), []).append(np.log(q / (1 - q)))
    teams = list(logits.keys())
    z = np.array([np.mean(v) for v in logits.values()])
    p = 1.0 / (1.0 + np.exp(-z))
    p = p / p.sum()
    return (pd.DataFrame({"team": teams, "market_prob": p})
            .sort_values("market_prob", ascending=False)
            .reset_index(drop=True))


class StrengthModel:
    """Per-team scalar strength → two independent Poisson goal rates → W/D/L.

    lambda_home = exp(base + scale*(s_home - s_away)/2 + home_adv*(not neutral))
    lambda_away = exp(base - scale*(s_home - s_away)/2)
    base = log(~1.3) baseline goals per side at equal strength.
    """

    def __init__(self, strengths: dict, scale: float = 1.0, base: float = 0.262,
                 home_adv: float = 0.0, max_goals: int = 8):
        self.strengths = dict(strengths)
        self.scale = float(scale)
        self.base = float(base)
        self.home_adv = float(home_adv)
        self.max_goals = int(max_goals)

    def lambdas(self, home: str, away: str, neutral: bool = True) -> tuple[float, float]:
        d = self.strengths.get(home, 0.0) - self.strengths.get(away, 0.0)
        ha = 0.0 if neutral else self.home_adv
        lh = float(np.exp(self.base + self.scale * d / 2.0 + ha))
        la = float(np.exp(self.base - self.scale * d / 2.0))
        return lh, la

    def _score_matrix(self, home: str, away: str, neutral: bool) -> np.ndarray:
        lh, la = self.lambdas(home, away, neutral)
        ph = poisson.pmf(np.arange(self.max_goals + 1), lh)
        pa = poisson.pmf(np.arange(self.max_goals + 1), la)
        m = np.outer(ph, pa)
        return m / m.sum()

    def proba(self, home: str, away: str, neutral: bool = True) -> tuple[float, float, float]:
        m = self._score_matrix(home, away, neutral)
        pw = float(np.tril(m, -1).sum())
        pd_ = float(np.trace(m))
        pl = float(np.triu(m, 1).sum())
        s = pw + pd_ + pl
        return pw / s, pd_ / s, pl / s

    def proba_fn(self):
        return lambda h, a, neutral=True: self.proba(h, a, neutral)

    def goals_fn(self):
        def _g(rng, h, a, neutral=True):
            lh, la = self.lambdas(h, a, neutral)
            return int(rng.poisson(lh)), int(rng.poisson(la))
        return _g
