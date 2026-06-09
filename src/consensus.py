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
