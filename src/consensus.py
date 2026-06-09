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


def _build_caches(model: "StrengthModel", teams: list) -> tuple[dict, dict]:
    """proba_cache[(a,b)]=(pw,pd,pl); score_cache[(a,b)]=flat normalized pmf.

    Matches simulate.run_monte_carlo_cached expectations (square flat matrix).
    """
    proba_cache: dict = {}
    score_cache: dict = {}
    mg = model.max_goals
    ar = np.arange(mg + 1)
    for a in teams:
        for b in teams:
            if a == b:
                continue
            lh, la = model.lambdas(a, b, True)
            ph = poisson.pmf(ar, lh)
            pa = poisson.pmf(ar, la)
            m = np.outer(ph, pa)
            m = m / m.sum()
            score_cache[(a, b)] = m.flatten()
            pw = float(np.tril(m, -1).sum())
            pd_ = float(np.trace(m))
            pl = float(np.triu(m, 1).sum())
            proba_cache[(a, b)] = (pw, pd_, pl)
    return proba_cache, score_cache


def _sim_champ_probs(strengths: dict, fixed_groups: list, scale: float,
                     base: float, n: int, seed: int, n_jobs: int) -> dict:
    from .simulate import run_monte_carlo_cached
    teams = [t for g in fixed_groups for t in g]
    model = StrengthModel(strengths, scale=scale, base=base)
    pc, sc = _build_caches(model, teams)
    sim = run_monte_carlo_cached(pc, sc, fixed_groups, n=n, seed=seed, n_jobs=n_jobs)
    return dict(zip(sim["team"], sim["P_Champion"]))


def infer_market_abilities(market_probs: pd.DataFrame, fixed_groups: list,
                           scale: float = 1.0, base: float = 0.262,
                           n_infer: int = 10000, n_iter: int = 30, lr: float = 0.5,
                           seed: int = 42, n_jobs: int = 1, tol: float = 1e-3,
                           verbose: bool = False) -> tuple[dict, dict]:
    """Inverse simulation: find per-team strengths whose simulated champion
    distribution reproduces the market consensus (corrects draw difficulty).

    Fixed-point multiplicative log-update on champion-prob ratios. Returns
    (strengths dict, info dict with kl_history + final_kl).
    """
    teams = [t for g in fixed_groups for t in g]
    mp = dict(zip(market_probs["team"].astype(str), market_probs["market_prob"]))
    target = np.array([max(mp.get(t, _EPS), _EPS) for t in teams])
    target = target / target.sum()
    # init strengths from centered log market prob
    lp = np.log(target)
    s = {t: float(lp[i] - lp.mean()) for i, t in enumerate(teams)}
    # champion prob is super-linear in strength (7 KO rounds compound an edge),
    # so a full log-ratio step overshoots → oscillation/divergence. Damp + clip.
    damp, cap = 0.25, 0.35
    kl_history: list = []
    best_kl, best_s = np.inf, dict(s)
    for it in range(n_iter):
        psim = _sim_champ_probs(s, fixed_groups, scale, base, n_infer, seed + it, n_jobs)
        p = np.array([max(psim.get(t, 0.0), _EPS) for t in teams])
        p = p / p.sum()
        # KL(sim || target): bounded (target>0 everywhere), robust to sim zeros.
        kl = float(np.sum(p * np.log(p / target)))
        kl_history.append(kl)
        if kl < best_kl:
            best_kl, best_s = kl, dict(s)
        if verbose:
            print(f"[infer] iter {it} KL={kl:.5f}")
        if kl < tol:
            break
        delta = np.clip(damp * lr * (np.log(target) - np.log(p)), -cap, cap)
        for i, t in enumerate(teams):
            s[t] += float(delta[i])
        mean_s = np.mean(list(s.values()))
        for t in teams:
            s[t] -= mean_s
    return best_s, {"kl_history": kl_history, "final_kl": kl_history[-1] if kl_history else None,
                    "best_kl": best_kl}


def _zscore_dict(raw: dict) -> dict:
    vals = np.array(list(raw.values()), dtype=float)
    mu = float(np.nanmean(vals))
    sd = float(np.nanstd(vals))
    if sd == 0 or not np.isfinite(sd):
        return {k: 0.0 for k in raw}
    return {k: float((v - mu) / sd) for k, v in raw.items()}


def hist_ability(dc, teams: list) -> dict:
    """Per-team scalar strength from Dixon-Coles: attack - defense, z-scored."""
    raw = {t: dc.attack.get(t, 0.0) - dc.defense.get(t, 0.0) for t in teams}
    return _zscore_dict(raw)


def squad_ability(squad, teams: list, date) -> dict:
    """Per-team log squad value (z-scored). Missing teams filled with the min."""
    raw: dict = {}
    for t in teams:
        v, _age = squad.features(t, pd.Timestamp(date))
        raw[t] = float(v) if np.isfinite(v) else np.nan
    finite = [v for v in raw.values() if np.isfinite(v)]
    fill = min(finite) if finite else 0.0
    raw = {t: (v if np.isfinite(v) else fill) for t, v in raw.items()}
    return _zscore_dict(raw)


def blend_strengths(components: dict, weights: dict) -> dict:
    """components: {name: {team: z}}. weights: {name: w}. Missing team in a
    component contributes 0 (mean) for that component. Returns {team: blended}."""
    teams: set = set()
    for d in components.values():
        teams |= set(d.keys())
    wsum = float(sum(weights.values())) or 1.0
    return {t: float(sum(weights[n] * components[n].get(t, 0.0) for n in components) / wsum)
            for t in teams}


def calibrate_to_market(strengths: dict, market_probs: pd.DataFrame,
                        fixed_groups: list, scale: float = 1.0, base: float = 0.262,
                        n: int = 20000, seed: int = 42, n_jobs: int = 1,
                        t_grid=None) -> tuple[float, pd.DataFrame, float]:
    """1-D temperature search: multiply all strengths by T, pick T minimizing
    squared diff between simulated champion probs and market (overconfidence fix).

    Returns (best_T, simulated stage-prob DataFrame at best_T, best_sse).
    """
    from .simulate import run_monte_carlo_cached
    if t_grid is None:
        t_grid = np.linspace(0.3, 1.5, 13)
    teams = [t for g in fixed_groups for t in g]
    mp = dict(zip(market_probs["team"].astype(str), market_probs["market_prob"]))
    target = np.array([mp.get(t, 0.0) for t in teams])
    best: tuple | None = None
    for T in t_grid:
        s = {t: float(T) * v for t, v in strengths.items()}
        model = StrengthModel(s, scale=scale, base=base)
        pc, sc = _build_caches(model, teams)
        sim = run_monte_carlo_cached(pc, sc, fixed_groups, n=n, seed=seed, n_jobs=n_jobs)
        psim = dict(zip(sim["team"], sim["P_Champion"]))
        p = np.array([psim.get(t, 0.0) for t in teams])
        sse = float(np.sum((p - target) ** 2))
        if best is None or sse < best[2]:
            best = (float(T), sim, sse)
    return best
