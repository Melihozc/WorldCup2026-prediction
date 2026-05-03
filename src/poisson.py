"""Dixon-Coles Poisson model.

λ_home = exp(α_home - β_away + γ + home_adv)
λ_away = exp(α_away - β_home)

Düşük skor (0-0, 1-0, 0-1, 1-1) için τ düzeltmesi (Dixon & Coles 1997).
Time-decay ağırlık: exp(-ξ * gün_farkı / 365).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


def _tau(x: int, y: int, lh: float, la: float, rho: float) -> float:
    """Dixon-Coles düşük skor düzeltmesi."""
    if x == 0 and y == 0:
        return 1 - lh * la * rho
    if x == 0 and y == 1:
        return 1 + lh * rho
    if x == 1 and y == 0:
        return 1 + la * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


@dataclass
class DixonColes:
    """Dixon-Coles MLE fit."""
    teams: list[str] = field(default_factory=list)
    attack: dict[str, float] = field(default_factory=dict)
    defense: dict[str, float] = field(default_factory=dict)
    home_adv: float = 0.25
    rho: float = -0.05
    xi: float = 0.0035  # ~5 yıl yarı-ömür

    def _negloglik(self, params: np.ndarray, hi: np.ndarray, ai: np.ndarray,
                    hs: np.ndarray, as_: np.ndarray, neutral: np.ndarray,
                    weights: np.ndarray, n: int) -> float:
        att = params[:n]
        deff = params[n:2 * n]
        gamma = params[2 * n]
        rho = params[2 * n + 1]
        lh = np.exp(att[hi] - deff[ai] + gamma * (1 - neutral))
        la = np.exp(att[ai] - deff[hi])
        ll = (
            hs * np.log(lh) - lh - _logfact(hs)
            + as_ * np.log(la) - la - _logfact(as_)
        )
        # tau düzeltme — vektörize
        tau = np.ones_like(lh)
        m00 = (hs == 0) & (as_ == 0)
        m01 = (hs == 0) & (as_ == 1)
        m10 = (hs == 1) & (as_ == 0)
        m11 = (hs == 1) & (as_ == 1)
        tau[m00] = 1 - lh[m00] * la[m00] * rho
        tau[m01] = 1 + lh[m01] * rho
        tau[m10] = 1 + la[m10] * rho
        tau[m11] = 1 - rho
        tau = np.maximum(tau, 1e-10)
        ll = ll + np.log(tau)
        return -np.sum(weights * ll)

    def fit(self, matches: pd.DataFrame, ref_date: pd.Timestamp | None = None,
            min_matches: int = 5) -> "DixonColes":
        """matches: date, home_team, away_team, home_score, away_score, neutral.

        Sadece her iki takımın min_matches+ verisi olanı tut.
        """
        df = matches.copy()
        counts = pd.concat([df["home_team"], df["away_team"]]).value_counts()
        valid = counts[counts >= min_matches].index
        df = df[df["home_team"].isin(valid) & df["away_team"].isin(valid)].copy()
        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        hi = df["home_team"].map(idx).to_numpy()
        ai = df["away_team"].map(idx).to_numpy()
        hs = df["home_score"].to_numpy(dtype=float)
        as_ = df["away_score"].to_numpy(dtype=float)
        neutral = df["neutral"].to_numpy(dtype=float)
        ref = ref_date or df["date"].max()
        days = (ref - df["date"]).dt.days.to_numpy()
        weights = np.exp(-self.xi * days / 365.0)

        x0 = np.concatenate([
            np.zeros(n),       # attack
            np.zeros(n),       # defense
            [0.25],            # home_adv
            [-0.05],           # rho
        ])
        # constraint: sum(attack) = 0
        cons = ({"type": "eq", "fun": lambda p: np.sum(p[:n])},)
        bounds = [(-3, 3)] * n + [(-3, 3)] * n + [(-0.5, 1.0), (-0.2, 0.2)]
        res = minimize(
            self._negloglik, x0,
            args=(hi, ai, hs, as_, neutral, weights, n),
            method="SLSQP", bounds=bounds, constraints=cons,
            options={"maxiter": 200, "ftol": 1e-6},
        )
        p = res.x
        self.teams = teams
        self.attack = {t: float(p[i]) for i, t in enumerate(teams)}
        self.defense = {t: float(p[n + i]) for i, t in enumerate(teams)}
        self.home_adv = float(p[2 * n])
        self.rho = float(p[2 * n + 1])
        return self

    def lambdas(self, home: str, away: str, neutral: bool = True
                 ) -> tuple[float, float]:
        a_h = self.attack.get(home, 0.0)
        d_h = self.defense.get(home, 0.0)
        a_a = self.attack.get(away, 0.0)
        d_a = self.defense.get(away, 0.0)
        gamma = self.home_adv * (0 if neutral else 1)
        lh = np.exp(a_h - d_a + gamma)
        la = np.exp(a_a - d_h)
        return float(lh), float(la)

    def score_matrix(self, home: str, away: str, neutral: bool = True,
                     max_goals: int = 8) -> np.ndarray:
        """Her (h,a) skoru için olasılık. tau düzeltmeli."""
        lh, la = self.lambdas(home, away, neutral)
        ph = poisson.pmf(np.arange(max_goals + 1), lh)
        pa = poisson.pmf(np.arange(max_goals + 1), la)
        m = np.outer(ph, pa)
        for i in (0, 1):
            for j in (0, 1):
                m[i, j] *= _tau(i, j, lh, la, self.rho)
        m /= m.sum()
        return m

    def predict_proba(self, home: str, away: str, neutral: bool = True,
                       max_goals: int = 8) -> tuple[float, float, float]:
        m = self.score_matrix(home, away, neutral, max_goals)
        p_w = float(np.tril(m, -1).sum())  # home > away
        p_d = float(np.trace(m))
        p_l = float(np.triu(m, 1).sum())
        return p_w, p_d, p_l

    def sample_score(self, rng: np.random.Generator, home: str, away: str,
                     neutral: bool = True, max_goals: int = 8) -> tuple[int, int]:
        m = self.score_matrix(home, away, neutral, max_goals)
        flat = m.flatten()
        flat /= flat.sum()
        idx = rng.choice(len(flat), p=flat)
        return int(idx // m.shape[1]), int(idx % m.shape[1])


# log faktöriyel — büyük integer için scipy.special kullanmadan hızlı
_LOG_FACT_CACHE = np.array([0.0])


def _logfact(x: np.ndarray) -> np.ndarray:
    global _LOG_FACT_CACHE
    m = int(np.max(x)) + 1
    if m > len(_LOG_FACT_CACHE):
        _LOG_FACT_CACHE = np.cumsum(np.concatenate([[0.0], np.log(np.arange(1, m + 1))]))
    return _LOG_FACT_CACHE[x.astype(int)]
