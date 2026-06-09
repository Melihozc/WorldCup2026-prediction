# Consensus-Hybrid (C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Consensus-Hybrid scale ("C") that brings the bookmaker market into the model as a team-strength signal — de-vig multi-book consensus, invert it through tournament simulation to recover draw-difficulty-corrected team abilities, blend with historical (Dixon-Coles) and squad-value abilities, then temperature-calibrate the champion distribution to the market — producing a defensible, well-calibrated WC2026 forecast and a thesis write-up around it.

**Architecture:** Single new module `src/consensus.py` holds a scalar-strength match model (`StrengthModel`), the inverse-simulation ability inference (`infer_market_abilities`), ability extractors (`hist_ability`, `squad_ability`), `blend_strengths`, and `calibrate_to_market`. It reuses `simulate.run_monte_carlo_cached` for all tournament simulation, `poisson.DixonColes` for the historical ability, `squad.SquadStrength` for squad value, and `market.implied_probs` for de-vigging. A thin runner `scripts/run_consensus.py` wires it end-to-end and emits `outputs/champion_probs_Consensus.csv` + meta. The honest-null finding becomes the thesis motivation; the consensus signal + calibration are the contribution.

**Tech Stack:** Python, numpy, pandas, scipy (`scipy.stats.poisson`), joblib (parallel MC, already used), pytest (tests).

**Why this is the right design (literature):** Zeileis et al. (the only public method with a WC track record — called 2010 winner, 3/4 of 2014 semis) fuse exactly three ability signals (historical bivariate-Poisson, bookmaker consensus, squad/market value) and simulate. Our shipped M+ already has historical + squad but **lacks the consensus signal** — the one that moves SOTA. The "inverse simulation" corrects for easy/hard groups; the calibrate-to-market step is the standard overconfidence fix (our current top-3 mass = 57% vs market 38%).

---

## File Structure

- **Create** `src/consensus.py` — all new modeling (consensus odds, StrengthModel, inverse-sim, blend, calibrate). One responsibility: turn signals into a calibrated per-team strength vector and simulate.
- **Create** `tests/test_consensus.py` — behavioral unit tests (no network, synthetic + tiny-N).
- **Create** `tests/__init__.py` — empty, makes `tests` a package.
- **Create** `scripts/run_consensus.py` — end-to-end wiring + outputs.
- **Modify** `requirements.txt` — add `pytest`.
- **Modify** `CLAUDE.md` — add architecture entry 16 (`src/consensus.py`) + C scale to Project section.
- **Create** `docs/thesis/dunya_kupasi_tahmin_modeli.md` — giriş / yöntem / bulgular / tartışma write-up.
- **Outputs** (gitignored dir, written at runtime): `outputs/champion_probs_Consensus.csv`, `outputs/model_Consensus_meta.csv`, `outputs/consensus_abilities.csv`.

Convention reminders from `CLAUDE.md`: outcome order `(W, D, L)` = `(0,1,2)`; `proba_fn(home, away, neutral)`, `goals_fn(rng, home, away, neutral)`; `fixed_groups` = 12 lists ordered A..L from `data.load_groups_2026()`. Project root has non-ASCII name — quote paths in shell.

---

## Task 1: Project test scaffold + pytest

**Files:**
- Create: `tests/__init__.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Create the test package marker**

Create `tests/__init__.py` with a single line:

```python
# test package
```

- [ ] **Step 2: Add pytest to requirements**

Append to `requirements.txt` (keep existing lines; add if absent):

```
pytest>=7.0
```

- [ ] **Step 3: Verify pytest available**

Run: `python -m pytest --version`
Expected: prints `pytest 7.x` (or later). If missing: `python -m pip install -r requirements.txt`.

- [ ] **Step 4: Commit**

```bash
git add tests/__init__.py requirements.txt
git commit -m "chore: add pytest test scaffold"
```

---

## Task 2: Consensus odds loader (multi-book de-vig + logit average)

**Files:**
- Create: `src/consensus.py`
- Test: `tests/test_consensus.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_consensus.py`:

```python
import numpy as np
import pandas as pd
import pytest

from src import consensus


def _write_odds(tmp_path, name, rows):
    p = tmp_path / name
    pd.DataFrame(rows, columns=["team", "decimal_odds", "book", "date", "source"]).to_csv(p, index=False)
    return p


def test_load_consensus_two_books_averages_and_normalizes(tmp_path):
    a = _write_odds(tmp_path, "wc2026_outright_2026-06-01.csv", [
        ["Spain", 5.0, "A", "2026-06-01", "x"],
        ["France", 6.0, "A", "2026-06-01", "x"],
        ["Brazil", 8.0, "A", "2026-06-01", "x"],
    ])
    b = _write_odds(tmp_path, "wc2026_outright_2026-06-02.csv", [
        ["Spain", 7.0, "B", "2026-06-02", "x"],
        ["France", 5.0, "B", "2026-06-02", "x"],
        ["Brazil", 9.0, "B", "2026-06-02", "x"],
    ])
    out = consensus.load_consensus_odds([a, b])
    assert list(out.columns) == ["team", "market_prob"]
    assert set(out["team"]) == {"Spain", "France", "Brazil"}
    assert out["market_prob"].sum() == pytest.approx(1.0, abs=1e-9)
    # Spain stronger in book A, France stronger in book B → close after averaging
    probs = dict(zip(out["team"], out["market_prob"]))
    assert probs["Spain"] > probs["Brazil"]
    assert probs["France"] > probs["Brazil"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_consensus.py::test_load_consensus_two_books_averages_and_normalizes -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.consensus'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/consensus.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_consensus.py::test_load_consensus_two_books_averages_and_normalizes -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/consensus.py tests/test_consensus.py
git commit -m "feat(consensus): multi-book consensus odds loader"
```

---

## Task 3: StrengthModel (scalar strength → Poisson goals → W/D/L)

**Files:**
- Modify: `src/consensus.py`
- Test: `tests/test_consensus.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_consensus.py`:

```python
def test_strength_model_proba_and_lambdas():
    m = consensus.StrengthModel({"A": 1.0, "B": -1.0}, scale=1.0, base=0.262)
    lh, la = m.lambdas("A", "B", neutral=True)
    assert lh > la  # stronger team scores more
    pw, pd_, pl = m.proba("A", "B", neutral=True)
    assert pw > pl  # stronger team more likely to win
    assert pw + pd_ + pl == pytest.approx(1.0, abs=1e-9)
    # symmetry: equal strength → pw ~= pl on neutral
    me = consensus.StrengthModel({"A": 0.0, "B": 0.0}, scale=1.0)
    pw2, pd2, pl2 = me.proba("A", "B", neutral=True)
    assert pw2 == pytest.approx(pl2, abs=1e-9)


def test_strength_goals_fn_callable_with_simulate_signature():
    m = consensus.StrengthModel({"A": 0.5, "B": -0.5})
    gf = m.goals_fn()
    rng = np.random.default_rng(0)
    gh, ga = gf(rng, "A", "B", True)
    assert isinstance(gh, int) and isinstance(ga, int)
    pf = m.proba_fn()
    out = pf("A", "B", True)
    assert len(out) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_consensus.py -k strength -v`
Expected: FAIL — `AttributeError: module 'src.consensus' has no attribute 'StrengthModel'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/consensus.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_consensus.py -k strength -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/consensus.py tests/test_consensus.py
git commit -m "feat(consensus): StrengthModel scalar-strength match model"
```

---

## Task 4: Cache builder + inverse-simulation ability inference

**Files:**
- Modify: `src/consensus.py`
- Test: `tests/test_consensus.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_consensus.py`:

```python
def _toy_groups():
    # 12 groups x 4 = 48 teams named T00..T47
    teams = [f"T{i:02d}" for i in range(48)]
    return [teams[i * 4:(i + 1) * 4] for i in range(12)]


def test_infer_market_abilities_recovers_rank_and_reduces_kl():
    groups = _toy_groups()
    teams = [t for g in groups for t in g]
    # synthetic market: geometric decay so ranks are unambiguous
    raw = np.array([0.97 ** i for i in range(len(teams))])
    market = pd.DataFrame({"team": teams, "market_prob": raw / raw.sum()})
    strengths, info = consensus.infer_market_abilities(
        market, groups, n_infer=3000, n_iter=8, lr=0.6, seed=1, n_jobs=1)
    assert set(strengths) == set(teams)
    # KL to market should drop over iterations
    assert info["kl_history"][-1] < info["kl_history"][0]
    # stronger market teams get higher inferred ability (rank agreement)
    mk = market.set_index("team")["market_prob"]
    s = pd.Series(strengths)
    corr = np.corrcoef(mk.rank(), s.reindex(mk.index).rank())[0, 1]
    assert corr > 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_consensus.py -k infer -v`
Expected: FAIL — `AttributeError: ... 'infer_market_abilities'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/consensus.py`:

```python
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
    kl_history: list = []
    for it in range(n_iter):
        psim = _sim_champ_probs(s, fixed_groups, scale, base, n_infer, seed + it, n_jobs)
        p = np.array([max(psim.get(t, 0.0), _EPS) for t in teams])
        p = p / p.sum()
        kl = float(np.sum(target * np.log(target / p)))
        kl_history.append(kl)
        if verbose:
            print(f"[infer] iter {it} KL={kl:.5f}")
        if kl < tol:
            break
        delta = lr * (np.log(target) - np.log(p))
        for i, t in enumerate(teams):
            s[t] += float(delta[i])
        mean_s = np.mean(list(s.values()))
        for t in teams:
            s[t] -= mean_s
    return s, {"kl_history": kl_history, "final_kl": kl_history[-1] if kl_history else None}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_consensus.py -k infer -v`
Expected: PASS. (Runs ~8 iters × 3000 sims; should finish in well under a minute.)

- [ ] **Step 5: Commit**

```bash
git add src/consensus.py tests/test_consensus.py
git commit -m "feat(consensus): inverse-simulation market ability inference"
```

---

## Task 5: Ability extractors + blend

**Files:**
- Modify: `src/consensus.py`
- Test: `tests/test_consensus.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_consensus.py`:

```python
def test_zscore_and_blend():
    z = consensus._zscore_dict({"A": 10.0, "B": 0.0, "C": -10.0})
    assert z["A"] > z["B"] > z["C"]
    assert abs(np.mean(list(z.values()))) < 1e-9
    blended = consensus.blend_strengths(
        {"m": {"A": 1.0, "B": -1.0}, "h": {"A": 2.0, "B": -2.0}},
        {"m": 0.5, "h": 0.5})
    assert blended["A"] == pytest.approx(1.5)
    assert blended["B"] == pytest.approx(-1.5)


def test_blend_handles_missing_component_team():
    blended = consensus.blend_strengths(
        {"m": {"A": 1.0, "B": 1.0}, "h": {"A": 3.0}},  # B missing in h
        {"m": 1.0, "h": 1.0})
    # B: (1*1 + 1*0)/2 = 0.5 ; A: (1+3)/2 = 2.0
    assert blended["A"] == pytest.approx(2.0)
    assert blended["B"] == pytest.approx(0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_consensus.py -k "zscore or blend" -v`
Expected: FAIL — `_zscore_dict` / `blend_strengths` undefined.

- [ ] **Step 3: Write minimal implementation**

Append to `src/consensus.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_consensus.py -k "zscore or blend" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/consensus.py tests/test_consensus.py
git commit -m "feat(consensus): ability extractors + z-score blend"
```

---

## Task 6: Temperature calibration to market

**Files:**
- Modify: `src/consensus.py`
- Test: `tests/test_consensus.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_consensus.py`:

```python
def test_calibrate_to_market_reduces_top_concentration():
    groups = _toy_groups()
    teams = [t for g in groups for t in g]
    raw = np.array([0.95 ** i for i in range(len(teams))])
    market = pd.DataFrame({"team": teams, "market_prob": raw / raw.sum()})
    # deliberately over-sharp strengths (large spread → overconfident champ dist)
    sharp = {t: 3.0 * (len(teams) - i) / len(teams) for i, t in enumerate(teams)}
    T, sim, sse = consensus.calibrate_to_market(
        sharp, market, groups, n=3000, seed=2, n_jobs=1,
        t_grid=np.linspace(0.2, 1.2, 6))
    assert 0.2 <= T <= 1.2
    # calibrated top-3 champ mass should be closer to market top-3 mass than raw
    mk_top3 = float(market.nlargest(3, "market_prob")["market_prob"].sum())
    cal_top3 = float(sim.nlargest(3, "P_Champion")["P_Champion"].sum())
    raw_sim = consensus._sim_champ_probs(sharp, groups, 1.0, 0.262, 3000, 2, 1)
    raw_top3 = sum(sorted(raw_sim.values(), reverse=True)[:3])
    assert abs(cal_top3 - mk_top3) <= abs(raw_top3 - mk_top3) + 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_consensus.py -k calibrate -v`
Expected: FAIL — `calibrate_to_market` undefined.

- [ ] **Step 3: Write minimal implementation**

Append to `src/consensus.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_consensus.py -k calibrate -v`
Expected: PASS.

- [ ] **Step 5: Run the full module test suite**

Run: `python -m pytest tests/test_consensus.py -v`
Expected: all PASS (8+ tests).

- [ ] **Step 6: Commit**

```bash
git add src/consensus.py tests/test_consensus.py
git commit -m "feat(consensus): temperature calibration to market"
```

---

## Task 7: End-to-end runner `scripts/run_consensus.py`

**Files:**
- Create: `scripts/run_consensus.py`

Mirror the wiring style of `scripts/run_m_plus.py` (read it first for the exact `load_results`, `EloRatings`/`DixonColes`, `load_groups_2026`, and output-dir conventions; adapt names as needed — the snippet below uses the documented APIs).

- [ ] **Step 1: Write the runner**

Create `scripts/run_consensus.py`:

```python
"""Consensus-Hybrid (C) runner.

Pipeline: market consensus → inverse-sim market ability → blend with historical
(Dixon-Coles) + squad ability → temperature-calibrate to market → MC the 2026
bracket → outputs/champion_probs_Consensus.csv + meta + abilities.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_results, load_groups_2026  # noqa: E402
from src.poisson import DixonColes  # noqa: E402
from src.squad import SquadStrength  # noqa: E402
from src import consensus  # noqa: E402
from src.market import compare_to_market  # noqa: E402
from src.simulate import run_monte_carlo_cached  # noqa: E402

OUT = ROOT / "outputs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000, help="final MC sims")
    ap.add_argument("--n-infer", type=int, default=15000, help="MC sims per inverse-sim iter")
    ap.add_argument("--iters", type=int, default=30, help="inverse-sim iterations")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--since", default="2014-01-01")
    ap.add_argument("--no-friendly", action="store_true")
    ap.add_argument("--w-market", type=float, default=0.6)
    ap.add_argument("--w-hist", type=float, default=0.3)
    ap.add_argument("--w-squad", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    groups_df = load_groups_2026()
    # 12 lists ordered A..L
    fixed_groups = [list(g["team"]) for _, g in groups_df.groupby("group", sort=True)]
    teams = [t for g in fixed_groups for t in g]

    # 1) market consensus
    market = consensus.load_consensus_odds()

    # 2) inverse-sim market ability
    print("[1/4] inverse-sim market abilities ...")
    market_str, info = consensus.infer_market_abilities(
        market, fixed_groups, n_infer=args.n_infer, n_iter=args.iters,
        seed=args.seed, n_jobs=args.jobs, verbose=True)
    market_ab = consensus._zscore_dict(market_str)

    # 3) historical (Dixon-Coles) + squad abilities
    print("[2/4] historical + squad abilities ...")
    hist = load_results()
    hist = hist[hist["date"] >= pd.Timestamp(args.since)]
    if args.no_friendly:
        hist = hist[hist["tournament"].str.lower() != "friendly"]
    dc = DixonColes().fit(
        hist.rename(columns={"home_team": "home_team", "away_team": "away_team"}))
    hist_ab = consensus.hist_ability(dc, teams)
    try:
        squad = SquadStrength.build()
        squad_ab = consensus.squad_ability(squad, teams, "2026-06-11")
    except Exception as e:  # squad data optional
        print(f"  squad unavailable ({e}); dropping squad component")
        squad_ab = {t: 0.0 for t in teams}

    # 4) blend + calibrate
    print("[3/4] blend + calibrate to market ...")
    blended = consensus.blend_strengths(
        {"market": market_ab, "hist": hist_ab, "squad": squad_ab},
        {"market": args.w_market, "hist": args.w_hist, "squad": args.w_squad})
    T, sim, sse = consensus.calibrate_to_market(
        blended, market, fixed_groups, n=args.n, seed=args.seed, n_jobs=args.jobs)

    # 5) final forecast already in `sim` at best T
    print("[4/4] writing outputs ...")
    sim.to_csv(OUT / "champion_probs_Consensus.csv", index=False)

    merged, summary = compare_to_market(sim, model_col="P_Champion")
    merged.to_csv(OUT / "model_vs_market_Consensus.csv", index=False)

    pd.DataFrame([{
        "team": t, "market_ability": market_ab.get(t, np.nan),
        "hist_ability": hist_ab.get(t, np.nan), "squad_ability": squad_ab.get(t, np.nan),
        "blended": blended.get(t, np.nan),
    } for t in teams]).to_csv(OUT / "consensus_abilities.csv", index=False)

    pd.DataFrame([{
        "n_sims": args.n, "n_infer": args.n_infer, "iters": args.iters,
        "temperature": T, "calibration_sse": sse,
        "infer_final_kl": info["final_kl"],
        "w_market": args.w_market, "w_hist": args.w_hist, "w_squad": args.w_squad,
        "spearman": summary["spearman"], "kl_model_market": summary["kl_model_market"],
        "model_top3_mass": summary["model_top3_mass"],
        "market_top3_mass": summary["market_top3_mass"],
    }]).to_csv(OUT / "model_Consensus_meta.csv", index=False)

    print(sim.head(10).to_string(index=False))
    print(f"T={T:.3f} KL={summary['kl_model_market']:.4f} "
          f"top3 model={summary['model_top3_mass']:.3f} market={summary['market_top3_mass']:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify imports + group wiring against the real codebase**

Run: `python -c "from src.data import load_results, load_groups_2026; df=load_groups_2026(); print(df.columns.tolist()); print(sorted(df['group'].unique()))"`
Expected: prints the group CSV columns and the 12 group labels. **If the column is not named `group` or `team`**, adjust `fixed_groups`/`load_consensus_odds` team handling accordingly (this is the one integration point most likely to differ — fix it here, not by guessing). Also confirm `DixonColes().fit(...)` expects `home_team, away_team, home_score, away_score, neutral, date` (per `poisson.py`); rename columns from `load_results()` if they differ.

- [ ] **Step 3: Smoke-run with tiny settings**

Run: `python scripts/run_consensus.py --n 2000 --n-infer 1500 --iters 4 --jobs 4`
Expected: completes, prints a top-10 table led by the market favorites (Spain/France/Argentina region), writes `outputs/champion_probs_Consensus.csv`. Top-3 mass should be **closer to market** than the M+ 0.57 (sanity: < 0.55).

- [ ] **Step 4: Commit**

```bash
git add scripts/run_consensus.py
git commit -m "feat(consensus): end-to-end Consensus-Hybrid runner"
```

---

## Task 8: Full production run + record numbers

**Files:** (outputs only — gitignored; meta is the record)

- [ ] **Step 1: Production run**

Run: `python scripts/run_consensus.py --n 50000 --n-infer 15000 --iters 30 --jobs 8 --no-friendly`
Expected: ~10–25 min. Writes `champion_probs_Consensus.csv`, `model_Consensus_meta.csv`, `consensus_abilities.csv`, `model_vs_market_Consensus.csv`.

- [ ] **Step 2: Sanity checks**

Run: `python -c "import pandas as pd; m=pd.read_csv('outputs/model_Consensus_meta.csv'); print(m.T)"`
Confirm: `model_top3_mass` materially below 0.57 and near `market_top3_mass`; `kl_model_market` below the M+ 0.161; `spearman` ≥ 0.85; `infer_final_kl` small (< 0.02).

- [ ] **Step 3: Commit the meta record**

```bash
git add outputs/model_Consensus_meta.csv
git commit -m "chore: record Consensus-Hybrid run metrics"
```

(Note: `outputs/*.csv` may be gitignored except meta — check `.gitignore`; if the dir is fully ignored, skip the add and rely on the thesis doc to carry the numbers.)

---

## Task 9: CLAUDE.md + thesis write-up

**Files:**
- Modify: `CLAUDE.md`
- Create: `docs/thesis/dunya_kupasi_tahmin_modeli.md`

- [ ] **Step 1: Add C scale to CLAUDE.md Project section**

In `CLAUDE.md`, after the M+ bullet in the Project list, add:

```markdown
- **C** — Consensus-Hybrid: bookmaker consensus as a strength signal. De-vig multi-book outright → inverse-simulation market ability (draw-difficulty corrected) → blend with Dixon-Coles + squad z-scores → temperature-calibrate champion dist to market (fixes M+ overconfidence). `scripts/run_consensus.py` → `outputs/champion_probs_Consensus.csv`. Zeileis et al. paradigm.
```

- [ ] **Step 2: Add architecture entry 16 to CLAUDE.md**

After entry 15 (`src/market.py`) add:

```markdown
16. **`src/consensus.py`** — Consensus-Hybrid (C). `load_consensus_odds` (multi-book de-vig + logit average), `StrengthModel` (scalar strength → independent-Poisson goals → W/D/L), `infer_market_abilities` (inverse-sim fixed point reproducing market champ probs), `hist_ability`/`squad_ability`/`_zscore_dict`/`blend_strengths`, `calibrate_to_market` (1-D temperature → min squared diff vs market). Simulation reuses `run_monte_carlo_cached`.
```

- [ ] **Step 3: Write the thesis document**

Create `docs/thesis/dunya_kupasi_tahmin_modeli.md` with this structure (fill Bulgular from Task 8 meta numbers):

```markdown
# 2026 FIFA Dünya Kupası Tahmin Modeli — Konsensüs-Hibrit Yaklaşım

## 1. Giriş
- Problem: 48 takım, 104 maç, yüksek varyans; amaç maç olasılıkları → turnuva olasılıkları.
- Literatür: Elo (Gilch & Müller 2018), Dixon-Coles/bivariate Poisson, Monte Carlo,
  hibrit ML (Zeileis et al. 2026), random forest (ACM 2024). Tek sinyal yetersiz;
  en iyi sonuç tarihsel + market + kadro sinyallerinin birleşiminde.

## 2. Yöntem
- S: Elo + heuristik beraberlik. M: Elo + Dixon-Coles. M+: Elo + XGB + DC + squad.
- C (katkı): konsensüs-hibrit — de-vig multi-book → inverse-sim market ability →
  blend (market/hist/squad z) → temperature kalibrasyon → MC (resmi 2026 bracket).
- Değerlendirme: RPS (birincil), bootstrap CI, paired significance, ECE; market KL/Spearman.

## 3. Bulgular
- Honest null: tarihsel-tek-başına (M+) Elo'yu anlamlı geçmiyor (RPS 0.18938 vs 0.18986,
  %95 CI sıfırı kesiyor) — literatürle (Gilch & Müller, peer modeller) tutarlı.
- M+ overconfidence: top-3 kütle %57 vs market %38.
- C sonucu: [Task 8 meta'dan doldur] temperature T=__, top-3 kütle %__ (markete yakın),
  KL=__ (M+ 0.161'den düşük), Spearman=__. Şampiyon dağılımı: [ilk 8 takım + olasılık].

## 4. Tartışma
- Model kesinlik değil, belirsizliği nicelleştiren çerçeve.
- Market sinyali + kalibrasyon overconfidence'ı düzeltti; tarihsel/kadro bağımsız bileşeni
  ağırlıklarla raporlandı. Sınır: outright market n=1, gerçek not 19 Tem 2026.
- Gelecek: tam Bayesian hierarchical Poisson (L), çok-bahisci canlı oran toplama.

## Kaynakça
(APA — kullanıcının listesi + Leitner-Zeileis-Hornik 2010, Baio & Blangiardo 2010,
Groll et al. 2018, Ley-Van de Wiele-Van Eetvelde 2019.)
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md docs/thesis/dunya_kupasi_tahmin_modeli.md
git commit -m "docs: add Consensus-Hybrid to CLAUDE.md + thesis write-up"
```

---

## Self-Review Notes

- **Spec coverage:** consensus signal (Task 2), strength model (Task 3), inverse-sim / draw-difficulty correction (Task 4), blend with historical+squad (Task 5), overconfidence calibration (Task 6), end-to-end forecast (Task 7-8), thesis framing (Task 9). All covered.
- **Integration risk (flagged in Task 7 Step 2):** the exact column names from `load_groups_2026()` and `load_results()` — verify against the real functions before the smoke run; this is the only place the plan assumes an API it has not read line-by-line.
- **Type consistency:** `StrengthModel.lambdas/proba` signatures match `simulate` callbacks `(home, away, neutral)`; `_build_caches` output shape matches `_sample_score_cached` (square flat pmf); `infer_market_abilities`/`calibrate_to_market` both consume `market_probs[[team, market_prob]]` and `fixed_groups` (A..L lists).
- **Data dependency:** Consensus needs ≥1 outright CSV (have BetMGM 2026-06-07). Multi-book improves the consensus but is not blocking — `load_consensus_odds` degrades to single-book. Gathering more books is a stretch enhancement, not in this plan.
```
