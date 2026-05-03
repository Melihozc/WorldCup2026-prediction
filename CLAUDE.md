# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

2026 FIFA World Cup match prediction model (48 teams, 104 matches, US/Canada/Mexico, 11 Jun – 19 Jul 2026). Scales:

- **S** — Elo baseline (shipped). `scripts/run_baseline.py` → `outputs/champion_probs_S.csv`.
- **M** — Elo + Dixon-Coles weighted ensemble, holdout-tuned ensemble weight (shipped). `scripts/run_m.py` → `outputs/champion_probs_M.csv`.
- **M+** — Elo + XGBoost ensemble (W_ELO=0.0 → effectively XGB-only) with Dixon-Coles for goal sampling, Pi-ratings + FIFA rank features (shipped). `scripts/run_m_plus.py` → `outputs/champion_probs_Mplus.csv`. xG_agg StatsBomb feature kept out of training (see auto-memory `project-wc2026-xg-decision.md`); production-only sanity + reserved for M++.
- **L** — Bayesian hierarchical Poisson + player-level features (planned, not started).

Plan file: `C:\Users\Melih\.claude\plans\u-an-n-m-zdeki-2026-serialized-hennessy.md`.

Project root contains a non-ASCII directory name (`dünya_kupası_tahmini`). When invoking Python via shell, quote the path or use `cd` from a parent.

## Common commands

```bash
# install
python -m pip install -r requirements.txt

# smoke test — synthetic data, no Kaggle needed, validates entire pipeline
python scripts/smoke_test.py

# real run — requires data/raw/results.csv from Kaggle
#   martj42/international-football-results-from-1872-to-2017
python scripts/run_baseline.py            # S: n=10000 default
python scripts/run_baseline.py --n 50000  # higher MC count
python scripts/run_baseline.py --since 2010-01-01

# M scale — Elo + Dixon-Coles weighted ensemble
python scripts/run_m.py --n 50000                  # default WC2022 holdout
python scripts/run_m.py --n 50000 --no-friendly    # exclude friendlies from DC fit

# M+ scale — Elo + XGBoost (W_ELO=0.0 currently optimal) + DC goals
python scripts/run_m_plus.py --n 50000 --jobs 8 --no-friendly

# outputs → outputs/champion_probs_{S,M,Mplus}.csv (sorted by P_Champion)
#         + outputs/model_{M,Mplus}_meta.csv (ensemble weights, backtest metrics)
```

No tests directory wired up yet. `scripts/smoke_test.py` is the integration check — it asserts `top-5 P_Champion > 0.30`. If you add real tests, use pytest under `tests/`.

## Architecture

Pipeline stages (top to bottom = data flow):

1. **`src/data.py`** — loads Kaggle `results.csv` into a sorted-by-date DataFrame; provides `teams_2026()` (hardcoded participant list, override via `data/raw/user_provided/teams_2026.csv`). The hardcoded list is a 2025-Q4 best guess; update once real qualification finishes.

2. **`src/elo.py`** — `EloRatings` (S model). `fit(matches_df)` walks chronologically, updates in-place; `predict_proba(home, away, neutral)` returns `(P_W, P_D, P_L)`. Also: `EloRatings.from_snapshot()` loads `data/raw/eloratings/World.tsv` (current). `HistoricalElo` walks yearly snapshots for **walk-forward, leakage-free** backtest (`predict_proba(h, a, neutral, date)`). Two non-obvious pieces:
   - K-factor varies by `tournament` string (`K_BY_TOURNAMENT`); unknown → `DEFAULT_K = 30`.
   - S draw probability is heuristic `0.28 * exp(-0.0017 * |elo_diff|)`. M/M+ replace this with Dixon-Coles output.

3. **`src/poisson.py`** — Dixon-Coles Poisson model (M, M+). MLE fit via `scipy.optimize.minimize` (SLSQP) with `sum(attack)=0` constraint, time-decay weights `exp(-xi * days/365)` (`xi=0.0035` ≈ 5yr half-life). Provides `predict_proba(h, a, neutral)` (W/D/L), `score_matrix(h, a, neutral, max_goals=8)` (joint goal PMF with τ correction for 0-0/1-0/0-1/1-1), `sample_score(rng, h, a, neutral)`.

4. **`src/ratings.py`** — Pi-ratings (goal-difference based dynamic rating, used as XGB feature in M+). `build_pi_features(df)` is walk-forward, leakage-free.

5. **`src/fifa_rank.py`** — FIFA ranking history loader. Returns `rank(team)` and `points(team)`; used in M+ feature pipeline.

6. **`src/ml.py`** — XGBoost / CatBoost wrappers (sklearn `HistGradientBoostingClassifier` fallback when not installed). `FEATURES` list defines column order. `fit_xgb`, `fit_logreg`, `fit_catboost`, `predict_proba`, `ensemble`. `xg_for_diff/xg_against_diff` exist in FEATURES but excluded from M+ in production (see auto-memory).

7. **`src/features.py`** — match-level feature builder. `build_match_features(history, matches, elo, dc=None, pi=None, fifa_rank=None, xg_agg=None)`. Always emits `elo_diff, rank_diff, fifa_pts_diff, home_advantage, form_diff, gd_avg_diff`; optional `attack_diff/defense_diff/pi_diff/xg_for_diff/xg_against_diff`. `recent_form()` is O(N·M) — cache before long backtests.

8. **`src/simulate.py`** — Monte Carlo simulator for **2026-specific** format: 12 groups of 4, top-2 + best-8 third-place → Round of 32 (NOT 2022's 16-team R16). Three runners:
   - `run_monte_carlo(elo, ...)` — S, Elo-only path with consistency-forced goals.
   - `run_monte_carlo_cb(proba_fn, goals_fn, ...)` — M, callback-based, model-agnostic.
   - `run_monte_carlo_cached(proba_cache, score_cache, ...)` + `build_cache(proba_fn, goals_fn, teams)` — M+, precomputed pairwise matrices for parallel speed.
   Knockout draws are uniformly random — no real bracket-pairing logic yet.

9. **`src/backtest.py`** — `split_by_tournament(df, holdout_tournament, holdout_year)`, `evaluate_predictor(proba_fn, test_df, date_aware=False)`, `outcomes_from_df`. Used by M and M+ runners.

10. **`scripts/run_baseline.py`** — S wiring. `EloRatings.fit()` called on **full** history (not just `since`); `--since` reserved for downstream filtering.

11. **`scripts/run_m.py`** — M wiring: HistoricalElo (backtest) + EloRatings.from_snapshot (production), DixonColes fit on `--since 2014-01-01` (default), grid-search ensemble weight `w` on holdout (currently log-loss; RPS preferred per Conventions). `--no-friendly` excludes friendly maçlar from DC fit (recommended; M+ found this helps).

12. **`scripts/run_m_plus.py`** — M+ wiring: Pi-ratings + FIFA rank + DC features → XGB; W_ELO=0.0, W_XGB=1.0 (grid-search optimal — XGB already sees `elo_diff`). DC kept for goal sampling (`score_matrix`). WC2022 backtest: XGB RPS=0.2056 vs Elo RPS=0.2204.

13. **`src/eval.py`** — `log_loss`, `brier`, `rps`, `accuracy`, `report`. **RPS is the primary metric** for football (ordered W/D/L); prefer over accuracy when comparing models.

## Conventions

- Outcome encoding throughout: **0 = home win, 1 = draw, 2 = away win**. `EloRatings.predict_proba` returns this order.
- Probability vectors are always `(N, 3)` numpy arrays in `[W, D, L]` order. Don't reshuffle.
- All modules assume input matches are sorted by `date` ascending. `data.load_results()` enforces this.
- Random seed for reproducibility lives in `run_monte_carlo*(seed=...)`. Smoke test pins seed=0; baseline / M / M+ pin seed=42.
- Backtest holdout default: WC 2022 (64 maç). Single-tournament holdout is high-variance; multi-tournament backtest (WC + Euro + Copa) is a near-term improvement target for M.

## Data layout

`data/raw/` is gitignored. Place `results.csv` (Kaggle martj42 dataset) here. User-supplied files go under `data/raw/user_provided/` and override defaults (e.g. `teams_2026.csv` with one `team` column).
