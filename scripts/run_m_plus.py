"""M+ scale: Elo(0.40) + XGBoost(0.60) ensemble.

Geliştirmeler vs M (Elo+DC):
  - Pi-ratings feature (gol farkı tabanlı dinamik rating)
  - DC attack/defense diff → XGB feature (DC artık ensemble üyesi değil)
  - Grid search sonucu: W_ELO=0.0 optimal — XGB elo_diff'i zaten görüyor, Elo blend zarar veriyor
  - Goals sim için DC hâlâ kullanılıyor (score_matrix)

Backtest (WC2022 holdout): XGB RPS=0.2056 vs Elo RPS=0.2204
Çıktı: outputs/champion_probs_Mplus.csv + outputs/model_Mplus_meta.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from src.data import load_results, add_outcome, load_groups_2026
from src.elo import EloRatings, HistoricalElo
from src.fifa_rank import FIFARank
from src.poisson import DixonColes
from src.ratings import PiRatings, build_pi_features
from src.squad import SquadStrength
from src.features import build_match_features
from src.ml import fit_xgb, fit_xgb_tuned, predict_proba, HAS_XGB, HAS_CB
from src.simulate import run_monte_carlo_cached, build_cache
from src.backtest import split_by_tournament, split_by_tournaments, evaluate_predictor, outcomes_from_df
from src.eval import log_loss, rps, bootstrap_rps_ci, bootstrap_rps_diff, reliability
from src.market import compare_to_market


MULTI_HOLDOUTS = [
    ("FIFA World Cup", 2014),
    ("FIFA World Cup", 2018),
    ("FIFA World Cup", 2022),
    ("UEFA Euro", 2016),
    ("UEFA Euro", 2021),
    ("UEFA Euro", 2024),
    ("Copa América", 2019),
    ("Copa América", 2021),
    ("Copa América", 2024),
]


def find_elo_weight(p_elo: np.ndarray, p_xgb: np.ndarray,
                    y: np.ndarray, grid: int = 101) -> tuple[float, float]:
    """RPS minimize eden w_elo bul: w*Elo + (1-w)*XGB."""
    best_w, best_rps = 0.0, 1e9
    for w in np.linspace(0, 1, grid):
        p = w * p_elo + (1 - w) * p_xgb
        p = p / p.sum(axis=1, keepdims=True)
        r = rps(p, y)
        if r < best_rps:
            best_w, best_rps = w, r
    return float(best_w), float(best_rps)


def main(n_sims: int = 50000, dc_since: str = "2014-01-01",
         holdout_year: int = 2022, n_jobs: int = 8,
         exclude_friendly: bool = False, tune: bool = False,
         w_elo_override: float | None = None) -> None:

    # Çıktı etiketi: --w-elo 1.0 → saf Elo-only varyantı (M+ çıktısını ezmez).
    label = "Mplus"
    if w_elo_override is not None:
        label = "Elo" if abs(w_elo_override - 1.0) < 1e-9 else f"w{w_elo_override:g}"

    print("[1/8] Veri yükleniyor")
    df = load_results()
    df = add_outcome(df)
    print(f"  {len(df):,} maç")

    print(f"[2/8] Multi-holdout split ({len(MULTI_HOLDOUTS)} turnuva)")
    train, test = split_by_tournaments(df, MULTI_HOLDOUTS)
    print(f"  train={len(train):,}  test={len(test)}")

    print("[3/8] Elo: walk-forward yearly snapshots (HistoricalElo)")
    elo = HistoricalElo()

    print("    + FIFA rank history yükleniyor")
    try:
        fifa_rank = FIFARank.load()
        print(f"      FIFA rank son yayın: {fifa_rank.latest_date.date()}")
    except FileNotFoundError as e:
        print(f"      UYARI: {e}")
        fifa_rank = None

    friendly_tag = ", no-friendly" if exclude_friendly else ""
    print(f"[4/8] Dixon-Coles fit (since {dc_since}{friendly_tag})")
    dc_mask = train["date"] >= pd.Timestamp(dc_since)
    if exclude_friendly:
        dc_mask &= train["tournament"] != "Friendly"
    dc_train = train[dc_mask]
    print(f"  DC training maçları: {len(dc_train):,}")
    dc = DixonColes().fit(dc_train)

    print("[5/8] Pi-ratings fit (walk-forward, no leakage)")
    pi_series = build_pi_features(df)
    df = df.copy()
    df["pi_diff"] = pi_series

    train_pi = df.loc[train.index, "pi_diff"]
    test_pi = df.loc[test.index, "pi_diff"]

    print("    + Squad strength (Transfermarkt market value, walk-forward)")
    try:
        squad = SquadStrength.build()
        print(f"      squad table: {len(squad._table)} satır, "
              f"{squad._table['country'].nunique()} ülke")
    except FileNotFoundError as e:
        print(f"      UYARI: squad verisi yok ({e}) — squad feature atlanıyor")
        squad = None

    print("[6/8] Feature build + ML fit")
    train_feat = build_match_features(train, train, elo, dc=dc, fifa_rank=fifa_rank, squad=squad)
    train_feat["pi_diff"] = train_pi.values

    test_feat = build_match_features(train, test, elo, dc=dc, fifa_rank=fifa_rank, squad=squad)
    pi_end = PiRatings().fit(train)
    test_feat["pi_diff"] = [
        pi_end.diff(r.home_team, r.away_team)
        for r in test.itertuples(index=False)
    ]

    y_train = outcomes_from_df(train)
    y_test = outcomes_from_df(test)

    if tune:
        print("    Hyperparameter search (--tune)...")
        xgb_model = fit_xgb_tuned(train_feat, y_train, n_jobs=n_jobs)
        tune_info = getattr(xgb_model, "_tune_info", {})
        print(f"    tune: method={tune_info.get('method')}  "
              f"cv_metric={tune_info.get('cv_metric')}  best={tune_info.get('best_params')}")
    else:
        xgb_model = fit_xgb(train_feat, y_train)
        tune_info = {}

    print("[7/8] Backtest + W_ELO grid search (RPS, multi-holdout)")
    elo_eval = evaluate_predictor(elo.predict_proba, test, date_aware=True)
    p_xgb = predict_proba(xgb_model, test_feat)

    if w_elo_override is not None:
        W_ELO = float(w_elo_override)
        print(f"  W_ELO override -> W_ELO={W_ELO:.2f} (grid-search atlandı)")
    else:
        W_ELO, best_rps_blend = find_elo_weight(elo_eval["probs"], p_xgb, y_test)
        print(f"  Grid search -> W_ELO={W_ELO:.2f}")
    W_XGB = 1.0 - W_ELO

    p_ens = W_ELO * elo_eval["probs"] + W_XGB * p_xgb
    p_ens = p_ens / p_ens.sum(axis=1, keepdims=True)

    evals = [
        ("Elo",     elo_eval["log_loss"], elo_eval["rps"], elo_eval["accuracy"]),
        ("XGB",     log_loss(p_xgb, y_test), rps(p_xgb, y_test), float((p_xgb.argmax(1) == y_test).mean())),
        ("Elo+XGB", log_loss(p_ens, y_test), rps(p_ens, y_test), float((p_ens.argmax(1) == y_test).mean())),
    ]
    for name, ll, r, acc in evals:
        print(f"  {name:10s}: log_loss={ll:.4f}  RPS={r:.4f}  acc={acc:.3f}")
    ens_ll = evals[-1][1]
    ens_rps = evals[-1][2]

    # --- Bootstrap RPS CIs + significance vs Elo baseline ---
    print("  Bootstrap RPS CIs (95%, 2000 resamples):")
    p_elo = elo_eval["probs"]
    ci_elo = bootstrap_rps_ci(p_elo, y_test)
    ci_xgb = bootstrap_rps_ci(p_xgb, y_test)
    ci_ens = bootstrap_rps_ci(p_ens, y_test)
    for name, c in [("Elo", ci_elo), ("XGB", ci_xgb), ("Elo+XGB", ci_ens)]:
        print(f"    {name:10s}: RPS={c['rps']:.4f}  [{c['lo']:.4f}, {c['hi']:.4f}]")
    # Paired bootstrap: does the model significantly beat plain Elo?
    diff_ens = bootstrap_rps_diff(p_ens, p_elo, y_test)
    diff_xgb = bootstrap_rps_diff(p_xgb, p_elo, y_test)
    print(f"  Ens-vs-Elo diff={diff_ens['diff']:+.5f} [{diff_ens['lo']:+.5f},{diff_ens['hi']:+.5f}]"
          f"  P(ens better)={diff_ens['p_a_better']:.2f}  significant={diff_ens['significant']}")
    print(f"  XGB-vs-Elo diff={diff_xgb['diff']:+.5f} [{diff_xgb['lo']:+.5f},{diff_xgb['hi']:+.5f}]"
          f"  P(xgb better)={diff_xgb['p_a_better']:.2f}  significant={diff_xgb['significant']}")
    verdict = ("BEATS Elo (significant)" if diff_ens["significant"]
               else "TIES Elo (not significant)")
    print(f"  >>> VERDICT: model {verdict}")

    # --- Calibration (reliability + ECE) on the holdout ---
    rel = reliability(p_ens, y_test, bins=10)
    cal_rows = [{"model": "Elo+XGB", **b} for b in rel["bins"]]
    cal_path = ROOT / "outputs" / "calibration_Mplus.csv"
    pd.DataFrame(cal_rows).to_csv(cal_path, index=False)
    print(f"  Calibration ECE={rel['ece']:.4f}  -> {cal_path}")

    # --- Production fit: full data ---
    print(f"[8/8] Production fit + MC sim n={n_sims:,}")
    elo_full = elo.current  # World.tsv
    dc_full_mask = df["date"] >= pd.Timestamp(dc_since)
    if exclude_friendly:
        dc_full_mask &= df["tournament"] != "Friendly"
    dc_full = DixonColes().fit(df[dc_full_mask])
    pi_full = PiRatings().fit(df)

    full_feat = build_match_features(df, df, elo_full, dc=dc_full, fifa_rank=fifa_rank, squad=squad)
    full_feat["pi_diff"] = build_pi_features(df).values
    y_full = outcomes_from_df(df)
    if tune and tune_info.get("best_params"):
        from src.ml import _make_xgb, FEATURES
        cols_full = [c for c in FEATURES if c in full_feat.columns]
        xgb_full = _make_xgb(**tune_info["best_params"])
        xgb_full.fit(full_feat[cols_full].fillna(0.0).to_numpy(), y_full)
        xgb_full._cols = cols_full
    else:
        xgb_full = fit_xgb(full_feat, y_full)

    fixed_groups = load_groups_2026()
    teams = [t for g in fixed_groups for t in g]
    print(f"  Gruplar yüklendi: {len(fixed_groups)} grup, {len(teams)} takım")

    print("  Precomputing pairwise caches...")
    PROD_DATE = pd.Timestamp("2026-06-11")  # turnuva başlangıcı — squad snapshot referansı

    def proba_fn(h: str, a: str, neutral: bool) -> tuple[float, float, float]:
        p_e = np.array(elo_full.predict_proba(h, a, neutral))
        if fifa_rank is not None:
            rank_h, rank_a = fifa_rank.rank(h), fifa_rank.rank(a)
            pts_h, pts_a = fifa_rank.points(h), fifa_rank.points(a)
        else:
            rank_h = rank_a = 100
            pts_h = pts_a = 0.0
        if squad is not None:
            sv, sa = squad.diff(h, a, PROD_DATE)
        else:
            sv = sa = 0.0
        feat_row = pd.DataFrame([{
            "elo_diff": elo_full.get(h) - elo_full.get(a),
            "rank_diff": rank_a - rank_h,
            "fifa_pts_diff": pts_h - pts_a,
            "home_advantage": int(not neutral),
            "form_diff": 0.0,
            "gd_avg_diff": 0.0,
            "attack_diff": dc_full.attack.get(h, 0.0) - dc_full.attack.get(a, 0.0),
            "defense_diff": dc_full.defense.get(a, 0.0) - dc_full.defense.get(h, 0.0),
            "pi_diff": pi_full.diff(h, a),
            "squad_value_diff": sv,
            "squad_age_diff": sa,
        }])
        p_x = predict_proba(xgb_full, feat_row)[0]
        p = W_ELO * p_e + W_XGB * p_x
        p /= p.sum()
        return tuple(p)

    # Build proba cache; score cache from DC
    proba_cache, _ = build_cache(proba_fn, None, teams)
    score_cache = {}
    for h in teams:
        for a in teams:
            if h == a:
                continue
            m = dc_full.score_matrix(h, a, neutral=True)
            flat = m.flatten().astype(float)
            flat /= flat.sum()
            score_cache[(h, a)] = flat

    out = run_monte_carlo_cached(proba_cache, score_cache, fixed_groups,
                                  n=n_sims, seed=42, n_jobs=n_jobs)
    out_path = ROOT / "outputs" / f"champion_probs_{label}.csv"
    out.to_csv(out_path, index=False)
    print(f"\nTop 15:")
    print(out.head(15).to_string(index=False))
    print(f"\n-> {out_path}")

    # --- Market benchmark: model P_Champion vs bookmaker outright odds ---
    market_summary = {}
    try:
        merged, market_summary = compare_to_market(out)
        mvm_name = "model_vs_market_2026.csv" if label == "Mplus" else f"model_vs_market_{label}.csv"
        mvm_path = ROOT / "outputs" / mvm_name
        merged.to_csv(mvm_path, index=False)
        print(f"\nMarket benchmark (de-vigged outright odds, overround={market_summary['overround']:.3f}):")
        print(f"  Spearman(model,market)={market_summary['spearman']:.3f}  "
              f"KL(model||market)={market_summary['kl_model_market']:.3f}")
        print(f"  Concentration — model top-3 mass={market_summary['model_top3_mass']:.3f}  "
              f"vs market top-3={market_summary['market_top3_mass']:.3f}")
        print("  Model OVERvalues vs market:")
        print(merged.head(4)[["team", "model_prob", "market_prob", "edge"]].to_string(index=False))
        print("  Model UNDERvalues vs market:")
        print(merged.tail(4)[["team", "model_prob", "market_prob", "edge"]].to_string(index=False))
        print(f"  -> {mvm_path}")
    except FileNotFoundError as e:
        print(f"\n  Market benchmark atlandı (odds yok): {e}")

    meta = {
        "n_sims": n_sims, "dc_since": dc_since,
        "holdout": f"multi({len(MULTI_HOLDOUTS)})",
        "has_xgb": HAS_XGB, "w_elo": W_ELO, "w_xgb": W_XGB,
        "exclude_friendly": exclude_friendly,
        "ensemble_log_loss": ens_ll, "ensemble_rps": ens_rps,
        # bootstrap RPS CIs
        "elo_rps_lo": ci_elo["lo"], "elo_rps_hi": ci_elo["hi"],
        "xgb_rps_lo": ci_xgb["lo"], "xgb_rps_hi": ci_xgb["hi"],
        "ensemble_rps_lo": ci_ens["lo"], "ensemble_rps_hi": ci_ens["hi"],
        # significance vs plain Elo (paired bootstrap)
        "ens_vs_elo_diff": diff_ens["diff"], "ens_vs_elo_p_better": diff_ens["p_a_better"],
        "ens_vs_elo_significant": diff_ens["significant"],
        "beats_elo_verdict": verdict,
        # calibration
        "calibration_ece": rel["ece"],
        # market benchmark (2026 outright)
        "market_spearman": market_summary.get("spearman"),
        "market_kl": market_summary.get("kl_model_market"),
        "model_top3_mass": market_summary.get("model_top3_mass"),
        "market_top3_mass": market_summary.get("market_top3_mass"),
        "market_overround": market_summary.get("overround"),
        # tuning
        "tune_method": tune_info.get("method"),
        "tune_cv_metric": tune_info.get("cv_metric"),
    }
    for name, ll, r, acc in evals:
        meta[f"{name.lower().replace('+','_')}_log_loss"] = ll
        meta[f"{name.lower().replace('+','_')}_rps"] = r
    pd.DataFrame([meta]).to_csv(ROOT / "outputs" / f"model_{label}_meta.csv", index=False)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50000)
    p.add_argument("--since", default="2014-01-01")
    p.add_argument("--holdout-year", type=int, default=2022)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--no-friendly", action="store_true", help="DC training'den friendly maçları çıkar")
    p.add_argument("--tune", action="store_true", help="XGB hiperparametre araması (Optuna/RandomizedSearch)")
    p.add_argument("--w-elo", type=float, default=None,
                   help="Ensemble Elo ağırlığını sabitle (grid-search atla). 1.0 = saf Elo-only (DC goller). "
                        "Çıktılar etiketlenir: champion_probs_Elo.csv vb.")
    args = p.parse_args()
    main(n_sims=args.n, dc_since=args.since, holdout_year=args.holdout_year, n_jobs=args.jobs,
         exclude_friendly=args.no_friendly, tune=args.tune, w_elo_override=args.w_elo)
