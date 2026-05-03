"""M scale: Elo + Dixon-Coles + ensemble → 2026 turnuva simülasyonu.

Çıktı: outputs/champion_probs_M.csv + ensemble ağırlıkları.
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
from src.poisson import DixonColes
from src.simulate import run_monte_carlo_cb
from src.backtest import (
    split_by_tournaments, evaluate_predictor, outcomes_from_df,
)
from src.eval import log_loss, rps

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


def find_ensemble_weight(p_elo: np.ndarray, p_dc: np.ndarray,
                         y: np.ndarray, grid: int = 21) -> tuple[float, float]:
    """0..1 grid ile w*Elo + (1-w)*DC için min RPS bul."""
    best_w, best_rps = 0.5, 1e9
    for w in np.linspace(0, 1, grid):
        p = w * p_elo + (1 - w) * p_dc
        p = p / p.sum(axis=1, keepdims=True)
        r = rps(p, y)
        if r < best_rps:
            best_w, best_rps = w, r
    return float(best_w), float(best_rps)


def main(n_sims: int = 50000, dc_since: str = "2014-01-01",
         exclude_friendly: bool = False) -> None:
    print(f"[1/6] Veri yükleniyor")
    df = load_results()
    df = add_outcome(df)
    print(f"  {len(df):,} maç")

    print(f"[2/6] Multi-holdout split ({len(MULTI_HOLDOUTS)} turnuva)")
    train, test = split_by_tournaments(df, MULTI_HOLDOUTS)
    print(f"  train={len(train):,}, test={len(test)}")

    print(f"[3/6] Elo: walk-forward yearly snapshots (HistoricalElo)")
    elo = HistoricalElo()

    friendly_tag = ", no-friendly" if exclude_friendly else ""
    print(f"[4/6] Dixon-Coles fit (since {dc_since}{friendly_tag})")
    dc_mask = train["date"] >= pd.Timestamp(dc_since)
    if exclude_friendly:
        dc_mask &= train["tournament"] != "Friendly"
    dc_train = train[dc_mask]
    print(f"  DC training maçları: {len(dc_train):,}")
    dc = DixonColes().fit(dc_train)

    # holdout değerlendir + ensemble ağırlığı bul
    print(f"[5/6] Backtest + ensemble tuning (RPS, multi-holdout)")
    elo_eval = evaluate_predictor(elo.predict_proba, test, date_aware=True)
    dc_eval = evaluate_predictor(dc.predict_proba, test)
    y = outcomes_from_df(test)
    print(f"  Elo: log_loss={elo_eval['log_loss']:.4f} RPS={elo_eval['rps']:.4f} "
          f"acc={elo_eval['accuracy']:.3f}")
    print(f"  DC : log_loss={dc_eval['log_loss']:.4f} RPS={dc_eval['rps']:.4f} "
          f"acc={dc_eval['accuracy']:.3f}")
    w, best_rps = find_ensemble_weight(elo_eval["probs"], dc_eval["probs"], y)
    print(f"  Ensemble: w_elo={w:.2f}, w_dc={1-w:.2f}, RPS={best_rps:.4f}")

    # 2026 grupları + takımları
    fixed_groups = load_groups_2026()
    teams = [t for g in fixed_groups for t in g]
    from src.elo import _norm
    missing_elo = [t for t in teams if _norm(t) not in elo.current.ratings]
    missing_dc = [t for t in teams if t not in dc.attack]
    if missing_elo:
        print(f"  UYARI Elo eksik: {missing_elo}")
    if missing_dc:
        print(f"  UYARI DC eksik (default 0,0): {missing_dc}")

    # MC için: production = current snapshot (World.tsv)
    print(f"[6/6] Production: World.tsv current Elo + MC sim n={n_sims:,}")
    elo_full = elo.current  # current snapshot
    dc_full_mask = df["date"] >= pd.Timestamp(dc_since)
    if exclude_friendly:
        dc_full_mask &= df["tournament"] != "Friendly"
    dc_full = DixonColes().fit(df[dc_full_mask])

    def proba_fn(h: str, a: str, neutral: bool) -> tuple[float, float, float]:
        p_e = np.array(elo_full.predict_proba(h, a, neutral))
        p_d = np.array(dc_full.predict_proba(h, a, neutral))
        p = w * p_e + (1 - w) * p_d
        p /= p.sum()
        return tuple(p)

    def goals_fn(rng: np.random.Generator, h: str, a: str, neutral: bool):
        return dc_full.sample_score(rng, h, a, neutral)

    out = run_monte_carlo_cb(proba_fn, goals_fn, fixed_groups, n=n_sims, seed=42)
    out_path = ROOT / "outputs" / "champion_probs_M.csv"
    out.to_csv(out_path, index=False)
    print(f"\nTop 15:")
    print(out.head(15).to_string(index=False))
    print(f"\n-> {out_path}")

    # ensemble metadata
    meta = pd.DataFrame([{
        "n_sims": n_sims, "dc_since": dc_since,
        "holdout": f"multi({len(MULTI_HOLDOUTS)})",
        "exclude_friendly": exclude_friendly,
        "w_elo": w, "w_dc": 1 - w,
        "elo_log_loss": elo_eval["log_loss"],
        "dc_log_loss": dc_eval["log_loss"],
        "ensemble_rps": best_rps,
        "elo_rps": elo_eval["rps"], "dc_rps": dc_eval["rps"],
    }])
    meta.to_csv(ROOT / "outputs" / "model_M_meta.csv", index=False)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50000)
    p.add_argument("--since", default="2014-01-01")
    p.add_argument("--no-friendly", action="store_true",
                   help="DC training'den friendly maçları çıkar")
    args = p.parse_args()
    main(n_sims=args.n, dc_since=args.since, exclude_friendly=args.no_friendly)
