"""Consensus-Hybrid (C) runner.

Pipeline: market consensus → inverse-sim market ability → blend with historical
(Dixon-Coles) + squad ability → temperature-calibrate to market → MC the 2026
bracket → outputs/champion_probs_Consensus.csv + meta + abilities.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_results, load_groups_2026  # noqa: E402
from src.poisson import DixonColes  # noqa: E402
from src.squad import SquadStrength  # noqa: E402
from src import consensus  # noqa: E402
from src.market import compare_to_market  # noqa: E402

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

    # load_groups_2026() already returns 12 lists (A..L), each 4 teams
    fixed_groups = load_groups_2026()
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
    dc = DixonColes().fit(hist)
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
        "infer_final_kl": info.get("best_kl", info.get("final_kl")),
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
