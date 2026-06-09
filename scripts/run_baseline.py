"""S baseline: eloratings.net snapshot → 2026 turnuva sim → champion_probs_S.csv.

Custom Elo emekli. Artık doğrudan resmi snapshot kullanılıyor.
Kullanım:
    python scripts/run_baseline.py
    python scripts/run_baseline.py --snapshot 2024.tsv  # backtest
"""
from __future__ import annotations
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_groups_2026
from src.elo import EloRatings
from src.simulate import run_monte_carlo
from src.market import compare_to_market


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.strip().lower()


def main(n_sims: int = 50000, snapshot: str = "World.tsv") -> None:
    print(f"[1/3] Elo snapshot: {snapshot}")
    elo = EloRatings.from_snapshot(snapshot)
    top10 = elo.to_frame().head(10)
    print(top10.to_string(index=False))

    print("[2/3] 2026 grupları yükleniyor")
    fixed_groups = load_groups_2026()
    teams = [t for g in fixed_groups for t in g]
    missing = [t for t in teams if _norm(t) not in elo.ratings]
    if missing:
        print(f"  UYARI: snapshot'ta yok (initial=1500): {missing}")
    print(f"  {len(fixed_groups)} grup, {len(teams)} takım")

    print(f"[3/3] Monte Carlo (n={n_sims:,})")
    out = run_monte_carlo(elo, fixed_groups, n=n_sims)
    out_path = ROOT / "outputs" / "champion_probs_S.csv"
    out_path.parent.mkdir(exist_ok=True)
    out.to_csv(out_path, index=False)
    print("\nTop 15:")
    print(out.head(15).to_string(index=False))
    print(f"\n-> {out_path}")

    # --- Market benchmark (M+ ile aynı desen, src/market.py reuse) ---
    try:
        merged, summ = compare_to_market(out)
        mvm_path = ROOT / "outputs" / "model_vs_market_S.csv"
        merged.to_csv(mvm_path, index=False)
        print(f"\nMarket benchmark (overround={summ['overround']:.3f}):")
        print(f"  Spearman(model,market)={summ['spearman']:.3f}  KL(model||market)={summ['kl_model_market']:.3f}")
        print(f"  Concentration — model top-3={summ['model_top3_mass']:.3f}  vs market top-3={summ['market_top3_mass']:.3f}")
        print(f"  -> {mvm_path}")
    except FileNotFoundError as e:
        print(f"\n  Market benchmark atlandı (odds yok): {e}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--snapshot", default="World.tsv")
    args = p.parse_args()
    main(n_sims=args.n, snapshot=args.snapshot)
