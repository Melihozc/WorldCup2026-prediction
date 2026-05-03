"""Smoke test: World.tsv snapshot → 2026 grupları → MC sim. Sentetik veri yok."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.elo import EloRatings
from src.simulate import run_monte_carlo
from src.data import teams_2026, load_groups_2026


def main():
    try:
        fixed_groups = load_groups_2026()
        teams = [t for g in fixed_groups for t in g]
    except FileNotFoundError:
        teams = teams_2026()
        fixed_groups = [teams[i * 4:(i + 1) * 4] for i in range(12)]

    print(f"Snapshot Elo yükleniyor (World.tsv) — {len(teams)} takım, {len(fixed_groups)} grup")
    elo = EloRatings.from_snapshot()
    print("\nTop 10 Elo:")
    print(elo.to_frame().head(10).to_string(index=False))
    print("\nMC simülasyon (n=1000)...")
    out = run_monte_carlo(elo, fixed_groups, n=1000)
    print("\nTop 10 şampiyon olasılıkları:")
    print(out.head(10).to_string(index=False))
    top5_sum = out.head(5)["P_Champion"].sum()
    print(f"\nTop-5 toplam P_Champion = {top5_sum:.3f}")
    assert top5_sum > 0.30, "Sanity fail: top-5 olasılığı çok düşük"
    print("PASS")


if __name__ == "__main__":
    main()
