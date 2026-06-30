"""İnteraktif head-to-head tahmin — Elo + Dixon-Coles.

Elo snapshot + DC fit yükler, terminalde iki takım arası maç tahmini verir:
W/D/L (Elo+DC ensemble), beklenen gol (DC λ), en olası skor, top-5 skor dağılımı.

Kullanım:
    python scripts/oracle.py                       # interaktif döngü
    python scripts/oracle.py "Brazil vs Japan"     # tek tahmin, çık
    python scripts/oracle.py --since 2014-01-01 --no-friendly

Komutlar (döngü içinde):
    Brazil vs Japan   → maç tahmini   (vs | v | - | x | , ayraçları)
    teams             → 2026 takımları listele
    quit              → çık
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from src.data import load_results, load_groups_2026
from src.elo import EloRatings
from src.poisson import DixonColes

# M ölçeğinde holdout'ta tune edilmiş ensemble ağırlığı (outputs/model_M_meta.csv)
W_ELO = 0.65
W_DC = 0.35


def head_to_head(elo: EloRatings, dc: DixonColes, a: str, b: str,
                 neutral: bool = True) -> dict:
    """İki takım için W/D/L (ensemble), xG (DC λ), skor dağılımı."""
    p_elo = np.array(elo.predict_proba(a, b, neutral=neutral))
    p_dc = np.array(dc.predict_proba(a, b, neutral=neutral))
    p = W_ELO * p_elo + W_DC * p_dc
    p = p / p.sum()

    lh, la = dc.lambdas(a, b, neutral=neutral)
    m = dc.score_matrix(a, b, neutral=neutral)
    # en olası 5 skor
    flat = [(int(i), int(j), float(m[i, j]))
            for i in range(m.shape[0]) for j in range(m.shape[1])]
    flat.sort(key=lambda t: t[2], reverse=True)
    top5 = flat[:5]
    return {
        "p_w": p[0], "p_d": p[1], "p_l": p[2],
        "xg_a": lh, "xg_b": la,
        "best_score": f"{top5[0][0]}-{top5[0][1]}",
        "top5": top5,
        "elo_a": elo.get(a), "elo_b": elo.get(b),
    }


def _resolve(teams: list[str], query: str) -> str | None:
    """Takım adını gevşek eşle (tam → prefix → substring)."""
    q = query.strip().lower()
    for t in teams:
        if t.lower() == q:
            return t
    for t in teams:
        if t.lower().startswith(q):
            return t
    for t in teams:
        if q in t.lower():
            return t
    return None


def print_match(elo, dc, teams, a_in: str, b_in: str) -> None:
    a = _resolve(teams, a_in)
    b = _resolve(teams, b_in)
    if a is None:
        print(f"  ! Bilinmeyen takım: '{a_in}'. 'teams' yaz.")
        return
    if b is None:
        print(f"  ! Bilinmeyen takım: '{b_in}'. 'teams' yaz.")
        return
    if a == b:
        print("  ! İki farklı takım seç.")
        return
    r = head_to_head(elo, dc, a, b)
    print()
    print(f"  {a} (Elo {r['elo_a']:.0f})  vs  {b} (Elo {r['elo_b']:.0f})   [nötr saha]")
    print("  " + "-" * 50)
    print(f"  {a} galip : {r['p_w'] * 100:5.1f}%")
    print(f"  Beraberlik : {r['p_d'] * 100:5.1f}%")
    print(f"  {b} galip : {r['p_l'] * 100:5.1f}%")
    print(f"  Beklenen gol : {a} {r['xg_a']:.2f} – {r['xg_b']:.2f} {b}")
    print(f"  En olası skor : {r['best_score']}")
    print("  Top-5 skor:")
    for ga, gb, prob in r["top5"]:
        print(f"     {ga}-{gb}  {prob * 100:4.1f}%")
    print()


def build(since: str, exclude_friendly: bool):
    print("[1/2] Elo snapshot yükleniyor")
    elo = EloRatings.from_snapshot()
    print(f"[2/2] Dixon-Coles fit (since {since}"
          f"{', no-friendly' if exclude_friendly else ''})")
    df = load_results()
    mask = df["date"] >= pd.Timestamp(since)
    if exclude_friendly:
        mask &= df["tournament"] != "Friendly"
    dc = DixonColes().fit(df[mask])
    teams = [t for g in load_groups_2026() for t in g]
    return elo, dc, teams


def interactive_loop(elo, dc, teams) -> None:
    print("\nHead-to-head tahmin. İki takım yaz, örn:  Brazil vs Japan")
    print("Komutlar:  teams | quit")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nÇıkış.")
            return
        if not line:
            continue
        low = line.lower()
        if low in ("quit", "exit", "q", "çık"):
            print("Çıkış.")
            return
        if low == "teams":
            for t in teams:
                print(f"  {t}")
            continue
        sep = next((s for s in (" vs ", " v ", " - ", " x ", ",") if s in low), None)
        if sep is None:
            print("  ! Format: <takım> vs <takım>   (ya da 'teams', 'quit')")
            continue
        a_part, b_part = line.split(sep, 1)
        print_match(elo, dc, teams, a_part.strip(), b_part.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description="Head-to-head oracle (Elo + DC)")
    ap.add_argument("match", nargs="?", help="tek seferlik 'A vs B'; boşsa döngü")
    ap.add_argument("--since", default="2014-01-01", help="DC fit başlangıç tarihi")
    ap.add_argument("--no-friendly", action="store_true", dest="no_friendly")
    args = ap.parse_args()

    elo, dc, teams = build(args.since, args.no_friendly)

    if args.match:
        sep = next((s for s in (" vs ", " v ", " - ", " x ", ",")
                    if s in args.match.lower()), None)
        if sep is None:
            print("  ! Format: 'A vs B'")
            return
        a_part, b_part = args.match.split(sep, 1)
        print_match(elo, dc, teams, a_part.strip(), b_part.strip())
        return

    interactive_loop(elo, dc, teams)


if __name__ == "__main__":
    main()
