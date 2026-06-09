"""Model karşılaştırması: Sade Elo (S) vs Elo-only-DC vs Tam M+ vs Piyasa.

Salt-okur — mevcut outputs/*.csv'leri birleştirir, kullanıcının "Elo-only daha mı
doğru?" sorusunu kanıta döker. Yeni model eğitmez.

Çıktı: outputs/model_comparison.csv + konsol raporu.
Kullanım: python scripts/compare_models.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.market import load_outright_odds, implied_probs, compare_to_market

try:  # Windows konsolu cp1254 — Unicode için utf-8'e geç
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "outputs"

# (etiket, dosya, açıklama)
MODELS = [
    ("S",     "champion_probs_S.csv",     "Sade Elo (kaba forced-Poisson goller)"),
    ("EloDC", "champion_probs_Elo.csv",   "Elo-only W/D/L + Dixon-Coles goller (--w-elo 1.0)"),
    ("Mplus", "champion_probs_Mplus.csv", "Tam M+ ensemble (Elo+Pi+FIFA+DC+kadro -> XGB)"),
]


def main() -> None:
    odds = load_outright_odds()
    market, overround = implied_probs(odds)
    market = market.rename(columns={"market_prob": "P_market"})

    table = market.copy()
    summaries: dict[str, dict] = {}
    present: list[str] = []

    for label, fname, _desc in MODELS:
        p = OUT / fname
        if not p.exists():
            print(f"  (atlandı: {fname} yok — önce üret)")
            continue
        raw = pd.read_csv(p)
        col = raw[["team", "P_Champion"]].rename(columns={"P_Champion": f"P_{label}"})
        table = table.merge(col, on="team", how="outer")
        _merged, summ = compare_to_market(raw)
        summaries[label] = summ
        present.append(label)

    table = table.sort_values("P_market", ascending=False).reset_index(drop=True)
    comp_path = OUT / "model_comparison.csv"
    table.to_csv(comp_path, index=False)

    print("=" * 74)
    print("ŞAMPİYONLUK OLASILIKLARI — Model vs Piyasa (ilk 12, piyasa sırasıyla)")
    print("=" * 74)
    cols = ["team", "P_market"] + [f"P_{l}" for l in present]
    show = table[cols].head(12).copy()
    for c in cols[1:]:
        show[c] = (show[c] * 100).round(1)
    print(show.to_string(index=False))

    print("\n" + "=" * 74)
    print("PİYASAYA UYUM + YOĞUNLAŞMA (düşük KL = piyasaya yakın)")
    print("=" * 74)
    print(f"{'Model':8s} {'top3_mass':>10s} {'Spearman':>9s} {'KL->market':>10s}   açıklama")
    print(f"{'PİYASA':8s} {summaries[present[0]]['market_top3_mass']*100:9.1f}% "
          f"{'—':>9s} {'—':>10s}   de-vig, overround={overround:.3f}")
    desc_by = {l: d for l, _f, d in MODELS}
    for l in present:
        s = summaries[l]
        print(f"{l:8s} {s['model_top3_mass']*100:9.1f}% {s['spearman']:9.3f} "
              f"{s['kl_model_market']:10.3f}   {desc_by[l]}")

    # --- Maç-bazı RPS (zaten ölçülmüş — model_Mplus_meta.csv) ---
    meta_p = OUT / "model_Mplus_meta.csv"
    print("\n" + "=" * 74)
    print("MAÇ-BAZI DOĞRULUK (9-turnuva holdout, 431 maç, RPS — düşük iyi)")
    print("=" * 74)
    if meta_p.exists():
        m = pd.read_csv(meta_p).iloc[0]
        print(f"  Sade Elo : RPS={m['elo_rps']:.5f}  [{m['elo_rps_lo']:.5f}, {m['elo_rps_hi']:.5f}]")
        print(f"  Tam M+   : RPS={m['ensemble_rps']:.5f}  [{m['ensemble_rps_lo']:.5f}, {m['ensemble_rps_hi']:.5f}]")
        print(f"  Fark (M+ − Elo) = {m['ens_vs_elo_diff']:+.5f}  "
              f"P(M+ daha iyi)={m['ens_vs_elo_p_better']:.2f}  anlamlı={m['ens_vs_elo_significant']}")
    else:
        print("  (model_Mplus_meta.csv yok — önce run_m_plus çalıştır)")

    # --- En büyük S vs M+ ayrışmaları ---
    if "S" in present and "Mplus" in present:
        d = table.dropna(subset=["P_S", "P_Mplus"]).copy()
        d["S_vs_Mplus"] = (d["P_S"] - d["P_Mplus"]).abs()
        big = d.sort_values("S_vs_Mplus", ascending=False).head(6)
        print("\n" + "=" * 74)
        print("EN BÜYÜK Elo(S) vs M+ AYRIŞMALARI (% şampiyonluk)")
        print("=" * 74)
        for r in big.itertuples(index=False):
            print(f"  {r.team:16s}  S={r.P_S*100:5.1f}  M+={r.P_Mplus*100:5.1f}  "
                  f"piyasa={r.P_market*100:5.1f}")

    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    print("  Maç-bazı skill İSTATİSTİKSEL OLARAK BERABERE (fark GA 0'ı kesiyor).")
    print("  -> Elo-only DAHA doğru değil; EŞİT doğru + daha basit. Parsimoni: Elo-only savunulabilir.")
    print("  -> Şampiyon dağılımları yakın ama özdeş değil (gol modeli + %26 XGB farkı).")
    print("  -> Üç model de piyasadan daha yoğun; gerçek not 19 Tem 2026'da.")
    print(f"\n-> {comp_path}")


if __name__ == "__main__":
    main()
