"""Bookmaker market benchmark — 2026 WC outright (champion) odds.

Pre-tournament dış ölçüt: modelin P_Champion dağılımını piyasanın ima ettiği
şampiyonluk olasılıklarıyla kıyasla. Kapanış çizgisini yenmek pratikte çok zor;
amaç piyasayı yakalamak (yakın olmak) + en büyük anlaşmazlıkları (edge) görmek.

Şampiyonluk piyasası n=1: gerçek not ancak kupa bitince (19 Tem 2026) verilir.
Bu modül turnuva öncesi BETİMLEYİCİdir (kim nerede ayrışıyor), kazanç/kayıp değil.

Veri: data/raw/odds/wc2026_outright_*.csv (team, decimal_odds, book, date).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ODDS_DIR = ROOT / "data" / "raw" / "odds"


def load_outright_odds(path: Path | None = None) -> pd.DataFrame:
    """En güncel outright odds CSV'sini yükle. Kolonlar: team, decimal_odds, ...

    path verilmezse ODDS_DIR içindeki en yeni wc2026_outright_*.csv seçilir.
    """
    if path is None:
        candidates = sorted(ODDS_DIR.glob("wc2026_outright_*.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"Outright odds CSV yok: {ODDS_DIR}/wc2026_outright_*.csv")
        path = candidates[-1]
    df = pd.read_csv(path)
    if "decimal_odds" not in df.columns or "team" not in df.columns:
        raise ValueError(f"{path}: 'team' ve 'decimal_odds' kolonları gerekli")
    df["team"] = df["team"].astype(str).str.strip()
    df["decimal_odds"] = df["decimal_odds"].astype(float)
    return df


def implied_probs(odds_df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Decimal odds → de-vigged ima edilen şampiyonluk olasılığı.

    raw_p = 1/decimal (vig dahil). 48 takımlık kitap aşırı-yuvarlama (overround)
    içerir → toplam 1'e normalize ederek vig'i orantısal dağıt.

    Döndürür: (DataFrame[team, market_prob], overround).
    overround = sum(raw_p) (1.0'dan ne kadar büyükse o kadar vig).
    """
    raw = 1.0 / odds_df["decimal_odds"].to_numpy()
    overround = float(raw.sum())
    out = odds_df[["team"]].copy()
    out["market_prob"] = raw / raw.sum()
    return out.sort_values("market_prob", ascending=False).reset_index(drop=True), overround


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(p || q) — nat. p=model, q=market."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank korelasyonu (scipy'siz)."""
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def compare_to_market(model_probs: pd.DataFrame,
                      odds_df: pd.DataFrame | None = None,
                      model_col: str = "P_Champion",
                      team_col: str = "team") -> tuple[pd.DataFrame, dict]:
    """Model şampiyonluk olasılıkları vs piyasa ima olasılıkları.

    model_probs: DataFrame, en az [team_col, model_col].
    odds_df: load_outright_odds() çıktısı (None → otomatik yükle).

    Döndürür: (merged DataFrame [team, model_prob, market_prob, edge, edge_ratio]
    azalan edge, summary dict {overround, spearman, kl_model_market,
    n_matched, n_model_only, n_market_only, top_overvalued, top_undervalued}).
    """
    if odds_df is None:
        odds_df = load_outright_odds()
    market, overround = implied_probs(odds_df)

    m = model_probs[[team_col, model_col]].rename(
        columns={team_col: "team", model_col: "model_prob"})
    m["team"] = m["team"].astype(str).str.strip()

    merged = m.merge(market, on="team", how="outer", indicator=True)
    n_matched = int((merged["_merge"] == "both").sum())
    n_model_only = int((merged["_merge"] == "left_only").sum())
    n_market_only = int((merged["_merge"] == "right_only").sum())

    merged["model_prob"] = merged["model_prob"].fillna(0.0)
    merged["market_prob"] = merged["market_prob"].fillna(0.0)
    merged["edge"] = merged["model_prob"] - merged["market_prob"]
    # edge_ratio: model piyasanın kaç katı (0 bölmeye karşı korumalı)
    merged["edge_ratio"] = merged["model_prob"] / merged["market_prob"].replace(0.0, np.nan)
    merged = merged.drop(columns="_merge").sort_values("edge", ascending=False).reset_index(drop=True)

    both = merged[(merged["model_prob"] > 0) & (merged["market_prob"] > 0)]
    summary = {
        "overround": overround,
        "spearman": _spearman(both["model_prob"].to_numpy(), both["market_prob"].to_numpy()),
        "kl_model_market": _kl(merged["model_prob"].to_numpy(), merged["market_prob"].to_numpy()),
        "n_matched": n_matched,
        "n_model_only": n_model_only,
        "n_market_only": n_market_only,
        "model_top3_mass": float(merged.nlargest(3, "model_prob")["model_prob"].sum()),
        "market_top3_mass": float(merged.nlargest(3, "market_prob")["market_prob"].sum()),
        "top_overvalued": merged.head(3)[["team", "edge"]].to_dict("records"),
        "top_undervalued": merged.tail(3)[["team", "edge"]].to_dict("records"),
    }
    return merged, summary
