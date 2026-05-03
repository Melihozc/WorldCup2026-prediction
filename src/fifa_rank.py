"""FIFA world ranking history loader (1992-2024).

Kaynak: Dato-Futbol/fifa-ranking GitHub mirror (FIFA resmi siteden scrape).
data/raw/fifa_ranking/ranking_fifa_historical.csv

Walk-forward kullanım: maç tarihinden ÖNCE en yakın yayın (publication date)
takım puanını döndür. Backtest leakage'sız.
"""
from __future__ import annotations
from pathlib import Path
import unicodedata
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RANK_CSV = ROOT / "data" / "raw" / "fifa_ranking" / "ranking_fifa_historical.csv"

# CSV team adı vs kullandığımız ad farklılıkları
_NAME_MAP = {
    "ir iran": "iran",
    "korea republic": "south korea",
    "korea dpr": "north korea",
    "usa": "united states",
    "czechia": "czech republic",
    "côte d'ivoire": "ivory coast",
    "cote d'ivoire": "ivory coast",
    "cabo verde": "cape verde",
}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.strip().lower()
    return _NAME_MAP.get(s, s)


class FIFARank:
    """Walk-forward FIFA rank lookup by (team, date)."""

    def __init__(self, df: pd.DataFrame):
        self.df = df  # team_norm, date, total_points
        # her takım için tarih sıralı (team_norm: sorted DataFrame)
        self._by_team = {
            t: g.sort_values("date").reset_index(drop=True)
            for t, g in df.groupby("team_norm", sort=False)
        }
        self.latest_date = df["date"].max()

    @classmethod
    def load(cls, path: Path = RANK_CSV) -> "FIFARank":
        if not path.exists():
            raise FileNotFoundError(f"FIFA rank CSV bulunamadı: {path}")
        df = pd.read_csv(path, parse_dates=["date"])
        df = df.dropna(subset=["total_points"])
        df["total_points"] = pd.to_numeric(df["total_points"], errors="coerce")
        df = df.dropna(subset=["total_points"])
        # "Eritrea (unranked)" gibi satırları temizle
        df["team"] = df["team"].str.replace(r"\s*\(unranked\)$", "", regex=True)
        df["team_norm"] = df["team"].map(_norm)
        return cls(df[["team_norm", "date", "total_points"]].reset_index(drop=True))

    def points(self, team: str, date=None) -> float:
        """En son yayın <= date için total_points. date=None → en güncel."""
        n = _norm(team)
        g = self._by_team.get(n)
        if g is None or g.empty:
            return 0.0
        if date is None:
            return float(g["total_points"].iloc[-1])
        ts = pd.Timestamp(date)
        idx = g["date"].searchsorted(ts, side="right") - 1
        if idx < 0:
            return 0.0
        return float(g["total_points"].iloc[idx])

    def rank(self, team: str, date=None) -> int:
        """Takımın yayın tarihindeki sıralaması (1 = top). Bulunamazsa 999."""
        ts = pd.Timestamp(date) if date is not None else self.latest_date
        # en son yayın <= ts
        publications = self.df["date"].sort_values().unique()
        idx = pd.Timestamp(ts).to_datetime64()
        # find latest publication <= ts
        valid = [p for p in publications if p <= idx]
        if not valid:
            return 999
        pub = valid[-1]
        snapshot = self.df[self.df["date"] == pub].sort_values("total_points", ascending=False)
        snapshot = snapshot.reset_index(drop=True)
        n = _norm(team)
        match = snapshot[snapshot["team_norm"] == n]
        if match.empty:
            return 999
        return int(match.index[0]) + 1
