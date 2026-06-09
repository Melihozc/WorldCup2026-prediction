"""Kadro gücü öznitelikleri — Transfermarkt market value zaman serisinden.

Ulusal takım kadro değeri, Elo'dan bağımsız bir yetenek sinyali olabilir
(yükselen genç oyuncuları sonuç-tabanlı Elo'dan önce yakalar). Walk-forward:
her maç için maç tarihinden ÖNCEKİ yıllık snapshot kullanılır → sızıntı yok.

VERİ + SINIRLAMALAR:
- Oyuncu → ülke ataması BİRİNCİL vatandaşlıkla yapılır (citizenship ilk token).
  Transfermarkt'ta team_id → ülke adı haritası yok (team_details sadece kulüp).
  Çifte vatandaşlar birincil vatandaşlığa atanır → diaspora ağırlıklı küçük
  takımlar (DR Kongo, Fildişi Sahili) düşük değerlenir. Favoriler (İspanya,
  Fransa, Brezilya...) doğru kapsanır.
- Değer = ülkenin o snapshot'taki en değerli `top_n` oyuncusunun toplamı.
- Yaş = aynı top_n oyuncunun snapshot tarihindeki ortalama yaşı.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TM_DL = ROOT / "data" / "raw" / "transfermarkt_dl"

# Transfermarkt birincil-vatandaşlık adı -> results.csv kanonik adı.
# Çoğu aynı; sadece farklı yazımlar map'lenir.
_CITIZENSHIP_ALIASES: dict[str, str] = {
    "Türkiye": "Turkey",
    "Korea, South": "South Korea",
    "Korea, North": "North Korea",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Cote d'Ivoire": "Ivory Coast",
    "Curacao": "Curaçao",
    "USA": "United States",
    "Republic of Ireland": "Republic of Ireland",
}


def _canon_country(primary: str) -> str:
    p = primary.strip()
    return _CITIZENSHIP_ALIASES.get(p, p)


class SquadStrength:
    """Yıllık snapshot tablosu: (country, snapshot_date) -> (top_n değer toplamı, ort yaş).

    .features(team, date) → (log10_value, mean_age). Bilinmiyorsa (nan, nan).
    """

    def __init__(self, table: pd.DataFrame):
        # table: country, snap_date, squad_value, mean_age, n_players (snap_date artan)
        self._table = table
        self._by_country = {
            c: g.sort_values("snap_date").reset_index(drop=True)
            for c, g in table.groupby("country")
        }

    @classmethod
    def build(cls, top_n: int = 23, start_year: int = 2008,
              end_year: int = 2027, min_players: int = 8) -> "SquadStrength":
        prof = pd.read_csv(
            TM_DL / "player_profiles.csv",
            usecols=["player_id", "date_of_birth", "citizenship"],
            encoding="utf-8",
        )
        prof = prof.dropna(subset=["citizenship"])
        prof["country"] = (
            prof["citizenship"].astype(str).str.split("  ").str[0].map(_canon_country)
        )
        prof["dob"] = pd.to_datetime(prof["date_of_birth"], errors="coerce")
        prof = prof[["player_id", "country", "dob"]]

        mv = pd.read_csv(TM_DL / "player_market_value.csv")
        mv["date"] = pd.to_datetime(mv["date_unix"], errors="coerce")
        mv = mv.dropna(subset=["date", "value"])
        mv = mv[mv["value"] > 0].sort_values("date")

        snap_dates = [pd.Timestamp(f"{y}-01-01") for y in range(start_year, end_year)]

        # Her snapshot için her oyuncunun o tarihten ÖNCEKİ son değeri (merge_asof).
        rows = []
        mv_sorted = mv[["player_id", "date", "value"]].sort_values("date")
        for s in snap_dates:
            grid = prof[["player_id", "country", "dob"]].copy()
            # Her oyuncunun snapshot tarihinden ÖNCEKİ son market value'su (walk-forward).
            valid = mv_sorted[mv_sorted["date"] < s]
            last_val = valid.groupby("player_id")["value"].last()
            g = grid.set_index("player_id")
            g["value"] = last_val
            g = g.dropna(subset=["value"])
            if g.empty:
                continue
            g["age"] = (s - g["dob"]).dt.days / 365.25
            # top_n per country
            g = g.sort_values("value", ascending=False)
            agg = g.groupby("country").apply(
                lambda d: pd.Series({
                    "squad_value": d["value"].head(top_n).sum(),
                    "mean_age": d["age"].head(top_n).mean(),
                    "n_players": min(len(d), top_n),
                }),
                include_groups=False,
            ).reset_index()
            agg["snap_date"] = s
            rows.append(agg)

        table = pd.concat(rows, ignore_index=True)
        table = table[table["n_players"] >= min_players].reset_index(drop=True)
        return cls(table)

    def features(self, team: str, date: pd.Timestamp) -> tuple[float, float]:
        g = self._by_country.get(team)
        if g is None:
            return (np.nan, np.nan)
        prior = g[g["snap_date"] < pd.Timestamp(date)]
        if prior.empty:
            return (np.nan, np.nan)
        row = prior.iloc[-1]
        val = float(row["squad_value"])
        log_val = float(np.log10(val)) if val > 0 else np.nan
        return (log_val, float(row["mean_age"]))

    def diff(self, home: str, away: str, date: pd.Timestamp) -> tuple[float, float]:
        vh, ah = self.features(home, date)
        va, aa = self.features(away, date)
        value_diff = (vh - va) if (np.isfinite(vh) and np.isfinite(va)) else 0.0
        age_diff = (ah - aa) if (np.isfinite(ah) and np.isfinite(aa)) else 0.0
        return (value_diff, age_diff)
