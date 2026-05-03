"""Feature engineering — S baseline + M+ extensions."""
from __future__ import annotations
import pandas as pd


def recent_form(df: pd.DataFrame, team: str, before: pd.Timestamp,
                window: int = 10) -> dict:
    """Son N maçtan form ve gol farkı ortalaması (tarih öncesi)."""
    mask = (df["date"] < before) & (
        (df["home_team"] == team) | (df["away_team"] == team)
    )
    recent = df[mask].tail(window)
    if recent.empty:
        return {"form": 1.0, "gd_avg": 0.0, "n": 0}
    pts = 0
    gd = 0
    for r in recent.itertuples(index=False):
        if r.home_team == team:
            diff = r.home_score - r.away_score
        else:
            diff = r.away_score - r.home_score
        gd += diff
        if diff > 0:
            pts += 3
        elif diff == 0:
            pts += 1
    n = len(recent)
    return {"form": pts / n, "gd_avg": gd / n, "n": n}


def build_match_features(
    df: pd.DataFrame,
    matches: pd.DataFrame,
    elo,
    dc=None,
    pi=None,
    fifa_rank=None,
    xg_agg: dict | None = None,
) -> pd.DataFrame:
    """Her maç için feature seti.

    fifa_rank: dict[str, int] (sabit) | FIFARank (walk-forward) | None.
    dc: DixonColes nesnesi (opsiyonel) → attack_diff, defense_diff ekler.
    pi: PiRatings nesnesi (opsiyonel) → pi_diff ekler.
    xg_agg: dict[team] = {xg_per_match, xga_per_match, ...} (StatsBomb türevi).
        WALK-FORWARD UYARISI: xg_agg sabit dict — caller, holdout dönemi
        ÖNCESİ turnuvalardan oluşturmalı (leakage).
    """
    has_fr_obj = fifa_rank is not None and hasattr(fifa_rank, "points")

    rows = []
    for m in matches.itertuples(index=False):
        date = m.date
        h, a = m.home_team, m.away_team
        rh, ra = elo.get(h, date), elo.get(a, date)
        fh = recent_form(df, h, date)
        fa = recent_form(df, a, date)
        if has_fr_obj:
            pts_h = fifa_rank.points(h, date)
            pts_a = fifa_rank.points(a, date)
            # rank() is expensive per-row; use points-derived sort substitute (lower pts → higher rank #)
            rank_h = rank_a = 100  # constant; pts_diff carries ordering signal
        else:
            d = fifa_rank or {}
            rank_h = d.get(h, 100)
            rank_a = d.get(a, 100)
            pts_h = pts_a = 0.0
        row: dict = {
            "date": date,
            "home": h, "away": a,
            "elo_diff": rh - ra,
            "rank_diff": rank_a - rank_h,
            "fifa_pts_diff": pts_h - pts_a,
            "home_advantage": int(not getattr(m, "neutral", True)),
            "form_diff": fh["form"] - fa["form"],
            "gd_avg_diff": fh["gd_avg"] - fa["gd_avg"],
        }
        if dc is not None:
            row["attack_diff"] = dc.attack.get(h, 0.0) - dc.attack.get(a, 0.0)
            row["defense_diff"] = dc.defense.get(a, 0.0) - dc.defense.get(h, 0.0)
        if pi is not None:
            row["pi_diff"] = pi.diff(h, a)
        if xg_agg is not None:
            xh = xg_agg.get(h, {"xg_per_match": 0.0, "xga_per_match": 0.0})
            xa = xg_agg.get(a, {"xg_per_match": 0.0, "xga_per_match": 0.0})
            row["xg_for_diff"] = xh["xg_per_match"] - xa["xg_per_match"]
            row["xg_against_diff"] = xa["xga_per_match"] - xh["xga_per_match"]
        rows.append(row)
    return pd.DataFrame(rows)
