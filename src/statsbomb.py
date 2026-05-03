"""StatsBomb open-data lazy fetcher.

Match indexes pre-downloaded in data/raw/statsbomb/matches/<comp>/<season>.json.
Event files lazy fetch from GitHub raw — match_id bazında.

Kullanım:
    sb = StatsBombFetcher()
    matches = sb.matches(comp_id=43, season_id=106)  # WC2022 64 matches
    xg_h, xg_a = sb.match_xg(matches[0]['match_id'])

Cache: data/raw/statsbomb/events/<match_id>.json (download once).

Tam xG aggregate per team için:
    df = sb.team_xg_table([(43,106),(55,282)])  # WC2022 + Euro2024
"""
from __future__ import annotations
from pathlib import Path
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SB_DIR = ROOT / "data" / "raw" / "statsbomb"
MATCHES_DIR = SB_DIR / "matches"
EVENTS_DIR = SB_DIR / "events"
RAW_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


# resmi liste — open-data dahilindeki international competitions
INTL_COMPS = [
    (43, 106, "FIFA World Cup 2022"),
    (43, 3,   "FIFA World Cup 2018"),
    (55, 282, "UEFA Euro 2024"),
    (55, 43,  "UEFA Euro 2020"),
    (223, 282, "Copa America 2024"),
    (1267, 107, "African Cup of Nations 2023"),
]


def _load_json(path: Path) -> list | dict:
    return json.loads(path.read_text(encoding="utf-8"))


class StatsBombFetcher:
    def __init__(self):
        EVENTS_DIR.mkdir(parents=True, exist_ok=True)

    def matches(self, comp_id: int, season_id: int) -> list[dict]:
        path = MATCHES_DIR / str(comp_id) / f"{season_id}.json"
        if not path.exists():
            url = f"{RAW_BASE}/matches/{comp_id}/{season_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, path)
        return _load_json(path)

    def events(self, match_id: int) -> list[dict]:
        path = EVENTS_DIR / f"{match_id}.json"
        if not path.exists():
            url = f"{RAW_BASE}/events/{match_id}.json"
            urllib.request.urlretrieve(url, path)
        return _load_json(path)

    def match_xg(self, match_id: int) -> tuple[float, float]:
        """Return (home_xg, away_xg) summed from shot events."""
        ev = self.events(match_id)
        # match_id JSON yok, takım adları event içinde — match index'ten al
        # bunun yerine shot.statsbomb_xg + team.name kullan, sonra match index'le eşleştir
        team_xg: dict[str, float] = {}
        for e in ev:
            if e.get("type", {}).get("name") != "Shot":
                continue
            xg = e.get("shot", {}).get("statsbomb_xg")
            if xg is None:
                continue
            t = e.get("team", {}).get("name", "?")
            team_xg[t] = team_xg.get(t, 0.0) + xg
        # caller match index ile home/away ile eşleştirir
        return team_xg

    def team_xg_table(self, comp_seasons: list[tuple[int, int]] | None = None):
        """Build per-team total xG over given comps. Triggers full event download."""
        import pandas as pd
        comp_seasons = comp_seasons or [(c, s) for c, s, _ in INTL_COMPS[:4]]
        rows = []
        for comp, season in comp_seasons:
            for m in self.matches(comp, season):
                team_xg = self.match_xg(m["match_id"])
                h, a = m["home_team"]["home_team_name"], m["away_team"]["away_team_name"]
                rows.append({
                    "match_id": m["match_id"],
                    "date": m["match_date"],
                    "comp": comp, "season": season,
                    "home": h, "away": a,
                    "home_score": m["home_score"], "away_score": m["away_score"],
                    "home_xg": team_xg.get(h, 0.0),
                    "away_xg": team_xg.get(a, 0.0),
                })
        return pd.DataFrame(rows)

    def team_aggregate(self, comp_seasons: list[tuple[int, int]] | None = None):
        """Per-team xG, xGA, n_matches over comps. Returns dict[team] = {xg, xga, n, xg_per_match, xga_per_match}."""
        df = self.team_xg_table(comp_seasons)
        agg: dict = {}
        for r in df.itertuples(index=False):
            for team, xg_for, xg_against in [
                (r.home, r.home_xg, r.away_xg),
                (r.away, r.away_xg, r.home_xg),
            ]:
                a = agg.setdefault(team, {"xg": 0.0, "xga": 0.0, "n": 0})
                a["xg"] += xg_for
                a["xga"] += xg_against
                a["n"] += 1
        for team, a in agg.items():
            a["xg_per_match"] = a["xg"] / a["n"] if a["n"] else 0.0
            a["xga_per_match"] = a["xga"] / a["n"] if a["n"] else 0.0
        return agg
