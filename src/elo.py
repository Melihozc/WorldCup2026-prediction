"""eloratings.net resmi Elo loader.

Custom Elo (eski) emekliye alındı — bugları (Euro K=55, integer rounding eksik,
Nations League uydurma K) vardı. Artık doğrudan eloratings.net snapshot'larını
kullanıyoruz: `data/raw/eloratings/<year>.tsv` (yıl sonu) veya `World.tsv` (current).

İsim eşleme: `data/raw/eloratings/en.teams.tsv` ISO benzeri kod → İngilizce ad
(+ kısa formlar). Bazı takımlar için CSV'lerimizdeki isimle birebir tutmaz
(ör. "Czech Republic" vs "Czechia"). Synonym mapping bunu çözer.

predict_proba: Elo farkı + home advantage → 3-sınıf dağılım. Beraberlik
heuristiği kalibre değil ama eloratings.net rating üretir, tahmin yapmaz —
M+ ensemble'da XGBoost zaten bunu öğreniyor, S için yeterli.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import unicodedata
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ELO_DIR = ROOT / "data" / "raw" / "eloratings"

HOME_ADV = 100  # eloratings.net resmi: +100 home

# CSV'lerimizdeki isimle eloratings ad farklılıkları
_SYNONYMS = {
    "czech republic": "czechia",
    "usa": "united states",
    "ivory coast": "ivory coast",       # zaten match
    "south korea": "south korea",
    "north korea": "korea dpr",
}


def _norm(s: str) -> str:
    """Lowercase + accent strip + trim."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.strip().lower()


def _load_team_aliases(path: Path) -> dict:
    """en.teams.tsv → {normalized_name: code}.
    Format: CODE\tFullName[\tShortName1\t...]  (tab separated, variable cols).
    """
    code_by_name: dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or not parts[0]:
                continue
            code = parts[0]
            for name in parts[1:]:
                if name:
                    code_by_name[_norm(name)] = code
    return code_by_name


def _load_ratings_tsv(path: Path) -> dict:
    """<year>.tsv or World.tsv → {country_code: elo_rating}.
    Column 0=rank, col 1=tie?, col 2=code, col 3=current Elo.
    """
    out: dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            code = parts[2]
            try:
                rating = float(parts[3])
            except ValueError:
                continue
            out[code] = rating
    return out


@dataclass
class EloRatings:
    """Resmi eloratings.net snapshot tabanlı rating store.

    Kullanım:
        elo = EloRatings.from_snapshot()                   # World.tsv (current)
        elo = EloRatings.from_snapshot("2021.tsv")         # backtest için
        p_w, p_d, p_l = elo.predict_proba("Spain", "France", neutral=True)
    """
    initial: float = 1500.0
    ratings: dict = field(default_factory=dict)   # {team_name_normalized: elo}
    _name_orig: dict = field(default_factory=dict)  # {normalized: original_input_name}

    @classmethod
    def from_snapshot(cls, snapshot: str = "World.tsv",
                       teams_file: str = "en.teams.tsv") -> "EloRatings":
        snap_path = ELO_DIR / snapshot
        teams_path = ELO_DIR / teams_file
        if not snap_path.exists():
            raise FileNotFoundError(
                f"Elo snapshot bulunamadı: {snap_path}. "
                f"data/raw/eloratings/ altına `{snapshot}` indir.")
        if not teams_path.exists():
            raise FileNotFoundError(f"Team alias dosyası bulunamadı: {teams_path}")
        code_to_elo = _load_ratings_tsv(snap_path)
        alias_to_code = _load_team_aliases(teams_path)
        # synonym'ları da ekle
        for src, target in _SYNONYMS.items():
            if target in alias_to_code:
                alias_to_code[src] = alias_to_code[target]
        ratings: dict = {}
        for alias_norm, code in alias_to_code.items():
            if code in code_to_elo:
                ratings[alias_norm] = code_to_elo[code]
        return cls(ratings=ratings)

    def get(self, team: str, date=None) -> float:
        # date arg ignored — for API symmetry with HistoricalElo
        n = _norm(team)
        if n in _SYNONYMS:
            n = _SYNONYMS[n]
        return self.ratings.get(n, self.initial)

    def predict_proba(
        self, home: str, away: str, neutral: bool = True, date=None,
    ) -> tuple:
        """3-class P(home_win, draw, away_win).
        Beraberlik heuristic: eloratings.net rating üreticisi, tahmin değil — basit kalibre.
        """
        rh, ra = self.get(home), self.get(away)
        diff = rh - ra + (0 if neutral else HOME_ADV)
        p_home_or_win = 1.0 / (1.0 + 10 ** (-diff / 400))
        p_away_or_loss = 1.0 - p_home_or_win
        p_draw = 0.28 * (2.71828 ** (-0.0017 * abs(diff)))
        scale_w = p_home_or_win / (p_home_or_win + p_away_or_loss)
        scale_l = p_away_or_loss / (p_home_or_win + p_away_or_loss)
        p_w = max(p_home_or_win - p_draw * scale_w, 1e-6)
        p_l = max(p_away_or_loss - p_draw * scale_l, 1e-6)
        s = p_w + p_draw + p_l
        return p_w / s, p_draw / s, p_l / s

    def to_frame(self) -> pd.DataFrame:
        return (
            pd.DataFrame(
                [{"team": t, "elo": r} for t, r in self.ratings.items()]
            )
            .sort_values("elo", ascending=False)
            .reset_index(drop=True)
        )

    # --- backward-compat: eski API'yi koru ---
    def fit(self, matches: pd.DataFrame) -> "EloRatings":
        """LEGACY no-op. Yeni kod: EloRatings.from_snapshot() kullan.
        Eski scripts kırılmasın diye duruyor — match data ignore edilir.
        """
        if not self.ratings:
            other = EloRatings.from_snapshot()
            self.ratings = other.ratings
        return self


class HistoricalElo:
    """Walk-forward Elo: maç tarihine göre uygun yıl snapshot'ı döndürür.

    Backtest + XGB feature build için time-aware Elo. Bir 2022-XX maçında
    `get(team, date)` → 2021.tsv ratings (önceki yıl sonu).

    Production sim için `current` (World.tsv) kullan — predict_proba'da
    date=None geçilirse current alınır.
    """

    def __init__(self, current_snapshot: str = "World.tsv"):
        self.current = EloRatings.from_snapshot(current_snapshot)
        self._by_year: dict[int, EloRatings] = {}
        self._available = sorted(
            int(p.stem) for p in ELO_DIR.glob("[0-9][0-9][0-9][0-9].tsv")
        )

    def at(self, year: int) -> EloRatings:
        if not self._available:
            return self.current
        # clamp
        if year < self._available[0]:
            year = self._available[0]
        elif year > self._available[-1]:
            return self.current
        if year not in self._by_year:
            self._by_year[year] = EloRatings.from_snapshot(f"{year}.tsv")
        return self._by_year[year]

    def _snap_for(self, date) -> EloRatings:
        if date is None:
            return self.current
        ts = pd.Timestamp(date)
        # use end-of-prev-year snapshot
        return self.at(ts.year - 1)

    def get(self, team: str, date=None) -> float:
        return self._snap_for(date).get(team)

    def predict_proba(
        self, home: str, away: str, neutral: bool = True, date=None,
    ) -> tuple:
        return self._snap_for(date).predict_proba(home, away, neutral)

    @property
    def ratings(self) -> dict:
        return self.current.ratings
