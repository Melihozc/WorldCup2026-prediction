"""Veri yükleme + temizleme."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
MARTJ42 = RAW / "International football results"
TRANSFERMARKT = RAW / "transfermarkt"


def _find_results_csv() -> Path:
    """results.csv ara: önce subfolder, sonra raw kökü."""
    for candidate in [MARTJ42 / "results.csv", RAW / "results.csv"]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"results.csv bulunamadı: {MARTJ42 / 'results.csv'} veya {RAW / 'results.csv'}"
    )


def load_results(path: Path | None = None) -> pd.DataFrame:
    """Kaggle martj42/international-football-results CSV yükle.

    Beklenen kolonlar: date, home_team, away_team, home_score, away_score,
    tournament, city, country, neutral.
    """
    path = path or _find_results_csv()
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].astype(bool)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def add_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """W/D/L kolonu ekle (home perspektifi)."""
    df = df.copy()
    diff = df["home_score"] - df["away_score"]
    df["outcome"] = "D"
    df.loc[diff > 0, "outcome"] = "W"
    df.loc[diff < 0, "outcome"] = "L"
    return df


def filter_recent(df: pd.DataFrame, since: str = "2010-01-01") -> pd.DataFrame:
    return df[df["date"] >= pd.Timestamp(since)].reset_index(drop=True)


def teams_2026() -> list[str]:
    """2026 WC katılımcıları (taslak — resmi kura sonrası güncellenecek).

    Ev sahipleri otomatik: USA, Canada, Mexico.
    Geri kalan 45 yer kıtasal eleme. Aşağıdaki liste 2025-Q4 itibariyle
    kesinleşmiş veya yüksek olasılıklı takımlardan örnek; kullanıcı
    `data/raw/user_provided/teams_2026.csv` koyarsa onu kullan.
    """
    return [
        # ev sahipleri
        "United States", "Canada", "Mexico",
        # UEFA (16)
        "France", "England", "Spain", "Germany", "Portugal", "Netherlands",
        "Italy", "Belgium", "Croatia", "Denmark", "Switzerland", "Poland",
        "Austria", "Turkey", "Ukraine", "Serbia",
        # CONMEBOL (6)
        "Argentina", "Brazil", "Uruguay", "Colombia", "Ecuador", "Paraguay",
        # AFC (8)
        "Japan", "South Korea", "Iran", "Saudi Arabia", "Australia",
        "Qatar", "Iraq", "Uzbekistan",
        # CAF (9)
        "Morocco", "Senegal", "Egypt", "Nigeria", "Algeria", "Tunisia",
        "Cameroon", "Ivory Coast", "Ghana",
        # CONCACAF (3 ek)
        "Costa Rica", "Panama", "Jamaica",
        # OFC (1)
        "New Zealand",
        # playoff (2 — varsayım)
        "Peru", "DR Congo",
    ]


# Türkçe -> İngilizce takım adı (results.csv İngilizce kullanıyor)
TR_TO_EN = {
    "ABD": "United States", "Meksika": "Mexico", "Kanada": "Canada",
    "Arjantin": "Argentina", "Brezilya": "Brazil", "Uruguay": "Uruguay",
    "Kolombiya": "Colombia", "Ekvador": "Ecuador", "Paraguay": "Paraguay",
    "Fransa": "France", "İngiltere": "England", "Portekiz": "Portugal",
    "İspanya": "Spain", "Almanya": "Germany", "Hollanda": "Netherlands",
    "İtalya": "Italy", "Belçika": "Belgium", "Hırvatistan": "Croatia",
    "Türkiye": "Turkey", "İskoçya": "Scotland", "Avusturya": "Austria",
    "İsviçre": "Switzerland", "Norveç": "Norway", "İsveç": "Sweden",
    "Çekya": "Czech Republic", "Danimarka": "Denmark",
    "Japonya": "Japan", "Güney Kore": "South Korea", "İran": "Iran",
    "Avustralya": "Australia", "Suudi Arabistan": "Saudi Arabia",
    "Katar": "Qatar", "Özbekistan": "Uzbekistan", "Irak": "Iraq",
    "Ürdün": "Jordan",
    "Fas": "Morocco", "Senegal": "Senegal", "Mısır": "Egypt",
    "Nijerya": "Nigeria", "Cezayir": "Algeria", "Tunus": "Tunisia",
    "Fildişi Sahili": "Ivory Coast", "Güney Afrika": "South Africa",
    "Kamerun": "Cameroon",
    "Yeni Zelanda": "New Zealand",
    "Panama": "Panama", "Kosta Rika": "Costa Rica", "Honduras": "Honduras",
    # ek olası takımlar
    "Polonya": "Poland", "Sırbistan": "Serbia", "Ukrayna": "Ukraine",
    "Gana": "Ghana", "Peru": "Peru", "DR Kongo": "DR Congo",
    "Şili": "Chile", "Bolivya": "Bolivia", "Venezuela": "Venezuela",
    "Jamaika": "Jamaica",
}


def _read_user_team_csv(p: Path) -> list[str]:
    """User CSV oku — Türkçe veya İngilizce kolonu destekle."""
    df = pd.read_csv(p)
    col = None
    for cand in ("team", "Takim", "Takım"):
        if cand in df.columns:
            col = cand
            break
    if col is None:
        col = df.columns[0]
    raw = df[col].dropna().astype(str).str.strip()
    raw = raw[raw != ""]
    out, missing = [], []
    for name in raw:
        en = TR_TO_EN.get(name, name)
        out.append(en)
        if name in TR_TO_EN:
            continue
        # İngilizce zaten ise sessiz geç
        if name not in TR_TO_EN and any(c in name for c in "ğüşöçıİĞÜŞÖÇ"):
            missing.append(name)
    if missing:
        print(f"  UYARI: TR->EN map eksik: {missing}")
    return out


def load_groups_2026(path: Path | None = None) -> list[list[str]]:
    """CSV'den grupları oku. Döndürür: 12 elemanlı liste, her eleman 4 takımlı liste (A→L sırası).
    CSV format: Group, Team kolonları.
    """
    if path is None:
        path = RAW / "2026_World_Cup_Groups.csv"
    df = pd.read_csv(path)
    groups_dict: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        g = str(row["Group"]).strip()
        t = str(row["Team"]).strip()
        groups_dict.setdefault(g, []).append(t)
    return [groups_dict[k] for k in sorted(groups_dict)]


def load_or_default_teams(path: Path | None = None) -> list[str]:
    """User klasöründe ilk CSV'yi al — yoksa default."""
    user_dir = RAW / "user_provided"
    if path and Path(path).exists():
        return _read_user_team_csv(Path(path))
    if user_dir.exists():
        candidates = sorted(user_dir.glob("*.csv"))
        for p in candidates:
            # ad eşleşme öncelik
            if "takim" in p.name.lower() or "team" in p.name.lower() or "2026" in p.name:
                return _read_user_team_csv(p)
        if candidates:
            return _read_user_team_csv(candidates[0])
    return teams_2026()
