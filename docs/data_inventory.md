# Veri envanteri — M eksikleri + L ölçeği

## M ölçeği eksik feature'ları (öncelik: yüksek)

| Feature | Kaynak | Format | Maliyet | Not |
|---|---|---|---|---|
| Bahis oranları (1X2 closing) | football-data.co.uk + Kaggle mexwell `historical-football-resultsbetting-odds-data` | CSV | Free | Kulüp ağırlıklı; milli takım kapsamı sınırlı |
| Bahis oranları (WC/Euro tarihsel) | OddsPortal scrape, Kaggle austro `beat-the-bookie` | CSV (scrape) | Free | Closing odds → market prior, calibration benchmark |
| FIFA ranking history (aylık) | Kaggle `cashncarry/fifaworldranking` (1992-2024) | CSV | Free | Walk-forward mümkün, Elo'ya yedek sinyal |
| Squad-aggregate (avg market value, age, caps) | dcaribou/transfermarkt-datasets (GitHub, weekly refresh) | CSV/Parquet | Free | `players.csv` + `national_teams.csv` join |
| Travel/rest days | Maç tarih + venue → türev | Hesaplanır | Free | Mevcut `results.csv`'den çıkarılabilir |
| Manager tenure / değişim | Wikipedia + transfermarkt | CSV (scrape) | Düşük | Az kullanılan ama kontrol için iyi |

## L ölçeği için (Bayesian + player-level)

| Feature | Kaynak | Format | Maliyet | Not |
|---|---|---|---|---|
| Tournament event data (xG, shots, passes) | StatsBomb open-data (GitHub) | JSON | Free | WC2018, WC2022, Euro2020/2024, Copa2024, AFCON2023 — tam event-level |
| xG/xGA per match (genel) | FBref `comp/218` international friendlies | HTML scrape (worldfootballR / soccerdata) | Free | Opta xG, 2017+ |
| Squad lists per match (XI + bench) | Transfermarkt + StatsBomb lineups | JSON/CSV | Free | StatsBomb tournament için tam, friendly için TM scrape |
| Player ratings | FBref per-90 + Sofascore + Transfermarkt market value | Karışık | Free (FBref/TM) | FIFA game ratings paid (SoFIFA scrape mümkün) |
| Caps + age per player | Transfermarkt dcaribou `players.csv` | CSV | Free | `national_caps`, `date_of_birth` direkt var |
| Lineups + minutes | StatsBomb (tournament) + TM (friendly) | JSON/CSV | Free | Tournament için tam, friendly sınırlı |
| Injuries / availability | Transfermarkt `injuries.csv` (salimt repo) | CSV | Free | Tarih bazlı, maç-öncesi state çıkarılabilir |

## Toplama planı

1. **Hızlı kazanç (1-2 gün):**
   - Kaggle `cashncarry/fifaworldranking` indir → `data/raw/fifa_ranking.csv`
   - Kaggle `mexwell/historical-football-resultsbetting-odds-data` → bahis 1X2 (kulüp ağırlıklı)
   - dcaribou/transfermarkt-datasets `players.csv` + `national_teams.csv` → squad-aggregate features
   - Travel/rest günleri: `results.csv`'den hesapla (city + date)

2. **Orta vadeli (3-7 gün):**
   - StatsBomb open-data clone → WC2018, WC2022, Euro2024 event data → per-match xG hesapla
   - FBref international friendlies scrape (worldfootballR) → 2017+ xG/xGA

3. **L ölçeği (kadrolar Mayıs sonu açıklandığında):**
   - 48 takım × 23 oyuncu = 1104 oyuncu → Transfermarkt market value + caps + age + minutes-club
   - Bayesian hierarchical Poisson: takım-ofansif/defansif latent + oyuncu-katkı offset
   - PyMC veya numpyro

## Vazgeçilecekler (cost > benefit şu an)

- SoFIFA / FIFA game ratings — license sorunu, scrape kırılgan
- Sofascore per-match player ratings — agresif anti-scrape, paid API gerekiyor
- Manager tenure — etki marjinal, scrape pahalı

## Sonraki karar noktası

WC kadroları açıklanınca (Mayıs 2026 sonu): L go/no-go. Şu an için **M+ feature ekleme** (FIFA rank, market value, travel) → backtest RPS düşüşü görülürse L motivasyonu güçlü.

---

## İndirilenler (2026-04-29)

| Dosya | Yol | Boyut | İçerik |
|---|---|---|---|
| FIFA rank history | `data/raw/fifa_ranking/ranking_fifa_historical.csv` | 67K satır, 1992-09 → 2024-09 | team, total_points, date, team_short |
| Transfermarkt player profiles | `data/raw/transfermarkt_dl/player_profiles.csv` | 92K oyuncu | player_id, name, dob, citizenship, current_club, position |
| TM national performances | `data/raw/transfermarkt_dl/player_national_performances.csv` | 92K satır | player_id, team_id (national), matches (caps), goals |
| TM market value | `data/raw/transfermarkt_dl/player_market_value.csv` | 901K satır | player_id, date_unix, value (zaman serisi) |
| FIFA game ratings | `data/raw/odds/davidcamilo_intl_matches.csv` | 23K maç | per-match team-mean: GK, defense, offense, midfield ratings (FIFA video game) |
| StatsBomb match indexes | `data/raw/statsbomb/matches/<comp>/<season>.json` | 6 turnuva | WC2018, WC2022, Euro2020, Euro2024, Copa2024, AFCON2023 |

## Yazılan loader'lar

- `src/fifa_rank.py` → `FIFARank.load()`, walk-forward `points()` + `rank()`
- `src/statsbomb.py` → `StatsBombFetcher`, lazy event download, `match_xg()` + `team_xg_table()`

## Bekleyen loader'lar

- TM squad-aggregate: kadro listesi açıklanınca (Mayıs 2026)
- davidcamilo FIFA game ratings → maç bazlı feature'a wrap
- StatsBomb full event download (xG team-aggregate) — n=600 maç, ~600MB JSON

## Bahis odds

International public odds dataset zayıf. football-data.co.uk sadece kulüp ligleri. Kaggle WC2022 datasetlerinde odds yok. Erteleme: ya scrape (oddsportal/checkbestodds) ya da paid (Pinnacle API). M+ için kritik değil — XGB market sinyalini başka feature'larla yakalıyor.
