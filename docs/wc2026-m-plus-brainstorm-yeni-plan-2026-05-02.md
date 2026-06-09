# WC2026 Tahmini — Sıfırdan Düşünme & Mevcut Yaklaşımla Karşılaştırma

## Bağlam

Mevcut projeyi (S baseline shipped + M/L planlanmış) bir kenara bırakıp 2026 Dünya Kupası tahmin problemine taze gözle bakıyoruz. Amaç:

1. Akademik literatür + endüstri standartlarını tarayıp bizim **kullanmadığımız yolları** bulmak.
2. **Yeni feature fikirleri** üretmek — özellikle current plan'da olmayan veya gözden kaçırılmış olanlar.
3. Eski yaklaşımın **zayıf noktalarını** ölçülebilir kıstaslarla göstermek.
4. Implementasyon yok — düşünce egzersizi.

---

## Bölüm 1: Mevcut Yaklaşımın Hızlı Özeti

| Scale | Model | Features | Sim |
|-------|-------|----------|-----|
| **S (shipped)** | eloratings.net snapshot + heuristic draw `0.28*exp(-0.0017·|Δ|)` | 5 (elo_diff, rank_diff, home_adv, form, gd_avg) | Poisson goals + forced W/D/L consistency |
| **M (yazılı, wire'lanmamış)** | Dixon-Coles MLE + XGBoost ensemble | ~15 (DC strengths, market odds, EWMA goals, attack/def) | 100K MC, real joint goal model |
| **L (planlanmış)** | Bayesian hierarchical PyMC + stacking | ~30+ (player aggregates, travel, tactical xG/xGA, fatigue) | 100K MC × 1K posterior bootstrap |

**Acknowledged weaknesses** (CLAUDE.md'den):
- Draw heuristic data'dan öğrenilmedi
- Goal sampling outcome'a forced bump'lı (S)
- Penalty ~50/50 (knockout)
- recent_form O(N·M) — backtesting'de yavaş
- Betting odds sadece placeholder
- Bracket pairing uniform random (gerçek seeding yok)

---

## Bölüm 2: State-of-Art Tarama (2024-2026)

### Akademik literatürden bulgular

1. **Bayesian conjugate Gamma-Poisson, FIFA WC2026 paper (March 2026)**
   - 23,921 maç datasetiyle log-linear Poisson + Bayesian logistic regression
   - Posterior mean goals/team: 1.3354, 95% CI [1.143, 1.542]
   - Bizim DC modelimiz MLE (point estimate); Bayesian'a geçiş = uncertainty propagation

2. **Axial Transformer in-game forecasting (arXiv 2511.18730, late 2025)**
   - 13 individual action types için joint prediction (player + team + match seviyesinde)
   - Temporal dynamics + player interaction self-attention
   - Bizim use-case için overkill — ama **player-team heterogeneous graph transformer** (arXiv 2507.10626) milli takım için uygulanabilir

3. **Generalizable ML for WC outcome prediction (arXiv 2505.01902)**
   - Player-level features: top-11 minutes, market value, age distribution
   - Cross-tournament generalization: WC + Euro + Copa data karışımı

4. **"Is Football Unpredictable?" (MDPI 2024)**
   - NN'ler Poisson'u XGBoost'u geçiyor ama **calibration kötü** — accuracy ≠ profit
   - Calibration-optimized model accuracy-optimized'a göre %69.86 daha yüksek return

### Endüstri benchmark'ları

5. **Opta Power Rankings** — modern Elo-style + xG performance + opposition quality + recency. Bizim Elo'dan farkı: skor-bazlı değil, xG-bazlı update.

6. **538 SPI** — offensive/defensive ratings, Transfermarkt market value preseason prior. WC için her takım offense/defense pair (bizimki tek skalar Elo).

7. **Bookmaker market** — "betting odds rating system" (PLOS One) en güçlü baseline. Closing odds'tan implied prob > tüm akademik modeller (literatür konsensüsü). **Bizim model market'i baseline olarak kullanmıyor**.

### Domain bilgisi

8. **Altitude effect** (Mexico 2026 host: Mexico City 2,240m) — repeated-sprint capacity hypoxia ile düşüyor. Bizim modelde altitude feature **yok**.

9. **Travel fatigue** — 57 yıllık Bundesliga datasında home advantage'ın travel mesafesiyle artışı kanıtlandı. WC2026 üç ülke arası fixture → travel cumulative load **modellenmemiş**.

10. **Calibration over accuracy** — Platt scaling / isotonic regression post-hoc. Bizim eval.py'da RPS var ama **calibration plot (reliability diagram) yok**, isotonic refinement yok.

---

## Bölüm 3: Bizim Kaçırdığımız / Düşünmediğimiz Yollar

### A. **Modelleme Yaklaşımları**

| # | Yaklaşım | Mevcut planımızda var mı? | Beklenen kazanç |
|---|----------|--------------------------|-----------------|
| A1 | **Market-implied prob baseline** (closing odds → devig → Platt-scale) | Hayır (placeholder) | RPS'de literatürün en güçlü baseline'ı; **ensemble'da %20-30 ağırlık vermek standart** |
| A2 | **Calibration layer** (isotonic regression on validation fold) | Hayır | Brier %5-10 düzelir, profit/RPS ilişkisi düzelir |
| A3 | **Offense/Defense ratings (538-style)** Elo yerine ayrı | Hayır (tek-skalar Elo) | Düşük-skor vs yüksek-skor takımları ayırt eder; goal modeline doğal feed |
| A4 | **Bivariate Poisson / Skellam** Dixon-Coles yerine | Sadece DC planlandı | DC'nin τ correction'ı sadece düşük skor; Bivariate joint correlation modellemesi daha temiz |
| A5 | **Hierarchical Bayesian by confederation** (UEFA/CONMEBOL/CAF/AFC/CONCACAF/OFC pooling) | L scale'de "var ama belirsiz" | Az sample'lı takımlar (Curaçao, Cape Verde) için partial pooling — flat prior'dan iyi |
| A6 | **Gaussian Process on time-varying strength** (Elo'nun smooth versiyonu) | Hayır | Static snapshot yerine takım strength'i smooth time series; injury/coach change recovery |
| A7 | **Neural rating (mini graph NN)** — takımları node, maçları directed edge | Hayır | Transfermarkt/UEFA ranking gibi external + Elo + recent form'u tek embed'e koymak |
| A8 | **Multi-task NN** — joint predict (W/D/L) + (goals_home, goals_away) + (yellow cards) shared backbone | Hayır | Auxiliary task'lar regularizer; small data için faydalı |

### B. **Yeni Feature Aileleri**

#### B1. **Coğrafi & venue-spesifik**
- **`altitude_diff`**: ev sahibi şehrin rakımı − son 5 maç ortalama rakım. Mexico City (2,240m), Toluca (2,667m) ekstrem.
- **`temperature_humidity_idx`**: maç saati + şehir + Haziran-Temmuz iklim ortalaması. Houston/Miami nem vs Vancouver serin.
- **`stadium_familiarity`**: takımın o stadyumda son 5 yıl maç sayısı (US/Mexico takımları home advantage'ı geri alıyor)
- **`pitch_size_diff`**: standardize edilmiş ama küçük varyasyon var; FBref/Wyscout'tan

#### B2. **Travel & yorgunluk**
- **`cumulative_travel_km`**: turnuvada o ana kadar kat edilen mesafe (US-Canada-Mexico fixture'ları)
- **`timezone_shifts`**: total saat dilimi değişimi (jetlag)
- **`days_since_last_match`** + **`days_since_last_match_squared`** (hem kısa hem aşırı uzun dinlenme zarar)
- **`venue_change_count`**: kaç farklı şehirde oynadı

#### B3. **Squad-level (L scale'de var ama parçalı)**
- **`squad_market_value_top11_log`**: Transfermarkt — literatürde Amerika takımları için FIFA rank'ten daha iyi
- **`squad_age_distribution_skew`**: yaş çeşitliliği (yaşlı + genç karışımı vs homojen)
- **`top11_minutes_last_season`**: kulüp formundaki ana 11
- **`squad_caps_total`**: tecrübe (uluslararası maç sayısı toplamı)
- **`coach_tenure_days`**: koç ne kadardır görevde
- **`coach_prior_wc_appearances`**

#### B4. **Tactical / advanced stats**
- **`xg_per_match_last_24mo`** (FBref international friendlies + qualifiers — tournament-only değil!)
- **`xg_against_per_match`**
- **`shot_quality_index`**: shots / xG ratio (verimlilik)
- **`set_piece_xg_share`**: knockout düşük tempo'da set piece kritik
- **`pressing_intensity_proxy`**: sırasıyla high-press takımlar yorgunluğa daha açık

#### B5. **Match context**
- **`elimination_pressure`**: takımın ilerlemek için kazanma zorunluluğu (group stage 3. maç dinamiği)
- **`opponent_must_win`**: rakibin durumu — bilinçli berabere arayışı
- **`group_seeding_disadvantage`**: pot-based draw fairness (grup zorluk indeksi)
- **`bracket_path_difficulty_proxy`**: knockout polu beklenen rakip Elo ortalaması

#### B6. **Recency & momentum (heuristic ötesi)**
- **`elo_acceleration`**: son 6 ay Elo türevi (yükseliş trendi)
- **`form_streak_signed`**: ardışık W/L (değişim noktası)
- **`recent_quality_of_opposition`**: form_diff'i opposition Elo ile ağırlıklandır

#### B7. **Goal-process structure**
- **`expected_score_diff_first_30min`** (ev güçlü takımlar erken açar)
- **`comeback_prob_when_down1`** (takım-spesifik)
- **`extra_time_propensity`** (knockout'ta önemli)

### C. **Simülasyon İyileştirmeleri**

| # | Mevcut | Önerilen iyileştirme |
|---|--------|---------------------|
| C1 | Bracket sonrası uniform random pairing | Gerçek 2026 seeding kuralı (group winner vs runner-up cross-bracket) |
| C2 | Penalty ~50/50 | Takım-spesifik penalty success rate (historical PSO) + keeper save rate |
| C3 | Group H2H tiebreaker FIFA-compliant ✓ | Fair-play points tiebreaker eklensin (FIFA 2018+ kuralı) |
| C4 | Goal Poisson independent | Bivariate Poisson (correlation parameter ρ) |
| C5 | Single-shot 100K MC | **Posterior bootstrap**: Bayesian model'den 1K parameter draw × 100 MC her draw için → champion prob CI |
| C6 | Champion prob nokta tahmin | **Sensitivity analysis**: feature perturbation ile robustness ("Brezilya forvet sakat → ne değişir?") |

### D. **Validation & Eval Yaklaşımları**

| # | Mevcut | Önerilen |
|---|--------|----------|
| D1 | RPS, Brier, log-loss, accuracy | + **Reliability diagram** (kalibrasyon görsel) |
| D2 | Smoke test (synthetic) | + **WC2018 + WC2022 holdout backtest** (literatür konsensüsü test seti) |
| D3 | Per-match metrics | + **Per-confederation breakdown** (model güçlü takımlara overfit ediyor mu?) |
| D4 | Single seed | + **Seed sensitivity** (50 seed'e çalıştır, champion prob varyansı) |
| D5 | Yok | + **Bookmaker baseline comparison** — closing odds'a karşı RPS farkı pozitif mi? |
| D6 | Yok | + **Calibration-by-confidence** decile plot |

---

## Bölüm 4: Eski vs Yeni — Karşılaştırmalı Skor

| Boyut | S (current) | M (planned) | **Yeni Önerilen Yön** |
|-------|-------------|-------------|----------------------|
| Outcome model | Heuristic Elo + draw exp | Elo + DC + XGB ensemble | **+ Market baseline + Bivariate Poisson + Calibration layer** |
| Goal model | Poisson + forced bump | Dixon-Coles τ-corrected | **Bivariate Poisson with ρ + venue-adjusted λ** |
| Strength | Tek skalar (Elo) | DC dual (α, β) | **Hierarchical Bayesian (confederation pooling) + Offense/Defense decomposition** |
| Features | 5 | ~15 | **~25-30** (yukarıdaki B1-B7) |
| Sim | 100K MC random bracket | 100K MC + DC goals | **100K × posterior bootstrap + true 2026 bracket pairing + altitude/travel modulation** |
| Eval | RPS/Brier | + holdout | **+ Reliability + per-confed + market baseline + seed sensitivity** |
| Calibration | Yok | Yok (planned değil!) | **Isotonic regression refinement layer** |
| Uncertainty | Tek nokta | Tek nokta | **95% CI on champion prob (posterior + bootstrap)** |

**Beklenen RPS iyileşmesi (literatüre göre):**
- S → M (DC + XGB ensemble): %5-8 RPS düşüşü
- M → "Yeni Yön" (market baseline + calibration + venue features): **%10-15 ek RPS düşüşü** (calibration + market en büyük single-feature kazanç)

---

## Bölüm 5: Yeni Yön Önerisi — "M+" Scale

S/M/L planını değiştirmeden, **M üzerine bina** olarak şu paketi öneriyorum:

### M+ Paketi (M scale wire-up'tan SONRA, L'den ÖNCE)

**1. Market baseline + ensemble member**
- Closing odds (Pinnacle/Betfair) kazıyıcısı
- Devig (3-way → fair probabilities)
- Ensemble'a 3. üye olarak ekle (Elo + DC/XGB + Market)

**2. Calibration layer**
- Walk-forward validation fold'unda isotonic regression fit
- Final ensemble output'a uygula
- Reliability diagram raporla

**3. Venue/travel feature subset (B1+B2)**
- altitude_diff, temperature_humidity_idx, cumulative_travel_km, days_since_last_match
- 2026 fixture data'sından deterministik hesap (host + dates known)
- XGB feature olarak ekle

**4. Bivariate Poisson goal model**
- DC'nin τ correction yerine Karlis-Ntzoufras Bivariate Poisson (ρ parametresi)
- simulate.py'daki forced bump tamamen kalkar

**5. True 2026 bracket simulation**
- FIFA'nın açıklayacağı seeding kurallarına göre R32 pairing
- Currently uniform — düzelt

**6. Posterior bootstrap (mini-Bayesian)**
- DC parametrelerine MLE yerine MCMC (sadece DC için, full Bayesian değil)
- 100 posterior draw × 1K MC = champion prob CI
- L scale'in baby version'ı, kompleks olmadan

**Beklenen kazanç vs M:** RPS %10-15 + uncertainty quantification (CI'lar) + calibration düzgün.

**Skip edilenler (L'ye veya hiçbir yere):** Player-level squad aggregates (Transfermarkt scrape büyük iş), GNN, Transformer, multi-task NN — **hepsi marjinal kazanç, yüksek complexity**. Literatürde calibration + market + venue önerilerin tek başına büyük farkı yapan kısım.

---

## Bölüm 6: Yeni Feature Önceliklendirme

| Feature | Cost (saat) | Beklenen RPS impact | Veri kaynağı | Öncelik |
|---------|-------------|---------------------|--------------|---------|
| Market closing odds devig | 4-8 | **Yüksek** (literatür konsensüsü) | Pinnacle/Betfair scrape veya OddsPortal | **P0** |
| Isotonic calibration | 2-4 | **Yüksek** | sklearn IsotonicRegression | **P0** |
| altitude_diff | 1-2 | Orta | manuel city → elevation lookup | **P1** |
| cumulative_travel_km | 2-3 | Orta | haversine + fixture df | **P1** |
| days_since_last_match | 0.5 | Düşük-orta | fixture df'den trivial | **P1** |
| Bivariate Poisson | 6-10 | Orta | Karlis-Ntzoufras paper, scipy | **P1** |
| 2026 true bracket pairing | 3-5 | Düşük (sim varyansı) | FIFA tournament regs | **P2** |
| Confederation hierarchical Bayesian | 12-20 | Orta-yüksek (data-poor takımlar) | PyMC | **P2** |
| Squad market value (top-11) | 8-15 | Orta | Transfermarkt scrape | **P3** (M++/L) |
| xG_for/xG_against (FBref) | 6-10 | Belirsiz (memory'deki sebep — qualifier'da yok) | FBref scrape | **P3** (zaten kararlaştırıldı) |
| GNN team embedding | 20-40 | Belirsiz | torch_geometric | **Skip** (overkill 48 takım) |
| Transformer match seq | 30-60 | Marjinal | — | **Skip** |

---

## Bölüm 7: Risk & Sınırlamalar

1. **Closing odds erişimi** — WC için tüm 104 maç henüz fiyatlanmadı. Group stage opening'den bracket fixture'lara kadar dinamik. Live scrape gerekli.
2. **Calibration overfit** — küçük validation fold'unda isotonic regression overfit edebilir; 5-fold CV içinde nested calibration.
3. **Venue features sadece WC2026'ya özgü** — modelin generalizability'sini düşürür (eski WC backtest'lerde altitude/travel datası eksik).
4. **Bivariate Poisson MLE** — convergence sorunları olabilir; DC fallback hazır olsun.
5. **Posterior bootstrap compute** — 100 draw × 1K MC = 100K total sim; M baseline'la aynı, OK.

---

## Bölüm 8: Kullanıcı Kararları (locked in)

- **Yön:** Tüm M+ paketi (market + calibration + venue/travel + bivariate Poisson + true bracket + posterior bootstrap)
- **Veri toplama:**
  - ✅ Closing odds scraper (Pinnacle/Betfair/OddsPortal)
  - ✅ Stadium/altitude/iklim lookup — sadece 16 host şehir (US + Mexico + Canada), manageable
  - ❌ Transfermarkt squad scrape — şimdilik atla (M++/L için saklı)
- **Yeni eklenen modül:** **Sentiment Analysis** — closing odds zaten match-window'da oluşacağına göre, paralel olarak sentiment sinyali de toplansın

---

## Bölüm 9: Sentiment Analysis Modülü (yeni)

### Niye değerli?

- **Closing odds latency'sinden önce hareket eder** — sakatlık, dressing-room turmoil, koç beyanları news/social'a saatler önce düşer; odds 30-60dk sonra adapte olur.
- **Bookmaker bias proxy'si** — yoğun negatif sentiment + sabit odds → market henüz fiyatlamamış, model edge'i.
- **Calibration sinyalı** — yüksek volatilite (std up) durumunda ensemble'ın confidence'ı düşürülmeli.

### Üç boyut

**(1) Kaynak**
- **Twitter/X** — per-team handle + hashtag (#TR, #BRA), pre-match 48h window
- **Reddit r/soccer + national subs** (r/futbol, r/Brasil_soccer)
- **News headlines** — ESPN, BBC Sport, transfermarkt news, GA aggregator
- **Match preview articles** — bookmaker preview pages, betting tipsters (zayıf sinyal ama var)

**(2) Granülarite**
- `sentiment_team_pre48h_mean`: takım için 48h pencerede ortalama compound sentiment ([-1, +1])
- `sentiment_team_pre48h_std`: belirsizlik (yüksek = haber yoğun + kutuplaşmış)
- `sentiment_team_volume_24h`: total mention count (ilgi/önem proxy'si)
- `sentiment_change_24h_to_pre24h`: trend (negatife dönüş = sakatlık/skandal sinyali)
- `injury_keyword_density`: "injury|sakatlık|out|ruled out|doubt" anahtar kelime yoğunluğu (özel signal)

**(3) Zaman**
- **T-72h:** baseline sentiment snapshot
- **T-24h:** son güncellem (squad announcement sonrası)
- **T-3h:** lineup announcement etkisi (bonus, gerekli değil)

### Teknik yığın

| Katman | Seçim | Sebep |
|--------|-------|-------|
| Tweet scrape | snscrape (no API) veya Twitter Academic (paid) | snscrape free, 2026'da hâlâ çalışıyor |
| News aggregator | NewsAPI.org veya Google News RSS scrape | Free tier 100/gün, WC için yetersiz olabilir → fallback: GDELT |
| Reddit | PRAW + pushshift fallback | Free, rate-limited |
| Sentiment classifier | **XLM-RoBERTa multilingual** (HuggingFace `cardiffnlp/twitter-xlm-roberta-base-sentiment`) | 48 takım = ~25 dil; mono-lingual VADER kifayetsiz |
| Injury keyword | Curated multilingual lexicon (50-100 kelime, 25 dil) | Lightweight, hızlı |
| Aggregation | pandas groupby on (team, date, window) | Standard |

### Model entegrasyonu

İki seçenek:

**A) Feature olarak XGBoost'a ek**
- Yukarıdaki 5 sentiment metrik (mean, std, volume, change, injury_density) iki takım için → 10 yeni feature
- Mevcut M ensemble'a doğal ekleme

**B) Ayrı ensemble üyesi**
- Standalone sentiment model (logistic regression sentiment_diff features → W/D/L)
- Ensemble'a 4. üye olarak (Elo + DC + XGB + Sentiment)
- Düşük weight (0.05-0.10) bekleniyor

**Önerilen: A** — daha az kod, XGB nonlinear interaction'ları yakalar, tek hyperparameter tuning loop'u.

### Validation challenge

Sentiment'in geçmiş validation seti yok — backtest WC2018/WC2022 için tweet/news arşivi gerekir:
- **Twitter:** Internet Archive / academic dump (zor erişim)
- **News:** GDELT 2.0 (mevcut, free) — 2015+ kapsama
- **Reddit:** Pushshift archive (belirli yıllara kadar)

→ **Pratik karar:** Sentiment WC2026 prospektif olarak ölçülecek. WC2022 için GDELT ile mini backtest yapılabilir (kabaca validation).

### Cost & risk

- Twitter scrape: rate limit + ToS riski (snscrape periyodik kırılır)
- 25 dilli lexicon kurulumu: 4-6 saat
- HuggingFace inference: GPU yok → CPU'da batch, ~5K tweet/dk
- Bot/spam filtreleme: gerekli, yoksa noise dominant
- **Toplam ek effort:** 15-25 saat (M+ paketi üstüne)

### M+ paketine eklenmiş hali

```
M+ (final)
├── Market baseline (P0)
├── Calibration layer / isotonic (P0)
├── Sentiment module (P0.5 — yeni)
│   ├── Twitter/Reddit/News scraper
│   ├── XLM-R multilingual classifier
│   ├── Injury keyword lexicon
│   └── 10 sentiment features → XGBoost ensemble
├── Venue features (altitude, temp, stadium familiarity) (P1)
├── Travel features (cumulative km, days_since, timezone) (P1)
├── Bivariate Poisson goal model (P1)
├── True 2026 bracket pairing (P2)
└── Posterior bootstrap (mini-Bayesian on DC) (P2)
```

---

## Bölüm 10: Tahmini RPS Yolu

| Kademe | Beklenen RPS | Δ vs önceki | Notlar |
|--------|--------------|-------------|--------|
| S (current) | ~0.205 (tahmin) | — | Heuristic Elo |
| M (wire-up) | ~0.193 | -0.012 | DC + XGB ensemble |
| **M+ (P0: market + calibration)** | ~0.180 | -0.013 | En büyük tek atlama |
| **M+ (+sentiment)** | ~0.176 | -0.004 | Marjinal ama match-day signal |
| **M+ (+venue/travel)** | ~0.171 | -0.005 | WC2026 spesifik kazanç |
| **M+ (+bivariate Poisson + bracket + bootstrap)** | ~0.168 | -0.003 | Goal model + uncertainty |
| L (full Bayesian + player) | ~0.164 | -0.004 | Marjinal, yüksek effort |

(Rakamlar literatür ortalamalarından kaba; gerçek backtest sonrası kalibre edilecek.)

---

## Bölüm 11: Düşünce Özeti

Sıfırdan baktığımızda **3 büyük eksik** çıktı:

1. **Market baseline yok** — literatürün en güçlü tek kazanımı, mevcut planda placeholder
2. **Calibration layer yok** — RPS/Brier için %5-10 ücretsiz iyileşme
3. **WC2026'ya özgü context kullanılmıyor** — altitude (Mexico City), travel (3-ülke fixture'ları), iklim (Haziran-Temmuz Houston/Miami), 2026'nın yeni bracket formatı

Eklediğimiz **sentiment modülü** match-day signal boyutu açıyor — odds'tan önce hareket eden "soft" sinyaller. Backtest zorluğu var (WC2018/22 arşivi sınırlı) ama prospektif WC2026 için doğal fit.

**Skip kararı:** GNN, Transformer, multi-task NN, full player-level features. 48 takımlık bir turnuvada bu modeller marjinal kazanç sunar; veri sparsity overfit riski.

**Beklenen toplam etki:** S baseline'a göre **~%18 RPS düşüşü** (0.205 → 0.168), uncertainty quantification, ve canlı match-day adapte olabilen sistem.

---

## Bölüm 12: Final Tasarım Kararları (locked in)

| Karar | Seçim | Not |
|-------|-------|-----|
| Sentiment kaynakları | **Üçü birden**: Twitter (snscrape) + News (GDELT 2.0 + RSS) + Reddit (PRAW + pushshift) | Weighted aggregation, kaynak başı ayrı sentiment_* kolonları + global weighted ortalama |
| Compute | **Google Cloud VM** (kullanıcının kendi kredisi) — XLM-R batch inference local | HF Inference API'ye gerek yok, $0 ek; opsiyonel T4 GPU spot ~$0.10/h hızlandırma |
| Backtest | **GDELT WC2022 mini backtest** | Sadece news kanalıyla sanity check; Twitter/Reddit arşivi sınırlı, prospektif WC2026'da full ölçüm |

### GCP setup taslağı

- **VM:** n1-standard-4 (4 vCPU, 15GB), Ubuntu, ~$0.19/h ya da preemptible
- **Storage:** 50GB persistent disk (modeller + raw scrape arşivi ~30GB tahmin)
- **Cron schedule:** günde 4 kez (T-72h, T-48h, T-24h, T-3h pencereleri için), maç fixture'una bağlı
- **Output sync:** GCS bucket'a parquet, lokal model bunu okur

### Backtest yolu (GDELT WC2022)

1. GDELT 2.0 GKG (Global Knowledge Graph) tablosundan 2022-11-20 → 2022-12-18 takım-eşleşmiş haberleri çek
2. XLM-R ile sentiment skorla, takım-gün ortalama hesapla
3. M model output'una sentiment feature'ları ekle, retrain, RPS değişimi ölç
4. Eğer Δ RPS > 0.005 → WC2026 için full pipeline yatırımı haklı; aksi halde sadece P0 (market+calibration) yeterli, sentiment'e zaman harcama

---

## Bölüm 13: Final Yol Haritası

```
[CURRENT] S baseline (shipped)
   │
   ▼
[STEP 1] M wire-up — DC + XGB ensemble (mevcut planda yazılı, wire'lanmamış)
   │
   ▼
[STEP 2] M+ paketi — bu plan'ın sonucu
   │
   ├── 2a. Market baseline + calibration layer (P0)
   ├── 2b. Sentiment pipeline (GCP VM, üç kaynak, GDELT WC2022 backtest)
   ├── 2c. Venue features (16 host şehir lookup tablosu)
   ├── 2d. Travel features (cumulative_km, days_since, timezone)
   ├── 2e. Bivariate Poisson goal model (DC replacement)
   ├── 2f. True 2026 bracket pairing (FIFA seeding kuralı)
   └── 2g. Posterior bootstrap (mini-Bayesian on DC)
   │
   ▼
[STEP 3] L (orijinal plan) — Bayesian hierarchical + player-level (M++ M+ sonrası)
```

### Kritik dosyalar (implementasyon zamanı geldiğinde dokunulacak)

- `src/elo.py` — eloratings.net snapshot wrapper, draw heuristic (M+'da değişmiyor)
- `src/poisson.py` — Dixon-Coles MLE (M+'da Bivariate Poisson alternatifi eklenecek)
- `src/ml.py` — XGBoost pipeline (sentiment + venue + travel features eklenecek)
- `src/features.py` — feature builder (yeni 10 sentiment + 5 venue + 4 travel feature)
- `src/simulate.py` — bracket pairing düzeltilecek, goal sampling yenilenecek
- `src/eval.py` — reliability diagram + per-confederation breakdown eklenecek
- **YENİ:** `src/market.py` — odds devig + ensemble integration
- **YENİ:** `src/calibration.py` — isotonic regression refinement
- **YENİ:** `src/sentiment/` — scraper + classifier + aggregator
- **YENİ:** `data/raw/venues_2026.csv` — 16 stadium × (city, altitude, climate avg)
- **YENİ:** `data/raw/odds/` — günlük closing odds parquet'leri
- **YENİ:** `data/raw/sentiment/` — günlük sentiment parquet'leri (team × source × date)

### Verification (implementasyon sonrası)

1. **Smoke test extension:** scripts/smoke_test.py'a market + sentiment synthetic data eklenecek; full pipeline assertion: `top-5 P_Champion > 0.30` + `calibration ECE < 0.05`
2. **WC2018 + WC2022 holdout backtest:** RPS, Brier, log-loss üç model için (S, M, M+)
3. **GDELT WC2022 sentiment-only ablation:** sentiment çıkar/koy, RPS Δ ölç
4. **Reliability diagram:** holdout'ta 10-bin kalibrasyon görsel
5. **Per-confederation breakdown:** UEFA/CONMEBOL vs CAF/CONCACAF/AFC RPS farkı

---

## Bölüm 14: Sonuç

Sıfırdan düşündüğümüzde **3 kritik eksiği** yakaladık:
1. Market baseline yokluğu (literatürün en güçlü tek kazanımı)
2. Calibration layer eksikliği (ücretsiz Brier %5-10 iyileşme)
3. WC2026 host context (altitude, travel, iklim) hiç kullanılmıyor

Yeni eklenen **sentiment modülü** match-day signal boyutu açıyor — odds'tan saatler önce hareket eden soft sinyal. GCP'de local XLM-R ile $0 ek maliyet, GDELT backtest ile sanity check.

**Skip kararları:** GNN, Transformer, multi-task NN, full Transfermarkt scrape — 48 takımlık WC için marjinal kazanç, yüksek effort, overfit riski.

**Beklenen toplam etki:** S baseline → M+ tam paketi: **~%18 RPS düşüşü** (0.205 → 0.168), uncertainty CI'lar, live match-day adapte edebilen sistem.

Bu düşünce dokümanı hazır. **Hiçbir kod yazılmadı.** Implementasyon onayı geldiğinde `superpowers:writing-plans` skill'i ile detaylı execution plan'a geçilecek.

---

## Not: Mevcut Plan'la İlişki

Bu doküman mevcut S→M→L roadmap'i **yıkmıyor**, **M ile L arası "M+" kademesi** öneriyor. Mevcut plan'daki:
- Dixon-Coles, XGBoost, ensemble — geçerli, wire-up gerek
- Bayesian hierarchical (L) — eskidiği gibi geçerli ama M+'tan sonra
- Player-level features (L) — düşük öncelik (qualifier veri yok)

Yeni eklenen ana fikirler:
- **Market baseline 1. öncelik** (literatürün en güçlü tek baseline'ı, bizde yok)
- **Calibration layer ayrı bir step** (S/M/L planında yok!)
- **Venue/travel features** (host-spesifik, WC2026'ya özel kazanç)
- **Bivariate Poisson** (DC'nin daha temiz alternatifi)
- **Posterior bootstrap** (full Bayesian'dan ucuz uncertainty)
