# Literatür Taraması — 2026 WC Tahmin Modeli

Ortak dosya: sen + Claude bulduklarını buraya ekle.

## Format

```
### [Başlık](URL)
- **Kaynak:** dergi/konf/preprint
- **Yöntem:** ne yapıyor
- **Bizimle alakası:** M/L scale için ne alınabilir
- **Notlar:** —
```

---

## A. Temel Modeller

### Dixon & Coles (1997) — Modelling Association Football Scores
- **Kaynak:** Applied Statistics (JRSS-C), 46(2):265–280
- **Yöntem:** Bağımsız Poisson'dan sapan düşük-skor düzeltmesi (τ parametresi); attack/defence gücü tahmin
- **Bizimle alakası:** M scale'deki DC implementasyonunun orijinal referansı
- **Notlar:** τ düzeltmesi 0-0, 1-0, 0-1, 1-1 maçlarda önemli; şu an M'de var mı kontrol et

### Maher (1982) — Modelling Association Football Scores
- **Kaynak:** Statistica Neerlandica, 36(3):109–118
- **Yöntem:** Bağımsız Poisson, attack/defence parametreleri
- **Bizimle alakası:** DC'nin öncülü; L scale Bayesian için prior yapısı buradan alınabilir
- **Notlar:** —

---

## B. Elo & Rating Sistemleri

### [Hvattum & Arntzen (2010) — Using ELO ratings for match result prediction](https://www.sciencedirect.com/science/article/abs/pii/S0169207009001708)
- **Kaynak:** International Journal of Forecasting
- **Yöntem:** Elo'yu ordered logit modeline feature olarak sokar; W/D/L prediction
- **Bizimle alakası:** Bizim M modelinin tam iskeleti; Elo diff → tahmin
- **Notlar:** RPS baseline: ~0.2035; bizim M: 0.9866 log_loss (farklı metrik, karşılaştırılabilir değil doğrudan)

### [Constantinou & Fenton (2012) — Pi-Ratings](http://www.constantinou.info/downloads/papers/pi-ratings.pdf)
- **Kaynak:** Journal of Quantitative Analysis in Sports
- **Yöntem:** Gol farkını da kullanan dinamik rating; home/away ayrı parametreler
- **Bizimle alakası:** Elo'nun zayıf noktası gol farkını ignore etmesi; pi-rating bunu çözer
- **Notlar:** CatBoost + pi-ratings = en iyi 2017 Soccer Prediction Challenge sonucu (RPS 0.1925). XGBoost feature olarak dene: M'e ekle

---

## C. Makine Öğrenmesi (XGBoost / Ensemble)

### [Yeung et al. (2023) — Evaluating Soccer Match Prediction Models: Deep Learning & GBT](https://arxiv.org/abs/2309.14807)
- **Kaynak:** arXiv:2309.14807
- **Yöntem:** CatBoost + pi-ratings vs deep learning; feature optimization
- **Bizimle alakası:** Pi-ratings > Elo as feature; CatBoost/XGBoost en iyi tek model
- **Notlar:** Bizim için: XGBoost'a Elo yerine pi-rating dene; RPS primary metric (bizimle aynı)

### [arxiv 2403.07669 — ML for Soccer Match Result Prediction (survey)](https://arxiv.org/pdf/2403.07669)
- **Kaynak:** arXiv:2403.07669
- **Yöntem:** Survey; Poisson, Elo, XGBoost, neural net karşılaştırması
- **Bizimle alakası:** Ensemble = %70+ accuracy; Poisson+Elo+XGBoost+NN birleşimi
- **Notlar:** Tek model sınırı ~67% (XGBoost); ensemble +3-5pp

### [arxiv 2211.15734 — XAI for Football Prediction](https://arxiv.org/pdf/2211.15734)
- **Kaynak:** arXiv:2211.15734
- **Yöntem:** XGBoost + SHAP açıklanabilirlik
- **Bizimle alakası:** Feature importance analizi için; hangi feature M'i en çok etkiliyor
- **Notlar:** Fetch edip SHAP bulgularına bak

---

## D. Bayesian Hiyerarşik Modeller

### [Baio & Blangiardo (2010) — Bayesian hierarchical model for football results](https://gianluca.statistica.it/research/football/index.html)
- **Kaynak:** Journal of Applied Statistics, 37(2):253–264
- **Yöntem:** Poisson likelihood; attack/defence latent params; hierarchical prior (exchangeable teams); MCMC (OpenBUGS) → sonra INLA
- **Bizimle alakası:** L scale'in tam şablonu; Stan'a çevrilebilir
- **Notlar:** Overshrinkage sorunu var → mixture prior ile çözüyor; dikkat et

### [footBayes R package (Egidi et al.) — Diagonal-Inflated Bivariate Poisson](https://cran.r-project.org/web/packages/footBayes/index.html)
- **Kaynak:** CRAN / Stan
- **Yöntem:** DIBP (Diagonal-Inflated Bivariate Poisson); draw inflation; FIFA ranking farkı predictor; dynamic-autoregressive attack/defence priors
- **Bizimle alakası:** 2022 WC'de bookmakers'ı geçti (pseudo R²=0.40 vs 0.36). L scale için referans implementasyon
- **Notlar:** 3000+ uluslararası maç 2018-2022. Python'a çevirmek için Stan kodu açık kaynak

### [ResearchGate — Bayesian approach for FIFA WC 2026](https://www.researchgate.net/publication/389390461_A_Bayesian_approach_for_predicting_match_outcomes_FIFA_World_Cup_2026)
- **Kaynak:** ResearchGate preprint (2025)
- **Yöntem:** Bayesian logistic regression + gradient boosting; probabilistic framework
- **Bizimle alakası:** Doğrudan rakibimiz; metodoloji farklılıklarına bak
- **Notlar:** 403 hatası; başka yerden fetch dene veya sen PDF paylaş

### [Baio (2022) — WC 2022 Predictions with footBayes/Stan](https://statmodeling.stat.columbia.edu/2022/11/19/football-world-cup-2022-predictions-with-stan/)
- **Kaynak:** Statistical Modeling blog (Gelman)
- **Yöntem:** DIBP + autoregressive priors; real WC 2022 uygulaması
- **Bizimle alakası:** Canlı turnuva update yapısı; biz de round-by-round güncelleme yapabilir miyiz?
- **Notlar:** 403 aldık ama içerik search'ten özetlendi

### [Extending Dixon-Coles — Sarmanov Family (Michels et al. 2023)](https://arxiv.org/abs/2307.02139)
- **Kaynak:** arXiv:2307.02139 (stat.ME)
- **Yöntem:** DC'yi Sarmanov ailesi olarak genelleştirir; alternatif discrete dağılımlar
- **Bizimle alakası:** Mevcut DC implementasyonumuzun τ parametresi Sarmanov özel durumu; women's football'da farklı karakteristik → men's için ne değişiyor?
- **Notlar:** Direkt uygulama az; teorik arka plan için faydalı

---

## E. Turnuva Simülasyonu & Format

### [2026 FIFA World Cup — Wikipedia](https://en.wikipedia.org/wiki/2026_FIFA_World_Cup)
- **Kaynak:** Wikipedia / FIFA resmi
- **Kapsam:** Tam format, seeding, grup kuralları

### Format Özeti (simulate.py için kritik)

**Grup Aşaması:**
- 48 takım → 12 grup × 4 takım (A–L)
- 72 grup maçı, her takım 3 maç
- Grubun ilk 2'si otomatik R32 (24 takım)
- En iyi 8 üçüncü sıra → R32 (8 takım) → toplam 32

**Üçüncü Sıra Seçimi (simulate.py'de implement edilmeli):**
- 12 üçüncünün tamamı tek listede sıralanır
- Kriter sırası: Puan → Gol farkı → Atılan gol → Fair play puanı → FIFA ranking → kura
- En iyi 8'i R32'ye gider

**R32 Bracket Pairing (şu an simulate.py'de EKSİK/YANLIŞ):**
- FIFA önceden pairing tablosu belirler: hangi grubun 3.'sü hangi eşleşmeye gider
- Örnek: Group E 1. vs {A/B/C/D/F}'nin üçüncüsü
- `simulate.py` bunu uniform random yapıyor — **gerçek olmayan sonuç üretiyor**
- Fix: FIFA'nın resmi bracket tablosunu hardcode et

**Seeding Potları (draw için):**
- Pot 1: Ev sahipleri (ABD, Kanada, Meksika) + FIFA sıralama top-9 (12 takım)
- Pot 2–4: Kalan 36 takım FIFA sıralamasına göre bölünür
- Konfederasyon kısıtı: Grupta max 2 UEFA, max 1 diğer konfederasyondan

**Tiebreaker (grup sıralaması, sırayla):**
1. Puan
2. Gol farkı (grup içi)
3. Atılan gol (grup içi)
4. H2H puan
5. H2H gol farkı
6. H2H atılan gol
7. Fair play puanı
8. FIFA ranking
9. Kura

### simulate.py Boşluk Analizi

| Sorun | Etki | Fix |
|-------|------|-----|
| `_best_third_place()` yok/basit | Yanlış 8 takım ilerliyor | Tam tiebreaker uygula |
| R32 bracket pairing uniform random | Şampiyonluk olasılıkları çarpık | FIFA tablosunu hardcode et |
| Konfederasyon kısıtı yok | Grup çekimi gerçekçi değil | Simülasyon için önemsiz (zaten çekildi) |
| Penaltı simülasyonu yok | Knockout maçlarda draw sonucu | 50/50 coin flip ekle |

### [FIFA — Knockout Stage Match Schedule](https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/knockout-stage-match-schedule-bracket)
- **Kaynak:** FIFA resmi
- **Notlar:** R32 tam pairing tablosu; simulate.py fix için kaynak

---

## F. Kullanıcı Ekledi — Analiz Sonuçları

### [Gradient Boosting + Football Rating System (2022)](https://openaccess.cms-conferences.org/publications/book/978-1-958651-37-7/article/978-1-958651-37-7_9)

- **Kaynak:** CMS Conference Proceedings (2022)
- **Yöntem:** Pi-rating + Elo → GBM / XGBoost / LGBM / CatBoost feature olarak kullanır
- **Dataset:** 216,743 maç, 18 sezon (2001–2018), 35 ülke
- **Sonuçlar:** CatBoost + pi-rating en iyi; XGBoost+pi RPS=0.2063, acc=%52.4
- **Bizimle alakası:** **Doğrudan yol haritası.** Pi-rating ekle → XGBoost/CatBoost 3. ensemble üyesi yap → hedef RPS <0.20
- **Not:** 403 aldık ama openaccess mirror var; fetch denenebilir

### [MDPI Energies 18(6):1432](https://www.mdpi.com/1996-1073/18/6/1432)

- **Kaynak:** MDPI Energies
- **Konu:** Enerji tahmini (seri üretim/zaman serisi) — futbolla doğrudan ilgisiz
- **Bizimle alakası:** Zaman serisi bileşeni varsa (örn. dinamik rating decay) referans alınabilir
- **Karar:** **Düşük öncelik.** Metodoloji futbol tahmininden çok farklı; şimdilik pass

### [Reddit r/algobetting — CatBoost Football Log-Loss vs ROI](https://www.reddit.com/r/algobetting/comments/1bwh140/catboost_football_predictions_logloss_vs_roi/)

- **Kaynak:** Reddit r/algobetting (pratik uygulama tartışması)
- **Konu:** CatBoost ile futbol tahmini; log-loss vs ROI trade-off
- **Neden önemli:** Kalibrasyon sorunu — iyi log-loss ≠ iyi betting ROI; overconfident model
- **Bizimle alakası:** Biz betting yapmıyoruz ama kalibrasyon sorunu simülasyon için de geçerli; Platt scaling veya isotonic regression eklenebilir
- **Not:** Reddit WebFetch bloklu; içerik search'ten özetlenemedi — sen oku

### [SofaScore Rating Sistemi](https://www.sofascore.com/tr/news/sofascore-rating)

- **Kaynak:** SofaScore
- **Yöntem:** Oyuncu başına 200+ veri noktası; maç başı 6.5 baseline; 2000 iterasyon/maç; gol/asist/hata/kart vs
- **API durumu:** Kapalı/proprietary; resmi API yok; scraping gerektirir
- **Bizimle alakası:** **L scale player-level feature olarak güçlü aday.** Transfermarkt verisi zaten elimizde var — SofaScore aggregate (ortalama rating) eklenebilir
- **Kısıt:** Scraping ToS riski; mevcut `data/raw/transfermarkt/` verileri daha temiz alternatif
- **Karar:** Transfermarkt `appearances.csv` player value kullan önce; SofaScore zor ise bırak

### Sentiment Analysis — Sosyal Medya

- **Kaynak:** HuggingFace (genel NLP rehberi)
- **Fikir:** Maç öncesi Twitter/sosyal medya sentiment → takım moral feature
- **Bizimle alakası:** İlginç fikir ama uluslararası turnuva için pratik değil:
  - Milli takım için consistent sosyal medya verisi toplamak zor
  - Çok dilli (48 ülke farklı dil) sentiment analizi gürültülü
  - Veri toplama pipeline karmaşık, ödül belirsiz
- **Karar:** **Şimdilik pas.** L scale sonrası experimental branch olabilir

---

## Karar Matrisi

| Yöntem | Mevcut | Eklenecek | Öncelik | Kaynak |
|--------|--------|-----------|---------|--------|
| Elo | S ✓ M ✓ | — | — | Hvattum 2010 |
| Pi-ratings | — | M feature olarak dene | **Yüksek** | Constantinou 2012; GBT 2022 |
| Dixon-Coles (τ) | M ✓ (?) | τ var mı doğrula | **Yüksek** | Dixon 1997 |
| DC time-weighting (ξ) | M ✓ (?) | decay parametresi kontrol | Orta | Dixon 1997 |
| XGBoost 3. üye | — | M ensemble genişlet | Orta | Yeung 2023 |
| DIBP (draw inflation) | — | L scale veya M+ | Orta | footBayes/Egidi |
| Bayesian hier. Poisson | — | L scale | Düşük (şimdi) | Baio 2010 |
| Autoregressive priors | — | L scale | Düşük (şimdi) | footBayes 2022 |
| Player-level features | — | L scale | Düşük (şimdi) | — |
| Real bracket seeding | — | simulate.py | Orta | — |

## Acil Kontrol (M implementasyonu)

1. `src/poisson.py` — τ (rho) düzeltmesi var mı? Dixon-Coles'un ana katkısı bu.
2. `src/ml.py` — DC zaman ağırlıklandırması (ξ decay) var mı?
3. Eğer yoksa: M'in "Dixon-Coles" dediği şey aslında sadece bağımsız Poisson olabilir.
