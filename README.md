# 2026 FIFA Dünya Kupası — Tahmin Modeli ve Dürüst Bir Backtest

48 takım, 104 maç, 12 grup, yeni 32'li eleme formatı için uçtan uca bir maç/turnuva
tahmin pipeline'ı. **Ama asıl soru ve asıl bulgu şu:**

> **Zengin özellikli bir makine öğrenmesi modeli, uluslararası futbolda sade bir Elo
> taban çizgisini gerçekten geçebiliyor mu?**

Cevap, 9 turnuvalık titiz bir walk-forward backtest'te ölçüldü ve dürüstçe raporlanıyor.

---

## Ana sonuç (TL;DR)

9 turnuvalık holdout (WC 2014/18/22, Euro 2016/21/24, Copa 2019/21/24 — **431 maç**),
metrik **RPS** (düşük = iyi), %95 güven aralıkları **bootstrap** (2000 yeniden örnekleme):

| Model | RPS | %95 GA |
|---|---|---|
| **Sade Elo** (taban) | 0.18986 | [0.17656, 0.20383] |
| Yalnız XGBoost | 0.19330 | [0.17885, 0.20849] |
| **Ensemble** (Elo+Pi+FIFA+DC+kadro değeri → XGB, ağırlıklı) | **0.18938** | [0.17613, 0.20333] |

- **Ensemble − Elo farkı = −0.00048**, GA = [−0.00207, +0.00104] → **0'ı kesiyor.**
- **Karar: model Elo'ya istatistiksel olarak BERABERE — anlamlı geçemiyor** (P(ensemble daha iyi)=%73).
- Yalnız XGBoost aslında Elo'dan **biraz daha kötü**. Ensemble ancak **%74 Elo + %26 XGB**
  karışımıyla Elo'ya yetişiyor (`w_elo=0.74`, grid-search).

**Bu başarısızlık değil, bulgudur:** uluslararası futbolda takım gücü zaten büyük ölçüde
Elo'da özetlenmiş. Pi-ratings, FIFA puanı, Dixon-Coles, **kadro piyasa değeri** ve **kadro
yaşı** + ayarlı XGBoost'un Elo üstüne kattığı ölçülebilir sinyal ≈ 0. "Yarım sayfa Elo,
koca ML yığınına denk" — savunulabilir, gerçek bir sonuç.

---

## Piyasa kıyası (2026 şampiyonluk oranları)

Model `P_Champion` dağılımı, yayınlanan bahis **outright** oranlarının (BetMGM, 1 Haz 2026,
de-vig sonrası) ima ettiği olasılıklarla kıyaslandı (`outputs/model_vs_market_2026.csv`):

- **Spearman(model, piyasa) = 0.890** → sıralama olarak güçlü uyum.
- **Yoğunlaşma: model ilk-3'e %56.8 yığıyor, piyasa %38.4.** Model favorilere piyasadan
  daha emin.
- En büyük ayrışmalar:
  - Model **yüksek**: 🇪🇸 İspanya +10.6 puan (piyasanın 1.76×), 🇦🇷 Arjantin +7.2 puan (1.84×)
  - Model **düşük**: 🇵🇹 Portekiz −3.5, 🇳🇴 Norveç −2.1, 🏴 İngiltere −2.0, 🇺🇸 ABD −1.8, 🇩🇪 Almanya −1.4

> Outright piyasa n=1: gerçek not ancak 19 Tem 2026'da. Turnuva öncesi bu BETİMLEYİCİdir
> (kim nerede ayrışıyor), kazanç/kayıp değil. Tahminler `outputs/frozen/` altında dondurulup
> tarih + git SHA ile **pre-register** edildi → turnuva boyunca canlı notlanabilir.

## Kalibrasyon

Maç bazında **ECE = 0.0255** (`outputs/calibration_Mplus.csv`) → olasılıklar **iyi kalibre**.
Şampiyonluktaki aşırı yoğunlaşma maç kalibrasyonundan değil, **Monte Carlo turnuva
simülasyonunun** küçük maç-avantajlarını 7 turda üst üste bindirmesinden geliyor.

## Şampiyonluk olasılıkları (M+, 50.000 MC)

| # | Takım | Şampiyon | Final | Yarı final |
|---|---|---|---|---|
| 1 | İspanya | %24.6 | %36.7 | %50.4 |
| 2 | Fransa | %16.3 | %28.8 | %42.0 |
| 3 | Arjantin | %15.8 | %25.6 | %43.6 |
| 4 | İngiltere | %8.2 | %16.3 | %30.0 |
| 5 | Brezilya | %7.8 | %16.1 | %29.2 |

Tam liste: `outputs/champion_probs_Mplus.csv`.

## Sade Elo mu, tam model mi? (kıyas)

"Karmaşık model Elo'ya berabere kaldıysa, sadece Elo kullansak daha mı doğru olur?" —
ölçtük. Üç varyant aynı bracket + 50.000 MC ile (`scripts/compare_models.py`):

| Model | top-3 yoğunlaşma | Spearman (piyasa) | KL→piyasa | Maç-bazı RPS |
|---|---|---|---|---|
| Sade Elo (S) | 55.6% | 0.874 | 0.182 | 0.18986 |
| Elo-only + DC goller | 59.9% | 0.899 | 0.193 | 0.18986 |
| Tam M+ | 56.8% | 0.890 | **0.161** | 0.18938 |
| Piyasa | 38.4% | — | — | — |

**Sonuç: Elo-only DAHA doğru değil — EŞİT doğru + daha basit.** Maç-bazı RPS berabere
(0.18986 vs 0.18938, fark anlamsız). Üç model neredeyse özdeş şampiyon dağılımı veriyor;
hepsi piyasadan çok daha yoğun. Tek nüans: M+ piyasaya **marjinal** olarak daha yakın
(KL 0.161) — XGB/kadro Brezilya'yı 4.5%→7.8% (piyasa 8.6%) çekiyor. Ama bu fark gürültü
içinde ve turnuva bitene dek kanıtlanamaz.

**Pratik öneri:** skill berabere olduğundan parsimoni Elo-only'yi savunulabilir kılar;
ama M+ şampiyon dağılımı piyasaya biraz daha yakın. İkisini + piyasayı birlikte raporla
(`outputs/model_comparison.csv`), kararı turnuva versin.

---

## Modeller

- **S** — Elo baseline (turnuvaya göre değişken K). `scripts/run_baseline.py`
- **M** — Elo + Dixon-Coles Poisson ensemble. `scripts/run_m.py`
- **M+** — Elo + XGBoost (Pi-ratings + FIFA + DC + **kadro değeri/yaş** öznitelikleri),
  goller için DC. `scripts/run_m_plus.py`

## Kurulum & çalıştırma

```bash
python -m pip install -r requirements.txt
python scripts/smoke_test.py                                  # hızlı entegrasyon testi

python scripts/run_m_plus.py --n 50000 --jobs 8 --no-friendly         # M+
python scripts/run_m_plus.py --n 50000 --jobs 8 --no-friendly --tune  # + Optuna hiperparametre araması
```

Çıktılar (`outputs/`): `champion_probs_Mplus.csv`, `model_Mplus_meta.csv` (ağırlıklar +
RPS + bootstrap GA + ECE + piyasa metrikleri), `model_vs_market_2026.csv`,
`calibration_Mplus.csv`, `frozen/` (pre-register edilmiş tahminler).

## Veri

`data/raw/` (gitignore). Kaynaklar: Kaggle martj42 `results.csv`, FIFA ranking history,
World Football Elo snapshots, Transfermarkt market value time series, 2026 kura CSV'si,
2026 outright odds snapshot (`data/raw/odds/wc2026_outright_*.csv`).

## Dürüst sınırlamalar

- **Beat-the-baseline hedefine ulaşılamadı** (Elo'ya berabere, anlamlı değil) — gerçekçi
  hedef bunu geçerli bir sonuç sayıyor.
- Kadro değeri ataması **birincil vatandaşlık** ile yapılır; çifte-vatandaş diaspora
  takımları (DR Kongo, Fildişi Sahili) düşük değerlenir. Favoriler doğru kapsanır.
- Kapanış 1X2 oranları diskte yok → maç-bazlı "marketi yen" testi yapılmadı; kıyas
  yalnızca outright şampiyonluk piyasası.
- xG (StatsBomb) sadece turnuva maçlarını kapsar (~%1 kapsama) → eğitime alınmadı.
