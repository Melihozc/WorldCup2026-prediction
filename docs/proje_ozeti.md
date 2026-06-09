# 2026 FIFA Dünya Kupası Tahmin Projesi — Kapsamlı Özet

---

## Problem: Ne Yapmaya Çalışıyoruz?

2026 FIFA Dünya Kupası 11 Haziran – 19 Temmuz 2026 tarihlerinde ABD, Kanada ve Meksika'da düzenlenecek. **48 takım** katılıyor, **104 maç** oynanacak. Soru basit ama cevabı zor: **Hangi takım şampiyon olur, ve ne ihtimalle?**

Bunu yapmak için bir bilgisayar modeli kuruyoruz. Model tarihsel maç verilerini kullanarak her takımın gücünü ölçüyor, sonra turnuvayı bilgisayarda on binlerce kez simüle ederek olasılıklar üretiyor.

---

## Veri: Ne Kullandık?

### Temel Veri

- **Kaynak:** Kaggle `martj42/international-football-results` — 1872'den günümüze ~50.000+ uluslararası maç
- **İçerik:** Tarih, ev sahibi, deplasman, skor, turnuva adı, tarafsız saha mı

### Ek Veriler (M+ ölçeği için)

| Veri | Kaynak | İçerik |
|------|--------|--------|
| FIFA sıralama tarihi | Kaggle `cashncarry/fifaworldranking` | 1992–2024, aylık puan |
| StatsBomb açık veri | GitHub statsbomb/open-data | WC2022 + Euro2024 maç başı xG (beklenen gol) |
| Transfermarkt | dcaribou/transfermarkt-datasets | Oyuncu profilleri, piyasa değerleri |

---

## Modeller: Nasıl Çalışıyor?

### S Ölçeği — Elo Tabanlı Baseline

**Elo nedir?** Satranç dünyasından gelen bir derecelendirme sistemi. Her takımın bir puanı var (başlangıç: 1500). Kazanırsan rakibinden puan alırsın, kaybedersen verirsin. Güçlü rakibe karşı kazanmak daha çok puan kazandırır.

**Turnuva tipine göre K-faktörü:** Dünya Kupası maçı > UEFA Uluslar Ligi > Hazırlık maçı. Yani önemli maçlar Elo'yu daha çok değiştirir.

**Tahmin:** Elo farkı büyükse güçlü takım kazanır (deterministik değil, olasılıksal). Beraberlik için özel bir formül: `0.28 × exp(-0.0017 × |elo_farkı|)` — iki takım ne kadar eşitse beraberlik o kadar muhtemel.

**Çıktı:** Her maç için `(kazanma%, beraberlik%, kaybetme%)` üçlüsü.

---

### M+ Ölçeği — Makine Öğrenmesi Ensemble

S modeli sadece Elo kullanıyor ve beraberlik olasılığını kabaca tahmin ediyor. M+ buna birçok ek sinyal ekliyor.

#### Dixon-Coles Modeli (DC)

İstatistiksel bir gol modeli. Her takım için ayrı **hücum gücü (α)** ve **savunma gücü (β)** parametresi öğreniyor. Formül:

```
Ev sahibinin gol atma hızı = exp(ev_α - deplasman_β + ev_avantajı)
Deplasman takımının gol atma hızı = exp(deplasman_α - ev_β)
```

Yüksek α → çok gol atar. Yüksek β → az gol yer (iyi savunma). Model ayrıca 0-0, 1-0 gibi düşük skorlara özel düzeltme yapıyor (Dixon & Coles 1997 makalesi).

**Neden önemli?** Elo sadece "kazandı mı kaybetti mi" bakıyor. DC "kaç gol attı/yedi" bakıyor — çok daha fazla bilgi.

#### Pi-Ratings

Gol farkı tabanlı dinamik derecelendirme sistemi (Constantinou & Fenton 2012). Elo'ya benzer ama sadece ev sahibi/deplasman performansını ayrı ayrı takip ediyor.

```
Pi_ev[takım] güncellenir ← (gerçek_gol_farkı - beklenen_fark) × öğrenme_hızı
```

**Neden önemli?** Bazı takımlar evde çok güçlü ama dışarıda zayıf (veya tam tersi). Pi-ratings bunu yakalıyor.

#### FIFA Sıralama Puanları

FIFA her ay takımları puanlar. Hazırlık maçı + güçlü rakip + deplasman galibiyeti = daha çok puan. Walk-forward kullandık: tahmin gününden önceki son sıralamayı alıyoruz (geleceğe bakış yasak).

#### XGBoost (Gradient Boosting)

Yukarıdaki tüm sinyalleri bir araya getiren makine öğrenmesi modeli. Girdi: her maç için şu 11 özellik:

| Özellik | Ne ölçüyor |
|---------|-----------|
| `elo_diff` | Elo puan farkı |
| `rank_diff` | FIFA sıralama farkı |
| `fifa_pts_diff` | FIFA puan farkı |
| `home_advantage` | Tarafsız saha mı? |
| `form_diff` | Son 10 maç form farkı |
| `gd_avg_diff` | Son 10 maçta ortalama gol farkı |
| `attack_diff` | DC hücum farkı |
| `defense_diff` | DC savunma farkı |
| `pi_diff` | Pi-rating farkı |
| `xg_for_diff` | xG farkı (üretilen) |
| `xg_against_diff` | xGA farkı (yenilen) |

XGBoost bu özellikleri kullanarak `0=ev galibi, 1=beraberlik, 2=deplasman galibiyeti` tahmin ediyor.

**Feature importance (hangi özellik ne kadar önemli):**

```
defense_diff    23.3%  ← en önemli
attack_diff     19.7%
gd_avg_diff     13.8%
pi_diff         11.6%
elo_diff        10.2%
home_advantage   9.6%
fifa_pts_diff    8.3%
form_diff        3.5%
rank_diff        0.0%  ← işe yaramıyor
```

**Final ensemble:** Sadece XGBoost kullanıyoruz (`W_ELO=0, W_XGB=1.0`). Çünkü grid-search gösterdi ki Elo eklemek performansı kötüleştiriyor — XGBoost zaten `elo_diff` özelliğini görüyor, Elo'yu ayrıca blend etmek bilgiyi tekrar etmiş oluyor.

---

## Simülasyon: Turnuva Nasıl Simüle Ediliyor?

### 2026 Formatı (Yeni!)

2026 formatı 2022'den farklı:

- **12 grup × 4 takım** = 48 takım
- Her gruptan **ilk 2** doğrudan eleme turu
- **En iyi 8 üçüncü** de eleme turuna giriyor
- Toplam **32 takım** Round of 32'de buluşuyor
- Klasik eleme: R32 → R16 → Çeyrekfinal → Yarıfinal → Final

### Monte Carlo Simülasyonu

Turnuvayı **50.000 kez** baştan sona simüle ediyoruz:

1. Her maç için `(K%, B%, M%)` olasılıklarını hesapla
2. Bu olasılıklara göre rastgele bir sonuç seç
3. Puanları hesapla, grubu sırala
4. Eleme turunda çiftleri eşleştir, devam et
5. Şampiyonu kaydet

50.000 simülasyonun sonunda: "Brazil 9.730 kez şampiyon oldu = %19.5 ihtimal."

**Gol simülasyonu:** Maç sonucu için XGBoost olasılıklarını kullanıyoruz. Skor için Dixon-Coles Poisson modelini kullanıyoruz — gerektiğinde skor ile sonucu tutarlı hale getiriyoruz.

---

## Backtest: Modelin Kalitesi Nasıl Ölçüldü?

### Holdout Yöntemi

2022 Dünya Kupası maçlarını "görmemiş" gibi davrandık:

- **Eğitim:** 2022 öncesi tüm veriler
- **Test:** WC2022'nin 64 maçı

Model test maçlarını hiç görmeden tahmin etti, gerçek sonuçlarla karşılaştırdık.

### RPS (Ranked Probability Score)

Futbol tahmininde en iyi metrik. Olasılık dağılımının gerçeğe ne kadar yakın olduğunu ölçüyor. **Düşük = iyi.**

| Model | RPS |
|-------|-----|
| Elo (S modeli) | 0.2204 |
| **XGBoost (M+ modeli)** | **0.2069** ← %6.1 iyileşme |

---

## Sonuçlar: Final Tahminler

50.000 simülasyon, M+ modeli:

| Takım | R32 | R16 | ÇF | YF | Final | **Şampiyon** |
|-------|-----|-----|----|----|-------|-------------|
| Brazil | 99.4% | 81.5% | 64.6% | 45.9% | 30.2% | **19.5%** |
| Spain | 98.8% | 75.1% | 51.3% | 42.2% | 28.8% | **15.1%** |
| Argentina | 98.1% | 68.4% | 56.5% | 43.2% | 24.5% | **13.7%** |
| France | 96.6% | 78.4% | 48.8% | 36.1% | 21.6% | **12.7%** |
| England | 97.1% | 74.6% | 53.3% | 30.5% | 17.0% | **9.2%** |
| Germany | 99.1% | 74.4% | 40.8% | 25.7% | 12.4% | **6.3%** |
| Portugal | 95.4% | 72.8% | 46.4% | 24.0% | 12.1% | **5.0%** |

**Not:** Brazil'in yüksekliği sorgulandı. Araştırdık: DC modeli Brazil'e tarihsel olarak iyi savunma puanı veriyor (az gol yiyor). Friendly dahil tüm verilerle eğitilen DC bu sinyali güvenilir buluyor. Friendly çıkarıp test ettik — RPS kötüleşti. Yani bu sinyal gerçek, artifact değil.

---

## Aşamalar: Ne Zaman Ne Yaptık?

| Aşama | İş | Durum |
|-------|-----|-------|
| **S modeli** | Elo + MC simülasyon | ✅ Tamamlandı |
| **M modeli** | + Dixon-Coles | ✅ Tamamlandı |
| **M+ modeli** | + Pi-ratings + XGBoost + FIFA rank | ✅ Tamamlandı |
| **StatsBomb xG** | WC2022+Euro2024 event data indir | ✅ 115 dosya cache'lendi |
| **xG kararı** | xG_agg XGBoost feature'ından çıkarıldı | ✅ Prod-only rezerve |
| **Ağırlık grid search** | W_ELO=0 optimal bulundu | ✅ Tamamlandı |
| **DC bias araştırması** | Friendly filtre testi | ✅ Baseline en iyi |
| **Production run** | n=50.000 simülasyon | ✅ Çıktı hazır |
| **L modeli** | Bayesian + oyuncu bazlı | ⏳ Mayıs 2026 kadrolar açıklanınca |

---

## Sonraki Adım

**Mayıs 2026 sonu:** 48 takım resmi kadrosunu açıklayacak. O zaman:

- Her oyuncunun transfermarkt piyasa değeri + yaşı + maç sayısı
- PyMC/NumPyro ile Bayesian hierarchical Poisson modeli
- Takım gücü = ortak latent faktör + oyuncu katkı offset'i

Bu **L modeli** — M+'dan çok daha güçlü olacak ama kadro verisi gelmeden başlanamaz.
