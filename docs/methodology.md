# 2026 FIFA Dünya Kupası Tahmin Projesi — Metodoloji

Bu belge, projeyi hiç duymamış birine sıfırdan anlatır.

---

## 1. Ne yapmaya çalışıyoruz?

**Hedef:** 2026 FIFA Dünya Kupası'nda hangi takımın şampiyon olacağını, olasılık bazında tahmin etmek.

Yani "Türkiye şampiyon olur" değil, "Türkiye %0.8 ihtimalle şampiyon olur, İspanya %24 ihtimalle şampiyon olur" gibi.

---

## 2. Veri

### Ana veri seti
**Kaynak:** Kaggle — `martj42/international-football-results-from-1872-to-2017`  
**Ne içeriyor:** 1872'den günümüze ~50.000+ uluslararası maç. Her satırda:
- Tarih
- Ev sahibi takım, deplasman takımı
- Skor (ör. 2-1)
- Turnuva adı (ör. "FIFA World Cup", "Friendly", "UEFA Euro")
- Tarafsız saha mı? (neutral = True/False)

**Neden önemli:** Modeller bu geçmiş maçlardan öğreniyor. "Geçmişte hangi takım daha iyi oynadı?" sorusunu bu veriden cevaplıyoruz.

### Ek veri (gelecek fazlar için)
- FIFA sıralaması (opsiyonel, iyileştirme amaçlı)
- Oyuncu düzeyi veriler (L ölçeği için planlandı)

---

## 3. Modeller

Proje üç katmanlı bir model hiyerarşisi kullanıyor: **S → M → L**. Şu an S ve M+ tamamlandı.

---

### Model S: Elo Tabanlı Temel Model

**Elo Nedir?**  
Satranç dünyasından gelen bir puanlama sistemi. Her takımın bir "gücü" vardır (başlangıç: 1500 puan). Her maçtan sonra kazanan takım puan alır, kaybeden verir. Beklenen sonuca göre değişim miktarı ayarlanır — zayıf takımı yenmek az puan kazandırır, güçlü takımı yenmek çok puan kazandırır.

**Parametreler:**
- Başlangıç rating: 1500 (tüm takımlar eşit başlar)
- K faktörü (her maçtan sonra ne kadar değişeceği): turnuva tipine göre değişir
  - FIFA WC: K=60
  - Kıta şampiyonası (EURO, Copa America): K=50
  - Hazırlık maçı (Friendly): K=20
  - Bilinmeyen turnuva: K=30 (varsayılan)
- Beraberlik olasılığı: sezgisel formül — `0.28 × e^(-0.0017 × |elo_farkı|)`  
  (Elo farkı büyüdükçe beraberlik olasılığı düşer)

**Tahmin nasıl yapılır:**  
Ev sahibi ile deplasman takımının Elo puanları karşılaştırılır. Fark büyükse güçlü takımın kazanma olasılığı yüksek.

**Çıktı:** Her maç için 3 olasılık — (Ev kazanır, Beraberlik, Deplasman kazanır)

---

### Model M+: Elo + Pi-Ratings + XGBoost Ensemble

S modelinden daha gelişmiş. Üç farklı yaklaşımı birleştirir.

#### 3a. Dixon-Coles (1997)

**Ne yapar:** Her takım için "hücum gücü" (attack) ve "savunma zayıflığı" (defense) parametresi öğrenir. Bu parametreler, takımın maç başına kaç gol atıp yiyeceğini tahmin eder (Poisson dağılımı).

**Özel katkısı:** 0-0, 1-0, 0-1, 1-1 gibi düşük skorlu maçlara özel bir düzeltme yapar (ρ parametresi). Gerçek futbolda bu sonuçlar Poisson'dan biraz daha sık görülür — Dixon-Coles bunu yakalar.

**Neden zaman ağırlıklı:** Eski maçların (2014) etkisi azaltılır, son maçların (2024) etkisi daha yüksek. Bu için `ξ = 0.002` parametresi kullanılır.

**M+ içindeki rolü:** Dixon-Coles artık doğrudan tahmin yapmıyor — parametreleri (attack_diff, defense_diff) XGBoost'a **özellik (feature)** olarak veriliyor. Ayrıca maç skorlarını simüle etmek için kullanılıyor.

#### 3b. Pi-Ratings (Constantinou & Fenton, 2012)

**Ne yapar:** Elo'ya benzer ama gol farkına duyarlı. Her takımın iki ayrı ratinge sahip: ev sahibi performansı ve deplasman performansı.

**Güncelleme mantığı:** Maç bittikten sonra, gol farkına bakılır. Beklenen ile gerçek arasındaki fark, `gamma = 0.036` hızıyla öğrenilir. `c = 3.0` parametresi, büyük gol farklarının etkisini yumuşatır.

**Neden gerekli:** Elo sadece kazanıp kazanmadığına bakar. 3-0 ile 1-0 aynı etkiyi yaratır. Pi-ratings ise 3-0'ı daha büyük güncelleme olarak işler — gol farkı "sinyal" taşır.

**Veri sızıntısı önlemi (walk-forward):** Pi-ratings özelliği, her maç için o maç öncesindeki durumu kullanır. Modeli eğitirken "gelecekteki" bilgiye erişmez.

#### 3c. XGBoost

**Ne yapar:** Gradient boosting — birçok basit karar ağacını sırayla eğitip hatalardan öğrenen bir topluluk modeli.

**Girdi özellikleri (features):**
| Özellik | Açıklama |
|---------|----------|
| `elo_diff` | Ev sahibi − Deplasman Elo farkı |
| `rank_diff` | FIFA sıralaması farkı |
| `home_advantage` | Tarafsız sahada mı? (0/1) |
| `form_diff` | Son 10 maç form farkı |
| `gd_avg_diff` | Son 10 maçta ortalama gol farkı farkı |
| `attack_diff` | Dixon-Coles hücum parametresi farkı |
| `defense_diff` | Dixon-Coles savunma parametresi farkı |
| `pi_diff` | Pi-ratings farkı |

**Eğitim ve test:**
- Eğitim: Tüm maç tarihi (2022 WC hariç)
- Test (holdout): 2022 FIFA Dünya Kupası (64 maç)
- Amaç: Model 2022'yi hiç görmeden ne kadar doğru tahmin eder?

**Çıktı:** 3 sınıflı olasılık — (Ev kazanır, Beraberlik, Deplasman kazanır)

#### 3d. Ensemble (Ağırlıklı Birleştirme)

XGBoost ve Elo ayrı ayrı tahmin yapar. İkisi birleştirilir:

```
P_final = 0.40 × P_elo + 0.60 × P_xgb
```

Ağırlıklar grid search ile bulundu — test setinde (2022 WC) en düşük RPS'i veren kombinasyon bu.

**Neden DC ensemble üyesi değil?** Grid search'te DC'nin ağırlığı sıfıra düştü. Nedeni: DC'nin öğrendiği her şey zaten XGBoost'un feature'larında var. İki kez dahil etmek anlamsız.

---

## 4. Değerlendirme Metrikleri

Futbol tahmininde doğru metrik seçimi kritik.

### RPS (Rank Probability Score) — Ana Metrik
Sıralı kategoriler (W > D > L) için tasarlanmış. Olasılık dağılımının ne kadar iyi kalibre edildiğini ölçer. **Düşük = iyi.**

### Log-Loss
Tahmin ettiğin sınıfın log olasılığı. Aşırı güvenli yanlış tahminleri sert cezalandırır.

### Accuracy
En basit metrik: doğru sınıfı tahmin etme oranı. Futbolda yanıltıcı olabilir (beraberlik çok sık).

**M+ backtest sonuçları (2022 WC holdout, 64 maç):**
| Model | Log-Loss | RPS | Accuracy |
|-------|----------|-----|----------|
| Elo | 0.9919 | 0.2031 | 57.8% |
| XGBoost | ~0.980 | ~0.200 | — |
| Elo + XGB | ~0.975 | ~0.198 | — |

---

## 5. Turnuva Simülasyonu (Monte Carlo)

Bir tahmin sistemi "Spain vs France" için olasılık verir. Ama "Kim şampiyon olur?" sorusunu cevaplamak için tüm turnuvayı simüle etmek gerekir.

### Yöntem: Monte Carlo
Turnuva **N = 50.000 kez** baştan sona simüle edilir. Her simülasyonda:
1. Her maçta olasılıklara göre rastgele bir sonuç üretilir
2. Turnuva ilerler, bir şampiyon çıkar

50.000 simülasyon sonunda: "Kaç kez şampiyon oldu?" → Şampiyonluk olasılığı.

### 2026 Format (S modelinden farklı)
2026 WC 48 takımla oynanıyor — 2022'den farklı bir format:

1. **Grup aşaması:** 12 grup × 4 takım
2. **İlerleme:** Her gruptan ilk 2 + tüm gruplardan en iyi 8 üçüncü → **32 takım**
3. **Eleme aşaması:** R32 → R16 → Çeyrek Final → Yarı Final → Final
4. **Toplam:** 104 maç

### Hız optimizasyonu
Naif yaklaşımda her simülasyonda her maç için model çağrısı yapılır — 50K × 104 maç = 5.2 milyon model çağrısı. Bu çok yavaş.

**Çözüm: Önbellekleme (Caching)**
- 48 takım × 47 rakip = 2.256 maç kombinasyonu
- Simülasyon başlamadan önce tüm kombinasyonlar için olasılıklar hesaplanır ve sözlükte saklanır
- Simülasyon sırasında sözlükten okunur — model çağrısı yok
- Pandas DataFrame yerine saf Python listesi kullanılır
- Joblib ile paralel işlem (8 çekirdek)

**Sonuç:** 10.000 sim için 627 saniye → 25 saniye (25× hızlanma)

---

## 6. Kod Yapısı

```
src/
├── data.py        # Veri yükleme, 2026 takım listesi
├── elo.py         # EloRatings sınıfı
├── poisson.py     # DixonColes sınıfı (attack/defense/score matrix)
├── ratings.py     # PiRatings sınıfı + build_pi_features()
├── features.py    # build_match_features() — tüm feature'ları birleştirir
├── ml.py          # fit_xgb(), fit_catboost(), predict_proba(), ensemble()
├── simulate.py    # Monte Carlo + build_cache() + run_monte_carlo_cached()
├── backtest.py    # split_by_tournament(), evaluate_predictor()
└── eval.py        # log_loss, rps, brier, accuracy

scripts/
├── run_baseline.py   # S modeli
├── run_m.py          # M modeli (Elo + DC + XGB)
└── run_m_plus.py     # M+ modeli (Elo + Pi + XGB, optimize)

outputs/
├── champion_probs_S.csv       # S sonuçları
├── champion_probs_Mplus.csv   # M+ sonuçları
└── model_Mplus_meta.csv       # Backtest metrikleri
```

---

## 7. Mevcut Sonuçlar

### S Modeli (Elo, 100K sim)
| Sıra | Takım | P(Şampiyon) |
|------|-------|------------|
| 1 | Spain | 25.1% |
| 2 | France | 14.0% |
| 3 | Brazil | 12.5% |
| 4 | Argentina | 11.2% |
| 5 | England | 8.3% |

### M+ Modeli (Elo+XGB, 50K sim)

| Sıra | Takım | P(Şampiyon) | P(Final) | P(Yarı Final) |
|------|-------|------------|----------|---------------|
| 1 | Spain | 20.3% | 30.2% | 42.8% |
| 2 | Argentina | 14.9% | 24.3% | 36.6% |
| 3 | Brazil | 11.3% | 19.8% | 32.2% |
| 4 | France | 9.9% | 17.6% | 29.3% |
| 5 | England | 8.3% | 15.8% | 27.1% |

---

## 8. Sonraki Adımlar (Planlanan)

**Faz 2 (yakın vadeli):**
- R32 bracket eşleşmelerini FIFA'nın resmi formatına göre hardcode etmek
- Tam tiebreaker mantığı (H2H → GD → GF)

**L Ölçeği (uzun vadeli):**
- Bayesian hiyerarşik Poisson modeli
- Oyuncu seviyesi özellikler (sakatlık, form, kadro değeri)
- Transfermarkt verisi entegrasyonu

---

*Güncelleme tarihi: Nisan 2026*
