# 2026 FIFA Dünya Kupası Tahmin Modeli — Konsensüs-Hibrit Yaklaşım

*Python · İstatistiksel Modelleme · Makine Öğrenmesi · Monte Carlo Simülasyonu*

---

## Özet

Bu çalışma, 2026 FIFA Dünya Kupası (48 takım, 104 maç) için olasılıksal bir tahmin
çerçevesi geliştirir. Dört ölçekli bir mimari kurulmuştur: Elo temelli taban (S),
Dixon-Coles Poisson eklentisi (M), makine öğrenmesi ensemble'ı (M+) ve bahis piyasası
konsensüsünü bir **güç sinyali** olarak modele dahil eden Konsensüs-Hibrit (C). Temel
bulgu iki yönlüdür: (1) tarihsel veriye dayalı zengin makine öğrenmesi yığını, çok-turnuvalı
holdout üzerinde sade Elo tabanını **anlamlı şekilde geçememektedir** (dürüst null sonuç);
(2) bahis piyasası konsensüsü modele dahil edildiğinde, en iyi kalibre edilmiş ve piyasaya
en sadık tahmin **saf konsensüs** ağırlığında elde edilir — tarihsel ve kadro sinyalleri
eklendikçe tahmin piyasadan uzaklaşıp sıralamayı bozar. Model, kesin sonuç veren bir araç
değil, turnuva belirsizliğini nicelleştiren bir çerçeve olarak konumlandırılmıştır.

---

## 1. Giriş

Dünya Kupası gibi kısa süreli ve yüksek belirsizlik içeren turnuvalarda tahmin modelleri
iki soruya odaklanır: tek maçta ne olur ve bu maç sonuçları turnuva boyunca nasıl yayılır.
Literatürdeki yaygın yaklaşım önce maç düzeyinde gol beklentisi veya kazanma olasılığı
üretmek, sonra bunu Monte Carlo simülasyonlarıyla grup ve eleme turlarına taşımaktır
(Gilch & Müller, 2018; Groll vd., 2018). Bu yapı, tek maçları açıklamaktan çok turnuva
genelindeki başarı olasılıklarını üretmek için uygundur.

2026 turnuvası iki yapısal nedenle önceki turnuvalardan daha belirsizdir: takım sayısı
32'den 48'e çıkmış, eleme aşamasına bir tur (Son 32) eklenmiştir. Üçüncü olan 12 takımdan
8'inin elemeye kalması ve grupların eşleştirilmesinde 495 olası permütasyon bulunması,
kura kaynaklı varyansı belirgin biçimde artırır (Zeileis vd., 2026).

Bu çalışmanın katkısı, literatürdeki en güçlü kamuya açık yöntemin (Zeileis vd.'nin
2010'dan beri Dünya Kupası tahmini yapan bahis-konsensüs hibridi) eksik kalan parçasını
mevcut sisteme eklemektir: **bahis piyasası konsensüsünün bir benchmark değil, bir model
girdisi (takım gücü sinyali) olarak kullanılması.**

---

## 2. Literatür ve Yöntem Ailesi

### 2.1 Elo temelli modeller
Elo, takım gücünü tek bir skorla özetler ve bunu maç tahmininde açıklayıcı değişken olarak
kullanır. Gilch & Müller (2018), Elo puanlarını Poisson regresyonuna katıp Monte Carlo ile
2018 Dünya Kupası ilerleme olasılıklarını üretmiştir. Avantajı basitlik, yorumlanabilirlik
ve kolay güncelleme; dezavantajı kadro kalitesi, oyuncu formu veya ekonomik değişkenleri
tek başına yansıtamamasıdır.

### 2.2 Poisson ve Dixon-Coles çerçevesi
Futbol skorları sayma verisi olduğundan Poisson doğal başlangıçtır. Temel Poisson düşük
skorların ve beraberliğin ağırlığını yetersiz modeller; Dixon-Coles (1997) düzeltmesi ve
bivariate Poisson yapıları iki takımın gol üretimini daha gerçekçi modeller. Eleme turlarında
skor dağılımı (uzatma, penaltı, averaj) maç sonucu kadar önemlidir.

### 2.3 Monte Carlo simülasyonu
Maç olasılıkları üretildikten sonra turnuva on binlerce kez tekrarlanarak her takımın
şampiyonluk/yarı-final/Son-16 olasılıkları sayılır. Zeileis vd. (2026) bu mantığı 100.000
simülasyonla uygular; Opta süper-bilgisayarı 25.000 simülasyon kullanır.

### 2.4 Hibrit makine öğrenmesi
Güncel yönelim, klasik istatistiksel modelleri zengin veriyle birleştiren hibrit
sistemlerdir. Zeileis vd. (2026) üç güç sinyalini bir random forest ile birleştirir:
tarihsel maç gücü (zaman-ağırlıklı bivariate Poisson), **bahis konsensüsü gücü** (24
bahisçinin de-vig edilmiş oranları + kurayı düzelten "ters simülasyon") ve güncel durum
(oyuncu ratingleri, piyasa değeri). Groll vd. (2018) random forest'ı sıralama-tabanlı
yetenek parametreleriyle besleyince tahmin gücünün belirgin arttığını gösterir; ancak
baskın özellik daima yetenek parametresidir.

### 2.5 Bu projedeki ölçekler
| Ölçek | Model | Durum |
|---|---|---|
| S | Elo + sezgisel beraberlik | shipped |
| M | Elo + Dixon-Coles ensemble | shipped |
| M+ | Elo + XGBoost + DC + Pi-rating + FIFA + kadro | shipped |
| **C** | **Konsensüs-Hibrit (bu çalışmanın katkısı)** | **shipped** |
| L | Bayesian hiyerarşik Poisson | planlanan |

### 2.6 Konsensüs-Hibrit (C) yöntemi
1. **Çoklu-kitap konsensüs:** her bahisçinin outright oranları ayrı ayrı de-vig edilir
   (overround düzeltmesi), logit ölçeğinde ortalanır (Leitner-Zeileis-Hornik, 2010).
2. **Ters simülasyon:** turnuva ağacı üzerinde, simüle edilen şampiyonluk olasılıkları
   piyasa konsensüsünü yeniden üretecek per-takım güçler aranır (sabit-nokta, damped
   log-oran güncellemesi). Bu adım kolay/zor grup etkisini düzeltir.
3. **Birleştirme:** market gücü, Dixon-Coles gücü (atak−defans) ve kadro değeri z-skorları
   ağırlıklı harmanlanır.
4. **Piyasaya kalibrasyon:** tek boyutlu sıcaklık T ile güçler ölçeklenir; simüle şampiyonluk
   olasılıkları ile piyasa arasındaki kare-fark minimize edilir (overconfidence düzeltmesi).
5. **Monte Carlo:** resmi 2026 Son-32 bracket'i üzerinde simülasyon.

### 2.7 Değerlendirme metrikleri
Birincil metrik **Ranked Probability Score (RPS)** — sıralı W/D/L için uygundur; yardımcı
metrikler log-loss ve Brier. Ek olarak bootstrap güven aralıkları, eşleştirilmiş anlamlılık
testi (`bootstrap_rps_diff`) ve kalibrasyon (ECE). Turnuva düzeyinde model, bahis piyasası
ile karşılaştırılır (KL ıraksaması, Spearman sıra korelasyonu, top-3 olasılık kütlesi).

---

## 3. Bulgular

### 3.1 Dürüst null: tarihsel ML, Elo'yu geçmiyor
Çok-turnuvalı holdout (9 turnuva: Dünya Kupası + Avrupa + Copa) üzerinde tam M+ yığını
(Elo + Pi + FIFA + Dixon-Coles + XGBoost) sade Elo tabanını anlamlı şekilde geçememiştir:

| Model | RPS | Not |
|---|---|---|
| Elo (taban) | 0.18986 | — |
| Ensemble (M+) | 0.18938 | fark −0.00048, %95 bootstrap CI **sıfırı kesiyor** |
| XGBoost (tek başına) | 0.19387 | Elo'dan **kötü** |

Sonuç literatürle tutarlıdır (Gilch & Müller, 2018; çeşitli açık-kaynak modeller): uluslararası
futbol maç sonuçlarında iyi ayarlanmış Elo çok güçlü bir tabandır.

### 3.2 M+ aşırı-güveni (overconfidence)
M+ şampiyonluk dağılımı piyasaya göre aşırı keskindir: ilk 3 takıma atadığı olasılık kütlesi
**%57**, piyasada ise **%38**. Şampiyon olmak için 7 ardışık eleme galibiyeti gerekir; küçük
maç-başı avantaj eleme boyunca **katlanarak** büyür ve favorileri şişirir. Piyasa bu turnuva
varyansını fiyatlar; ham Monte Carlo simülasyonu fiyatlamaz.

### 3.3 Konsensüs-Hibrit ağırlık taraması
Market gücü ile tarihsel+kadro sinyallerinin harmanlanma ağırlığı (`w_market`) tarandı
(geri kalan hist:kadro = 3:1). Üretim koşusu (n=20.000 MC, n_infer=10.000, 22 iterasyon,
friendly hariç):

| w_market | Sıcaklık T | KL→piyasa | Spearman | top-3 kütle | İlk sıra | Yorum |
|---|---|---|---|---|---|---|
| 1.0 (saf konsensüs) | 0.30 | **0.0051** | **0.988** | 0.364 | Spain | piyasaya sadık, en iyi kalibre |
| 0.8 | 0.40 | 0.0331 | 0.977 | 0.407 | Spain | hafif sapma, top-3 hafif şişti |
| 0.6 | 0.40 | 0.1365 | 0.916 | 0.365 | Spain | sıralama bozuluyor (Almanya #2'ye fırlıyor), aşırı düzleşme |

(Piyasa top-3 kütlesi = 0.384.) Eğilim monotondur: bağımsız (tarihsel + kadro) sinyal arttıkça
tahmin piyasadan **uzaklaşır** (KL 0.005 → 0.137), sıra korelasyonu düşer (0.988 → 0.916),
Almanya yükselir / Güney Amerika takımları düşer ve kalibrasyon aşırı düzleştirir. En iyi kalibre
edilmiş, piyasaya en sadık tahmin **saf konsensüstür**.

M+ ile karşılaştırma, kalibrasyon kazanımını net gösterir: M+ piyasaya KL = 0.161 ve top-3
kütle = **0.567** (aşırı-güvenli) iken, Konsensüs-Hibrit (w=1.0) KL = **0.005** ve top-3 kütle
= **0.364** (≈ piyasa 0.384) — overconfidence ortadan kalkar.

### 3.4 Nihai 2026 tahmini (headline = saf konsensüs)

| Sıra | Takım | P_Şampiyon | Piyasa |
|---|---|---|---|
| 1 | İspanya | 13.4% | 14.0% |
| 2 | Fransa | 12.6% | 14.0% |
| 3 | İngiltere | 10.4% | 10.3% |
| 4 | Arjantin | 8.6% | 8.6% |
| 5 | Brezilya | 8.1% | 8.6% |
| 6 | Portekiz | 7.9% | 7.7% |
| 7 | Almanya | 4.8% | 5.1% |
| 8 | Hollanda | 3.8% | 3.7% |

Bu tahmin, gerçek bir izleme geçmişi olan tek kamuya açık yöntemle (Zeileis vd.) aynı
paradigmadadır ve maç-sıralama düzeyinde piyasayla yüksek uyum gösterir (Spearman 0.988).
İnverse-simülasyon 22 iterasyonda yakınsadı (KL 0.568 → 0.003).

---

## 4. Tartışma

İki bağımsız bulgu aynı yöne işaret eder: ne tarihsel makine öğrenmesi (M+) ne de tarihsel +
kadro sinyalleri (C harmanı) **piyasa konsensüsünü geçer**; aksine ondan saparak sıralamayı
bozar. Bu, başarısızlık değil, olgun bir sonuçtur — bahis piyasası mevcut tüm bilgiyi
toplayan çok güçlü bir benchmark'tır ve onu yenmek pratikte çok zordur (Opta'nın kendi
ifadesiyle model "piyasa fiyatları için bir sağlama aracı" olarak en iyi kullanılır).

Katkı, bahis konsensüsünü bir model girdisine dönüştüren ters-simülasyon + kalibrasyon
hattıdır; bu hat M+'nın aşırı-güvenini düzeltir ve piyasa-rekabetçi, kalibre bir 2026
tahmini üretir. Sınırlamalar: outright piyasası n=1'dir (gerçek not ancak 19 Temmuz 2026'da
verilir); kadro değeri sinyali çifte-vatandaş diaspora takımlarını düşük değerler; konsensüs
şu an tek bahisçi snapshot'ına dayanır (çoklu-kitap canlı toplama bir genişleme hedefidir).

Gelecek çalışma: (i) tam Bayesian hiyerarşik bivariate Poisson (L ölçeği) — posterior
belirsizliğini yayarak aşırı-güveni ilkesel biçimde düzeltir, "overshrinkage" için karışım
modeli gerekir (Baio & Blangiardo, 2010); (ii) çoklu-bahisçi canlı oran toplama ile gerçek
konsensüs; (iii) turnuva sonrası canlı puanlama (donmuş tahminler hazır).

---

## Kaynakça

- Baio, G., & Blangiardo, M. (2010). Bayesian hierarchical model for the prediction of
  football results. *Journal of Applied Statistics, 37*(2), 253–264.
- Dixon, M. J., & Coles, S. G. (1997). Modelling association football scores and inefficiencies
  in the football betting market. *Journal of the Royal Statistical Society: Series C, 46*(2), 265–280.
- Gilch, L. A., & Müller, S. (2018). *On Elo based prediction models for the FIFA Worldcup 2018*.
  arXiv:1806.01930.
- Groll, A., Ley, C., Schauberger, G., & Van Eetvelde, H. (2018). *Prediction of the FIFA World
  Cup 2018 — A random forest approach with an emphasis on estimated team ability parameters*.
  arXiv:1806.03208.
- Leitner, C., Zeileis, A., & Hornik, K. (2010). Forecasting sports tournaments by ratings of
  (prob)abilities: A comparison for the EURO 2008. *International Journal of Forecasting, 26*(3),
  471–481.
- Ley, C., Van de Wiele, T., & Van Eetvelde, H. (2019). Ranking soccer teams on the basis of
  their current strength: A comparison of maximum likelihood approaches. *Statistical Modelling,
  19*(1), 55–73.
- Zeileis, A., Hanekov, A., Hvattum, L. M., Michels, R., Schauberger, G., Sukhanova, E., Witte,
  S. (2026). *Football meets machine learning: Forecasting the 2026 FIFA World Cup*.
  https://www.zeileis.org/news/fifa2026/

---

*Üretim çıktıları: `outputs/champion_probs_Consensus.csv` (+ `_w100/_w080/_w060`),
`outputs/model_Consensus_meta.csv`, `outputs/consensus_abilities.csv`. Birim testler:
`tests/test_consensus.py`. Plan: `docs/superpowers/plans/2026-06-09-consensus-hybrid.md`.*
