# LinkedIn Post — 2026 Dünya Kupası: Model vs Elo vs Piyasa

---

🏆 **2026 Dünya Kupası için bir tahmin modeli kurdum. En ilginç kısmı: modelim, 50 satırlık
bir Elo taban çizgisini geçemedi. Ve asıl değerli bulgu tam da bu.**

Çoğu "tahmin modeli" paylaşımı şampiyon listesiyle başlar. Ben tersinden gideceğim —
çünkü dürüst sonuç, parlak tablodan daha öğretici.

**Kurduğum şey:** 48 takım, 104 maç, 12 grup ve yeni 32'li format için uçtan uca pipeline.
Elo + Pi-ratings + FIFA puanları + Dixon-Coles Poisson + **Transfermarkt kadro piyasa değeri**
+ kadro yaşı öznitelikleri → Optuna ile ayarlanmış XGBoost. Goller için Dixon-Coles, turnuva
için 50.000 Monte Carlo (gerçek 2026 eleme bracket'i dahil).

**Sınadığım soru:** Tüm bu makine öğrenmesi yığını, sade Elo'yu *gerçekten* geçiyor mu?

**Cevap (9 turnuvalı walk-forward backtest, 431 maç, RPS — düşük daha iyi):**

| Model | RPS | %95 bootstrap GA |
|-------|-----|------------------|
| Sade Elo (taban) | 0.18986 | [0.1766, 0.2038] |
| Yalnız XGBoost | 0.19330 | [0.1789, 0.2085] |
| **Ensemble (tam yığın)** | **0.18938** | [0.1761, 0.2033] |

Ensemble, Elo'yu yalnızca **0.0005** geçti — ve güven aralığı **0'ı içeriyor.** Yani fark
istatistiksel olarak gürültüden ayırt edilemez. Dahası: yalnız XGBoost, Elo'dan **daha kötü.**
Ensemble ancak **%74 Elo + %26 XGB** karışımıyla başa baş geliyor.

**Karar: Model Elo'ya berabere — anlamlı bir üstünlük yok.** 🤷

**Neden bu bir başarı?** Çünkü ölçtüm, süsleme yapmadım. Ders şu: **uluslararası futbolda
takım gücü zaten Elo'da özetlenmiş.** Kadro değeri Elo ile ~%80+ örtüşür; ekstra öznitelikler
ölçülebilir sinyal katmıyor. Bu, ciddi spor-analitiği literatürüyle uyumlu — ve çoğu kişinin
paylaşmadığı bir gerçek.

**Dış ölçüt — piyasa:** Modeli, yayınlanan bahis şampiyonluk oranlarıyla kıyasladım (de-vig
sonrası). Sıralama uyumu güçlü (**Spearman 0.89**), ama model favorilere piyasadan **daha emin**:
ilk-3'e model %57, piyasa %38 olasılık veriyor. En büyük ayrışmalar — model İspanya'ya piyasanın
**1.76×'i**, Arjantin'e **1.84×'i**; buna karşılık Portekiz, Norveç, İngiltere ve ABD'yi piyasanın
**altında** görüyor. Bunlar ya gerçek "edge" ya da modelin aşırı-güveni — turnuva söyleyecek.

**Kalibrasyon:** Maç bazında ECE = 0.026 → olasılıklar iyi kalibre. Şampiyonluktaki yoğunlaşma
maç hatasından değil, Monte Carlo'nun küçük avantajları 7 turda üst üste bindirmesinden.

**Dürüstlük için:** Tüm tahminleri turnuvadan önce **dondurup tarih + commit ile pre-register
ettim.** Kupa boyunca canlı RPS ile notlanacak. Not edilemeyen tahmin, tahmin değil; hikâyedir.

**Yöntemsel altyapı:** bootstrap güven aralıkları + paired significance testi, walk-forward
(sızıntısız) Elo/Pi/kadro snapshot'ları, RPS-temelli Optuna araması, gerçek kura bracket'i.

Kod açık kaynak. Sonuç: parlak değil, ama **gerçek ve ölçülmüş.**

#MachineLearning #DataScience #FIFA #WorldCup2026 #Python #SportsAnalytics #Statistics

---

**Not:** "Spain'i seçen model" başlığı daha çok tıklanırdı. Ama "zengin model basit taban
çizgisini geçemedi, işte titiz kanıtı" çok daha dürüst bir mühendislik hikâyesi. 😄
