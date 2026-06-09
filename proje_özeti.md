# 2026 FIFA Dünya Kupası Maç Tahmin Modeli

*Python · Makine Öğrenmesi · İstatistiksel Modelleme*

48 takım, 104 maç kapsayan turnuva için çok katmanlı tahmin sistemi geliştirildi. Model, üç aşamalı ölçek mimarisiyle kurgulandı:

**Elo Derecelendirme Sistemi (S):** 1872'den günümüze uzanan 150+ yıllık uluslararası futbol verisini kronolojik olarak işleyerek takım güçlerini dinamik biçimde hesaplayan Elo algoritması uygulandı. Maç türüne göre değişken K-faktörü ve turnuva tipine özel ağırlıklandırma kullanıldı.

**Dixon-Coles Poisson Modeli (M):** Takım bazlı hücum/savunma parametrelerini maksimum olabilirlik tahminiyle (MLE) öğrenen Dixon-Coles modeli kuruldu. Zaman-azalma ağırlıklandırması (üstel bozunma, ~5 yıl yarı-ömür) ile yakın dönem maçlara daha yüksek ağırlık verildi. Düşük skorlu maçlar için τ (tau) düzeltmesi uygulandı.

**XGBoost Ensemble (M+):** Elo farkı, Pi-rating, FIFA sıralama puanı, son form, gol farkı ortalaması gibi 8+ özellikten oluşan feature pipeline'ı XGBoost modeliyle beslendi. Walk-forward cross-validation ile veri sızıntısı (data leakage) önlendi. Ensemble ağırlığı tek turnuvanın yüksek varyansından kaçınmak için çok-turnuvalı (9 turnuva: WC + Euro + Copa) holdout üzerinde grid-search ile seçildi (≈0.75 Elo / 0.25 XGB).

**Monte Carlo Simülasyonu:** 2026 formatına özgü (12 grup, 32'li eleme) 50.000 tekrarlı turnuva simülasyonu ile her takımın şampiyonluk olasılığı hesaplandı.

**Dürüst sonuç:** Çok-turnuvalı holdout üzerinde tam ML yığını (Elo + Pi + FIFA + Dixon-Coles + XGBoost) sade Elo baseline'ını **anlamlı şekilde geçemedi** — ensemble RPS 0.18938 vs Elo 0.18986 (fark −0.00048, %95 bootstrap güven aralığı sıfırı kesiyor). Tek turnuvalı (WC2022) holdout XGBoost'u öne çıkarsa da, çoklu turnuvada bu avantaj kayboluyor. Model Elo'yu **eşitliyor**, geçmiyor; raporlanan sonuç bu dürüst null'dır.

**Değerlendirme:** Futbol tahminine uygun sıralı metrik olan Ranked Probability Score (RPS) birincil metrik olarak kullanıldı; log-loss ve Brier skoru yardımcı metrik olarak raporlandı. Bootstrap güven aralıkları, eşleştirilmiş anlamlılık testi (model A, B'yi geçiyor mu?) ve kalibrasyon (ECE) eklendi. Çıktılar bahis piyasası outright oranlarıyla (de-vig) karşılaştırıldı.

**Araçlar:** Python, NumPy, Pandas, SciPy (optimizasyon), XGBoost, scikit-learn, Kaggle dataset
