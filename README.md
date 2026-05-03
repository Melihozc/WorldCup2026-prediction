# 2026 FIFA Dünya Kupası Tahmin Modeli

48 takım, 104 maç. S/M/L ölçekli tahmin pipeline.

## Kurulum

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

## Veri

`data/raw/` altına Kaggle indir:
- `results.csv` — martj42/international-football-results-from-1872-to-2017
- `fifa_ranking.csv` — cashncarry/fifaworldranking (opsiyonel)

Kullanıcı verisi: `data/raw/user_provided/`.

## Çalıştır (S baseline)

```bash
python scripts/run_baseline.py
```

Çıktı: `outputs/champion_probs_S.csv`.

## Yapı

```
src/
  data.py       # CSV ingest + temizleme
  elo.py        # World Football Elo
  features.py   # form, gol farkı vs.
  simulate.py   # turnuva Monte Carlo
  eval.py       # log-loss, Brier, RPS
notebooks/
  04_model_elo.ipynb
  09_tournament_simulation.ipynb
scripts/
  run_baseline.py
```
