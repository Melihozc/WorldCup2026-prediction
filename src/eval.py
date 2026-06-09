"""Değerlendirme metrikleri."""
import numpy as np


def log_loss(probs: np.ndarray, outcomes: np.ndarray, eps: float = 1e-12) -> float:
    """probs: (N,3) [W,D,L]. outcomes: (N,) içinde 0=W, 1=D, 2=L."""
    p = np.clip(probs, eps, 1 - eps)
    return -np.mean(np.log(p[np.arange(len(outcomes)), outcomes]))


def brier_multiclass(probs: np.ndarray, outcomes: np.ndarray) -> float:
    n = len(outcomes)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), outcomes] = 1.0
    return np.mean(np.sum((probs - one_hot) ** 2, axis=1))


def rps(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Rank Probability Score — sıralı sınıf (W,D,L) için futbol standardı."""
    n, k = probs.shape
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), outcomes] = 1.0
    cum_p = np.cumsum(probs, axis=1)
    cum_o = np.cumsum(one_hot, axis=1)
    return float(np.mean(np.sum((cum_p - cum_o) ** 2, axis=1) / (k - 1)))


def accuracy(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean(np.argmax(probs, axis=1) == outcomes))


def report(probs: np.ndarray, outcomes: np.ndarray) -> dict:
    return {
        "log_loss": log_loss(probs, outcomes),
        "brier": brier_multiclass(probs, outcomes),
        "rps": rps(probs, outcomes),
        "accuracy": accuracy(probs, outcomes),
        "n": len(outcomes),
    }


def bootstrap_rps_ci(probs: np.ndarray, outcomes: np.ndarray,
                     n_resample: int = 2000, ci: float = 95.0,
                     seed: int = 0) -> dict:
    """RPS nokta tahmini + bootstrap güven aralığı (maçlar üzerinde yeniden örnekleme).

    İki modeli kıyaslarken: A, B'yi 'anlamlı' geçti demek için A'nın CI üst sınırı
    < B'nin nokta RPS'i olmalı (kaba kural). Daha doğrusu paired bootstrap farkı —
    `bootstrap_rps_diff` ona bakar.
    """
    rng = np.random.default_rng(seed)
    n = len(outcomes)
    point = rps(probs, outcomes)
    if n == 0:
        return {"rps": point, "lo": point, "hi": point, "n_resample": 0}
    samples = np.empty(n_resample)
    for i in range(n_resample):
        idx = rng.integers(0, n, n)
        samples[i] = rps(probs[idx], outcomes[idx])
    lo = float(np.percentile(samples, (100 - ci) / 2))
    hi = float(np.percentile(samples, 100 - (100 - ci) / 2))
    return {"rps": float(point), "lo": lo, "hi": hi, "n_resample": n_resample}


def bootstrap_rps_diff(probs_a: np.ndarray, probs_b: np.ndarray,
                       outcomes: np.ndarray, n_resample: int = 2000,
                       ci: float = 95.0, seed: int = 0) -> dict:
    """Paired bootstrap: RPS(A) - RPS(B) farkının dağılımı (aynı maç indeksleri).

    Negatif fark = A daha iyi. CI tamamı 0'ın altındaysa A, B'yi anlamlı geçer.
    p_a_better = farkın < 0 olma oranı (tek taraflı bootstrap p benzeri).
    """
    rng = np.random.default_rng(seed)
    n = len(outcomes)
    point = rps(probs_a, outcomes) - rps(probs_b, outcomes)
    if n == 0:
        return {"diff": point, "lo": point, "hi": point, "p_a_better": 0.5}
    diffs = np.empty(n_resample)
    for i in range(n_resample):
        idx = rng.integers(0, n, n)
        diffs[i] = rps(probs_a[idx], outcomes[idx]) - rps(probs_b[idx], outcomes[idx])
    lo = float(np.percentile(diffs, (100 - ci) / 2))
    hi = float(np.percentile(diffs, 100 - (100 - ci) / 2))
    return {
        "diff": float(point), "lo": lo, "hi": hi,
        "p_a_better": float(np.mean(diffs < 0)),
        "significant": bool(hi < 0.0),
    }


def reliability(probs: np.ndarray, outcomes: np.ndarray, bins: int = 10) -> dict:
    """Top-label kalibrasyon: tahmin edilen sınıfın güveni vs gerçek doğruluk.

    Her maçta confidence = max(probs); correct = argmax==outcome. Güveni
    eşit-genişlikli kovalara böl. ECE = ağırlıklı |acc - conf|.
    Model aşırı yoğunsa (markete kıyasla) ECE büyük + conf>acc çıkar.
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == outcomes).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    ece = 0.0
    n = len(outcomes)
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        in_bin = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        cnt = int(in_bin.sum())
        if cnt == 0:
            rows.append({"bin_lo": lo, "bin_hi": hi, "count": 0,
                         "avg_conf": float("nan"), "accuracy": float("nan")})
            continue
        avg_conf = float(conf[in_bin].mean())
        acc = float(correct[in_bin].mean())
        ece += (cnt / n) * abs(acc - avg_conf)
        rows.append({"bin_lo": float(lo), "bin_hi": float(hi), "count": cnt,
                     "avg_conf": avg_conf, "accuracy": acc})
    return {"ece": float(ece), "bins": rows}
