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
