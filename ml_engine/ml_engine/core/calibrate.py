"""Honest calibration (temperature scaling of the FINAL VECTOR) + ECE.

Codex/Gemini fix: the final leaf vector is calibrated (not each node
separately), the temperature is fit by minimizing NLL on a SEPARATE
calibration split (no leakage), and ``ECE_after ≤ ECE_before`` is NOT imposed
as an invariant (temperature minimizes NLL, not ECE; using it as a test would
be leakage). ECE is REPORTED on untouched test data, not asserted as an
improvement.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar


def _matrix(probs: list[dict[str, float]], leaves: tuple[str, ...]) -> np.ndarray:
    return np.array([[p.get(leaf, 0.0) for leaf in leaves] for p in probs], dtype=float)


@dataclass(frozen=True)
class TemperatureCalibrator:
    leaves: tuple[str, ...]
    temperature: float = 1.0

    @classmethod
    def fit(
        cls, probs: list[dict[str, float]], y_true: list[str], leaves: tuple[str, ...]
    ) -> TemperatureCalibrator:
        p = np.clip(_matrix(probs, leaves), 1e-12, 1.0)
        log_p = np.log(p)
        y_idx = np.array([leaves.index(y) for y in y_true])
        rows = np.arange(len(y_idx))

        def nll(log_t: float) -> float:
            t = np.exp(log_t)  # T = exp(log_t) > 0
            z = log_p / t
            m = z.max(axis=1, keepdims=True)
            log_z = m[:, 0] + np.log(np.exp(z - m).sum(axis=1))
            ll = z[rows, y_idx] - log_z
            return float(-ll.mean())

        res = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")  # T ∈ [0.05, 20]
        return cls(tuple(leaves), float(np.exp(res.x)))

    def transform_one(self, p: dict[str, float]) -> dict[str, float]:
        v = np.clip(np.array([p.get(leaf, 0.0) for leaf in self.leaves]), 1e-12, 1.0)
        z = np.log(v) / self.temperature
        z -= z.max()
        e = np.exp(z)
        e /= e.sum()
        return {leaf: float(x) for leaf, x in zip(self.leaves, e, strict=True)}

    def transform(self, probs: list[dict[str, float]]) -> list[dict[str, float]]:
        return [self.transform_one(p) for p in probs]


def ece(
    probs: list[dict[str, float]],
    y_true: list[str],
    leaves: tuple[str, ...],
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error (argmax confidence vs accuracy, M bins)."""
    p = _matrix(probs, leaves)
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    y_idx = np.array([leaves.index(y) for y in y_true])
    correct = (pred == y_idx).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(conf)
    total = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi)
        k = int(mask.sum())
        if k:
            total += (k / n) * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)
