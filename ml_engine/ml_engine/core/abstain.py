"""Abstention via confidence gate → ``undetermined`` (INV-10).

Codex/Gemini fix: a SINGLE confidence gate (no geometric/Mahalanobis OOD,
which degenerates on one-hot data). The threshold ``τ`` is set on the
calibration split for a target COVERAGE and then locked. On abstention,
``undetermined=1.0`` is an operational SENTINEL (not a calibrated
probability); the 8 leaves go to 0.

Abstention is EVIDENCE for the reasoner/user; the ML does NOT escalate urgency
on its own (INV-11): the deterministic rails of A decide any escalation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConfidenceGate:
    threshold: float
    abstain_label: str = "undetermined"

    @classmethod
    def fit(
        cls,
        probs: list[dict[str, float]],
        *,
        target_coverage: float = 0.90,
        abstain_label: str = "undetermined",
    ) -> ConfidenceGate:
        """Sets τ = quantile of the confidence (max-prob) such that
        ``target_coverage`` of the cases are covered (abstains on the least
        confident (1-cov)). Computed on calibration and LOCKED.
        """
        conf = np.array([max(p.values()) for p in probs], dtype=float)
        q = float(np.clip(1.0 - target_coverage, 0.0, 1.0))
        tau = float(np.quantile(conf, q)) if len(conf) else 0.0
        return cls(threshold=tau, abstain_label=abstain_label)

    def apply(self, p: dict[str, float]) -> dict[str, float]:
        """Returns the vector with the abstention key added (Σ=1)."""
        conf = max(p.values()) if p else 0.0
        if conf < self.threshold:
            out = {k: 0.0 for k in p}
            out[self.abstain_label] = 1.0
            return out
        out = dict(p)
        out.setdefault(self.abstain_label, 0.0)
        return out

    def abstains(self, p: dict[str, float]) -> bool:
        return (max(p.values()) if p else 0.0) < self.threshold
