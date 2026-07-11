"""Abstención por gate de confianza → ``undetermined`` (INV-10).

Fix Codex/Gemini: UN solo gate de confianza (sin OOD geométrico/Mahalanobis,
que degenera sobre one-hot). El umbral ``τ`` se fija en el split de calibración
para una COBERTURA objetivo y se bloquea. Al abstenerse, ``undetermined=1.0`` es
un CENTINELA operacional (no una probabilidad calibrada); las 8 hojas van a 0.

La abstención es EVIDENCIA para el reasoner/usuario; el ML NO escala urgencia
por sí mismo (INV-11): quien decide escalar son los rieles deterministas de A.
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
        """Fija τ = cuantil de la confianza (max-prob) tal que se cubre
        ``target_coverage`` de los casos (se abstiene en el (1-cov) menos
        confiable). Se calcula en calibración y se BLOQUEA.
        """
        conf = np.array([max(p.values()) for p in probs], dtype=float)
        q = float(np.clip(1.0 - target_coverage, 0.0, 1.0))
        tau = float(np.quantile(conf, q)) if len(conf) else 0.0
        return cls(threshold=tau, abstain_label=abstain_label)

    def apply(self, p: dict[str, float]) -> dict[str, float]:
        """Devuelve el vector con la clave de abstención añadida (Σ=1)."""
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
