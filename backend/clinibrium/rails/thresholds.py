"""Constantes clínicas provisionales para los rieles (FAIL-SAFE).

REGLA DE ORO: ante duda, la constante debe empujar hacia MÁS seguridad
(ESCALAR/BLOQUEAR), nunca menos.  Todos los valores son provisionales y
llevan `# TODO(clinical)` — deben ser validados por el superespecialista.
"""
from __future__ import annotations

# TODO(clinical): bajo este umbral NO se recomienda Epley (no confidentemente VPPB-P)
BPPV_EPLEY_CONFIDENCE_FLOOR: float = 0.6

# TODO(clinical): top score del differential bajo esto ⇒ incertidumbre ⇒ ESCALAR
DIFFERENTIAL_UNCERTAINTY_FLOOR: float = 0.4

# TODO(clinical): si top1 y top2 difieren en menos que esto ⇒ ambiguo ⇒ ESCALAR
AMBIGUITY_EPSILON: float = 0.1
