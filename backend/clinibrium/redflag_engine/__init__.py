"""RedFlagEngine determinista — ¿es emergencia? (separado por régimen regulatorio).

API pública:
  - `evaluate(features)` → `RedFlagResult`
  - `RULES` (tabla de reglas, editable por el clínico validador)
  - `RedFlagRule`, `AGE_CENTRAL_THRESHOLD`

INV-5: este paquete SOLO importa `contracts`. Nunca `differential_engine`,
`reasoner`, `ml_client` ni `orchestrator`. Su veredicto no puede ser
anulado por nadie aguas abajo.
"""
from __future__ import annotations

from clinibrium.redflag_engine.engine import evaluate
from clinibrium.redflag_engine.rules import (
    AGE_CENTRAL_THRESHOLD,
    RULES,
    RedFlagRule,
)

__all__ = [
    "AGE_CENTRAL_THRESHOLD",
    "RULES",
    "RedFlagRule",
    "evaluate",
]
