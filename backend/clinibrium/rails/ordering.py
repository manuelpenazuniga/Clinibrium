"""Escala ordenada de urgencia (pura, sin I/O).

`inmediata > prioritaria > ambulatoria`. Usado por `rails/engine.py` para
componer contribuciones de urgencia con monotonía garantizada.
"""
from __future__ import annotations

from clinibrium.contracts.enums import Urgency

_URGENCY_RANK: dict[Urgency, int] = {
    Urgency.inmediata: 0,
    Urgency.prioritaria: 1,
    Urgency.ambulatoria: 2,
}


def urgency_max(a: Urgency, b: Urgency) -> Urgency:
    """Devuelve la urgencia MÁS ALTA (menor rank numérico) entre `a` y `b`.

    La escala es `inmediata (0) > prioritaria (1) > ambulatoria (2)`.
    """
    return a if _URGENCY_RANK[a] <= _URGENCY_RANK[b] else b
