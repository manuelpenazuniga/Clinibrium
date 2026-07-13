"""Ordered urgency scale (pure, no I/O).

`inmediata > prioritaria > ambulatoria`. Used by `rails/engine.py` to
compose urgency contributions with guaranteed monotonicity.
"""
from __future__ import annotations

from clinibrium.contracts.enums import Urgency

_URGENCY_RANK: dict[Urgency, int] = {
    Urgency.inmediata: 0,
    Urgency.prioritaria: 1,
    Urgency.ambulatoria: 2,
}


def urgency_max(a: Urgency, b: Urgency) -> Urgency:
    """Returns the HIGHEST urgency (lowest numeric rank) between `a` and `b`.

    The scale is `inmediata (0) > prioritaria (1) > ambulatoria (2)`.
    """
    return a if _URGENCY_RANK[a] <= _URGENCY_RANK[b] else b
