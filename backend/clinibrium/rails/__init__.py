"""Hard invariants applied AFTER Claude; they always win."""
from __future__ import annotations

from clinibrium.rails.engine import apply_rails
from clinibrium.rails.ordering import urgency_max
from clinibrium.rails.thresholds import (
    AMBIGUITY_EPSILON,
    BPPV_EPLEY_CONFIDENCE_FLOOR,
    DIFFERENTIAL_UNCERTAINTY_FLOOR,
)

__all__ = [
    "AMBIGUITY_EPSILON",
    "BPPV_EPLEY_CONFIDENCE_FLOOR",
    "DIFFERENTIAL_UNCERTAINTY_FLOOR",
    "apply_rails",
    "urgency_max",
]
