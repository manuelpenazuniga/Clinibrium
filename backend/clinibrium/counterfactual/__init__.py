"""What Would Change My Mind? — deterministic counterfactual analysis.

Clinical counterfactual explainability: which SINGLE finding would change the
management of this patient? The LLM does NOT decide — the deterministic core
(RedFlagEngine + rails) verifies each counterfactual and Claude (optional)
only explains it (INV-3).
"""
from clinibrium.counterfactual.engine import (
    Counterfactual,
    WhatWouldChangeResult,
    analyze,
)

__all__ = ["Counterfactual", "WhatWouldChangeResult", "analyze"]
