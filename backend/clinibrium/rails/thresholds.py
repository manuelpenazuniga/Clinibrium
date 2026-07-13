"""Provisional clinical constants for the rails (FAIL-SAFE).

GOLDEN RULE: when in doubt, the constant must push toward MORE safety
(ESCALAR/BLOQUEAR), never less.  All values are provisional and carry
`# TODO(clinical)` — they must be validated by the superspecialist.
"""
from __future__ import annotations

# TODO(clinical): below this threshold Epley is NOT recommended (not confidently posterior BPPV)
BPPV_EPLEY_CONFIDENCE_FLOOR: float = 0.6

# TODO(clinical): differential top score below this ⇒ uncertainty ⇒ ESCALAR
DIFFERENTIAL_UNCERTAINTY_FLOOR: float = 0.4

# TODO(clinical): if top1 and top2 differ by less than this ⇒ ambiguous ⇒ ESCALAR
AMBIGUITY_EPSILON: float = 0.1
