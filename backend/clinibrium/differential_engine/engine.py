"""DifferentialEngine — deterministic, pure scoring.

Takes `CaseFeatures` and returns a `DifferentialResult` with candidates
sorted desc by score (0..1). It is a PURE function:

  - same features ⇒ same result (same lists, same order),
  - no I/O, no clock, no LLM, no randomness,
  - no coupling to `redflag_engine` / `reasoner` / `ml_client` /
    `orchestrator` (INV-5: regulatory separation),
  - it does not detect red flags, does not set urgency and does not
    recommend treatment (that is RedFlagEngine + rails + reasoner;
    INV-1, INV-3).
"""
from __future__ import annotations

from clinibrium.contracts.enums import Diagnosis
from clinibrium.contracts.features import CaseFeatures
from clinibrium.contracts.results import DifferentialCandidate, DifferentialResult
from clinibrium.differential_engine.criteria import CRITERIA, DiagnosisCriterion

# Precomputed at import time: `Diagnosis → total possible weight`. It is
# deterministic (derives 1:1 from CRITERIA) and leaves evaluation as a
# single linear pass per diagnosis.
_MAX_POSSIBLE_BY_DIAGNOSIS: dict[Diagnosis, float] = {}


def _build_max_possible() -> dict[Diagnosis, float]:
    out: dict[Diagnosis, float] = {}
    for c in CRITERIA:
        out[c.diagnosis] = out.get(c.diagnosis, 0.0) + c.weight
    return out


_MAX_POSSIBLE_BY_DIAGNOSIS = _build_max_possible()


def _score_one(
    diagnosis: Diagnosis,
    features: CaseFeatures,
    criteria: list[DiagnosisCriterion],
) -> DifferentialCandidate | None:
    """Sums the weights of matched criteria and normalizes by the total
    possible for that diagnosis. Returns `None` if nothing matches or
    if the diagnosis has no defined criteria.
    """
    max_possible = _MAX_POSSIBLE_BY_DIAGNOSIS.get(diagnosis, 0.0)
    if max_possible <= 0.0:
        return None

    raw = 0.0
    rule_ids: list[str] = []
    for c in criteria:
        if c.diagnosis != diagnosis:
            continue
        if c.predicate(features):
            raw += c.weight
            rule_ids.append(c.id)

    if raw <= 0.0:
        return None

    return DifferentialCandidate(
        diagnosis=diagnosis,
        score=raw / max_possible,
        rule_ids=rule_ids,
    )


def evaluate(features: CaseFeatures) -> DifferentialResult:
    """Evaluates `features` and returns the prioritized pool of
    differential diagnoses. Same features ⇒ same result, always.

    Conventions:
      - Iterates `Diagnosis` in its definition order (stable) and applies
        `sorted(..., key=-score)` for descending order by score.
        Python `sorted` is stable ⇒ score ties are broken by insertion
        order, which is the enum order.
      - Excludes candidates with `score == 0` (no criterion matched).
    """
    candidates: list[DifferentialCandidate] = []
    for dx in Diagnosis:
        cand = _score_one(dx, features, CRITERIA)
        if cand is not None:
            candidates.append(cand)

    candidates.sort(key=lambda c: -c.score)
    return DifferentialResult(candidates=candidates)
