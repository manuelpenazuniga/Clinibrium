"""`Grounding` interface + `GroundingChunk` type.

AD-10: the reasoner consumes grounding **via an interface** (not pgvector
directly). This lets the implementation degrade gracefully to
`InlineGrounding` when pgvector is unavailable, without the reasoner
having to know the implementation detail.

Design rules:

- `GroundingChunk.text` is always the team's **own paraphrase**
  (AD-5). Verbatim ICVD text is not accepted.
- `GroundingChunk.diagnosis` is optional: a chunk can be generic
  (e.g. "red flags in AVS") or specific to a diagnosis.
- `GroundingChunk.source_id` identifies the chunk for traceability
  (AuditEvent, debugging, retrieval evaluation). Convention:
  `clinibrium-paraphrase:<diagnosis>-<n>`.
- `Grounding.retrieve(...)` is **deterministic** for the same
  `(candidates, features, k)`. No randomness, no clock.
"""
from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict

from clinibrium.contracts import CaseFeatures, Diagnosis, DifferentialResult


class GroundingChunk(BaseModel):
    """Clinical criterion snippet (own paraphrase) consumed by the
    reasoner as context.

    Attributes:
        text:        The team's ORIGINAL paraphrase (NOT verbatim ICVD).
        diagnosis:   Diagnosis it applies to (None = generic / cross-cutting).
        source_id:   Traceable chunk identifier. Convention:
                     `clinibrium-paraphrase:<diagnosis>-<n>`.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    diagnosis: Diagnosis | None = None
    source_id: str


class Grounding(Protocol):
    """Protocol for criteria chunk retrieval.

    An implementation (`InlineGrounding`, `PgvectorGrounding`) returns
    up to `k` chunks relevant to the candidate diagnoses of a
    `DifferentialResult`, optionally weighted by the features present
    in the case.
    """

    def retrieve(
        self,
        candidates: DifferentialResult,
        features: CaseFeatures,
        k: int = 4,
    ) -> list[GroundingChunk]:
        ...
