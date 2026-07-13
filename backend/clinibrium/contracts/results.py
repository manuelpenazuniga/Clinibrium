"""Pipeline output contracts (red flag, differential, ML, reasoner).

Leaf: NO logic, NO computation — data shapes only. No imports outside
the `clinibrium.contracts` package.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from clinibrium.contracts.audit import AuditEvent
from clinibrium.contracts.enums import Diagnosis, ForcedAction, Urgency


class RedFlagHit(BaseModel):
    id: str
    label: str
    forced_actions: list[ForcedAction]
    severity: Literal["high", "medium"]


class RedFlagResult(BaseModel):
    red_flag_activa: bool
    hits: list[RedFlagHit] = []
    forced_actions: set[ForcedAction] = set()


class DifferentialCandidate(BaseModel):
    diagnosis: Diagnosis
    score: float  # range [0.0, 1.0]; convention: sorted desc by score
    rule_ids: list[str] = []


class DifferentialResult(BaseModel):
    # Sorted desc by `score` (producer convention; the validator does not
    # re-sort).
    candidates: list[DifferentialCandidate] = []


class PredictResponse(BaseModel):
    """Contract for the `POST /predict` endpoint (track B — optional ML)."""

    probabilities: dict[str, float]
    shap: dict[str, float] | None = None
    model_version: str


class ReasonerOutput(BaseModel):
    """Reasoner (Claude) output.

    The reasoner EXPLAINS and RECONCILES; it does NOT set binding
    `urgency` or `diagnosis` — those are sealed by RedFlagEngine + rails (INV-1).
    """

    explanation: str
    reconciliation: str
    suggested_next_steps: list[str] = []
    model_used: str
    reasoner_suggested_urgency: Urgency | None = None  # AD-11: structured (enum), not text
    grounding_refs: list[str] = []  # AD-10: ICVD chunk source_ids (provenance for FHIR/frontend)


class PipelineResult(BaseModel):
    case_id: str
    urgency: Urgency
    red_flag: RedFlagResult
    differential: DifferentialResult
    ml: PredictResponse | None = None
    reasoning: ReasonerOutput | None = None
    forced_actions: set[ForcedAction] = set()
    applied_rails: list[str] = []
    audit_event_id: str | None = None
    audit_event: AuditEvent | None = None  # the emitted AuditEvent (for FHIR/frontend)
