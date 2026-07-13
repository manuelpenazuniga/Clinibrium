"""`POST /api/decision` endpoint — recorded human intervention (AD-4).

Records the physician's decision (accept/reject) on a previous evaluation,
emitting an AuditEvent of type ``clinician_decision`` with actor=clinician.
This is the meaningful human intervention required by Law 21.719.

Does NOT affect the evaluation pipeline — it is a separate follow-up action.
Emitting ITS own AuditEvent does NOT violate INV-4 (which is per-evaluation).
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from clinibrium.audit import emit_decision
from clinibrium.contracts.audit import AuditEvent

router = APIRouter()


class DecisionRequest(BaseModel):
    audit_event_id: str
    decision: str
    reason: str | None = None


@router.post("/api/decision", response_model=AuditEvent)
async def decision_endpoint(body: DecisionRequest) -> AuditEvent:
    """Records the clinical decision on a previous evaluation.

    Body:
        audit_event_id: ID of the AuditEvent of the original evaluation.
        decision: "accept" | "reject".
        reason: clinical justification (optional, de-identified free text).

    Emits a ``clinician_decision`` AuditEvent with actor=clinician,
    persisted and returned to the frontend.
    """
    decision = body.decision.lower()
    if decision not in ("accept", "reject"):
        raise HTTPException(
            status_code=422,
            detail=f"decision must be 'accept' or 'reject', got '{body.decision}'",
        )

    return await emit_decision(
        audit_event_id=body.audit_event_id,
        decision=decision,
        reason=body.reason,
    )
