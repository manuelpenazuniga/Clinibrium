"""`POST /api/decision` endpoint — recorded human intervention (AD-4).

Records the physician's decision (accept/reject) on a previous evaluation,
emitting an AuditEvent of type ``clinician_decision`` with actor=clinician.
This is the meaningful human intervention required by Law 21.719.

Does NOT affect the evaluation pipeline — it is a separate follow-up action.
Emitting ITS own AuditEvent does NOT violate INV-4 (which is per-evaluation).
"""
from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from clinibrium.audit import emit_decision
from clinibrium.contracts.audit import AuditEvent

router = APIRouter()

# P1.3 — fail-closed PII guard on the free-text ``reason``.
# The ``reason`` is persisted in the AuditEvent; a clinician could accidentally
# type an identifier (RUT, name+id, email, phone). README/INV-2 claim that PII
# never crosses the network, so we REJECT reasons that look like they carry PII
# (fail-closed: when in doubt, refuse) and cap the length.
_MAX_REASON_LEN = 280
_PII_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b\d{1,2}[.\s]?\d{3}[.\s]?\d{3}\s?-\s?[\dkK]\b"),  # Chilean RUT
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),                    # email
    re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)"),                       # phone-like
    re.compile(r"\b\d{6,}\b"),                                     # long digit run (IDs/records)
)


def _reject_if_pii(reason: str | None) -> None:
    if reason is None:
        return
    if len(reason) > _MAX_REASON_LEN:
        raise HTTPException(
            status_code=422,
            detail=(
                f"reason too long (max {_MAX_REASON_LEN} chars) — "
                "keep it a short, de-identified note"
            ),
        )
    for pattern in _PII_PATTERNS:
        if pattern.search(reason):
            raise HTTPException(
                status_code=422,
                detail=(
                    "reason appears to contain PII (id/RUT/email/phone). De-identify it — "
                    "PII must never cross the network (INV-2)."
                ),
            )


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

    _reject_if_pii(body.reason)  # P1.3: fail-closed — no PII in the persisted reason

    return await emit_decision(
        audit_event_id=body.audit_event_id,
        decision=decision,
        reason=body.reason,
    )
