"""Endpoint `POST /api/decision` — intervención humana registrada (AD-4).

Registra la decisión del médico (aceptar/rechazar) sobre una evaluación
previa, emitiendo un AuditEvent de tipo ``clinician_decision`` con
actor=clinician. Esto es la intervención humana significativa que
requiere la Ley 21.719.

NO afecta el pipeline de evaluación — es una acción posterior separada.
Emitir SU propio AuditEvent NO viola INV-4 (que es por-evaluación).
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
    """Registra la decisión clínica sobre una evaluación previa.

    Body:
        audit_event_id: ID del AuditEvent de la evaluación original.
        decision: "accept" | "reject".
        reason: justificación clínica (opcional, texto libre desidentificado).

    Emite un AuditEvent ``clinician_decision`` con actor=clinician,
    persistido y devuelto al frontend.
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
