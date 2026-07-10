"""Construcción y emisión del AuditEvent (INV-4: exactamente 1 por invocación).

`build_audit_event` es pura (sin I/O, sin `datetime.now()`).
`emit` compone build + persistencia asíncrona.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime

from clinibrium.contracts.audit import AuditEvent
from clinibrium.contracts.enums import ActorType, Urgency
from clinibrium.contracts.features import CaseFeatures
from clinibrium.contracts.results import PipelineResult
from clinibrium.storage.persist import persist_audit


def _features_hash(features: CaseFeatures) -> str:
    """Hash sha256 del dict desidentificado (NO PII).

    Usa `model_dump(mode="json")` con sort_keys para determinismo —
    mismo enfoque que `build_network_payload` pero sin la validación
    redundante del allowlist (CaseFeatures ya tiene extra=forbid).
    """
    payload = features.model_dump(mode="json")
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def build_audit_event(
    result: PipelineResult,
    features: CaseFeatures,
    *,
    reasoner_status: str,
    outcome: str,
    occurred_at: datetime,
) -> AuditEvent:
    """Construye un AuditEvent inmutable. Función pura — sin I/O ni side-effects.

    Args:
        result: PipelineResult sellado (post-rails).
        features: CaseFeatures desidentificadas del caso.
        reasoner_status: "ok" si el razonador respondió, "degraded" si no.
        outcome: "evaluation" para pipeline normal, "error" para fallo.
        occurred_at: timestamp inyectable (NO datetime.now() enterrado).
    """
    return AuditEvent(
        id=str(uuid.uuid4()),
        occurred_at=occurred_at,
        event_type="pipeline_evaluation",
        actor=ActorType.system,
        model_used=result.reasoning.model_used if result.reasoning else None,
        input_features_hash=_features_hash(features),
        urgency=result.urgency,
        forced_actions=list(result.forced_actions),
        red_flag_activa=result.red_flag.red_flag_activa,
        outcome_summary=_build_outcome_summary(result, outcome),
        reasoner_status=reasoner_status,  # type: ignore[arg-type]
        outcome=outcome,
    )


def _build_outcome_summary(result: PipelineResult, outcome: str) -> str:
    if outcome == "error":
        return "Pipeline error — revisión clínica urgente requerida."
    top = result.differential.candidates[0] if result.differential.candidates else None
    dx = top.diagnosis.value if top else "undetermined"
    return (
        f"Top diagnosis: {dx} | urgency: {result.urgency.value} | "
        f"red_flag: {result.red_flag.red_flag_activa} | "
        f"rails: {result.applied_rails}"
    )


async def emit(
    result: PipelineResult,
    features: CaseFeatures,
    *,
    reasoner_status: str,
    outcome: str,
    occurred_at: datetime,
) -> AuditEvent:
    """Construye el AuditEvent, lo persiste (best-effort) y lo devuelve.

    INV-4: el evento se construye siempre. Si la persistencia falla, se
    loguea y se sigue — el evento ya existe y es trazable.
    """
    event = build_audit_event(
        result,
        features,
        reasoner_status=reasoner_status,
        outcome=outcome,
        occurred_at=occurred_at,
    )
    await persist_audit(event)
    return event


async def emit_decision(
    *,
    audit_event_id: str,
    decision: str,
    reason: str | None = None,
    occurred_at: datetime | None = None,
) -> AuditEvent:
    """Emite un AuditEvent de decisión clínica (AD-4, intervención humana).

    NO pasa por el pipeline de evaluación — es una acción posterior
    separada que registra la intervención significativa del médico
    (Ley 21.719).  Emite SU propio AuditEvent (no viola INV-4 que es
    por-evaluación).
    """
    from datetime import datetime as _dt
    from datetime import timezone

    occurred = occurred_at if occurred_at is not None else _dt.now(timezone.utc)

    outcome_text = (
        f"decision={decision}; "
        f"reason={reason or 'n/a'}; "
        f"reference={audit_event_id}"
    )

    event = AuditEvent(
        id=str(uuid.uuid4()),
        occurred_at=occurred,
        event_type="clinician_decision",
        actor=ActorType.clinician,
        model_used=None,
        input_features_hash="",
        urgency=Urgency.ambulatoria,
        forced_actions=[],
        red_flag_activa=False,
        outcome_summary=outcome_text,
        reasoner_status="degraded",
        outcome=decision,
    )

    await persist_audit(event)
    return event
