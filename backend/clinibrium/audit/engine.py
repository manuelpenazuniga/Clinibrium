"""AuditEvent construction and emission (INV-4: exactly 1 per invocation).

`build_audit_event` is pure (no I/O, no `datetime.now()`).
`emit` composes build + async persistence.
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
    """sha256 hash of the de-identified dict (NO PII).

    Uses `model_dump(mode="json")` with sort_keys for determinism —
    same approach as `build_network_payload` but without the redundant
    allowlist validation (CaseFeatures already has extra=forbid).
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
    lang: str | None = None,
) -> AuditEvent:
    """Builds an immutable AuditEvent. Pure function — no I/O or side effects.

    Args:
        result: sealed PipelineResult (post-rails).
        features: de-identified CaseFeatures of the case.
        reasoner_status: "ok" if the reasoner answered, "degraded" if not.
        outcome: "evaluation" for the normal pipeline, "error" for failure.
        occurred_at: injectable timestamp (NO buried datetime.now()).
        lang: UI language the explanation was requested in (additive metadata;
            does NOT affect any safety decision. The reasoner prose in the FHIR
            ClinicalImpression follows it — see AD-19 precision).
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
        output_lang=lang,
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
    lang: str | None = None,
) -> AuditEvent:
    """Builds the AuditEvent, persists it (best-effort) and returns it.

    INV-4: the event is always built. If persistence fails, it is
    logged and execution continues — the event already exists and is
    traceable.
    """
    event = build_audit_event(
        result,
        features,
        reasoner_status=reasoner_status,
        outcome=outcome,
        occurred_at=occurred_at,
        lang=lang,
    )
    await persist_audit(event)
    return event


async def emit_decision(
    *,
    audit_event_id: str,
    decision: str,
    reason: str | None = None,
    occurred_at: datetime | None = None,
    lang: str | None = None,
) -> AuditEvent:
    """Emits a clinical-decision AuditEvent (AD-4, human intervention).

    Does NOT go through the evaluation pipeline — it is a separate
    subsequent action that records the physician's meaningful
    intervention (Chilean Law 21.719). It emits ITS own AuditEvent
    (does not violate INV-4, which is per-evaluation).
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
        output_lang=lang,
    )

    await persist_audit(event)
    return event
