"""Clinibrium evaluation pipeline — composes all modules.

It is the ONLY module that knows the full graph:
    orchestrator → engines + reasoner + rails + ml_client + audit + grounding + contracts

INV-4: exactly 1 AuditEvent per invocation, even under partial failure
(ml-down + reasoner-down) or unexpected exception. The `audited` flag
guarantees that neither 0 nor 2 events are ever emitted.

INV-6 / INV-8: ml_client and reasoner degrade to None without breaking the
pipeline. Safety (urgency / red_flag / forced_actions) is sealed by the
deterministic layers (RedFlagEngine + rails).

INV-5: importing orchestrator from engines / reasoner / rails is FORBIDDEN.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

import clinibrium.audit.engine as _audit_engine
import clinibrium.ml_client as _ml_client
import clinibrium.rails as _rails
import clinibrium.reasoner as _reasoner
from clinibrium.contracts.enums import ForcedAction, Urgency
from clinibrium.contracts.features import CaseFeatures
from clinibrium.contracts.results import (
    DifferentialResult,
    PipelineResult,
    RedFlagResult,
)
from clinibrium.differential_engine import evaluate as differential_evaluate
from clinibrium.grounding import Grounding, get_grounding
from clinibrium.redflag_engine import evaluate as redflag_evaluate

logger = logging.getLogger(__name__)

StageHook = Callable[[str, Any], Awaitable[None]]
"""Observational hook: receives (stage_name, payload) after each pipeline step.
The LLM/client must NEVER set binding urgency — this is purely observational
(AD-6 / hard rule 2). A hook failure is logged and does NOT break the
pipeline or affect INV-4."""


async def _notify(
    on_stage: StageHook | None,
    stage: str,
    payload: Any,
) -> None:
    """Invokes the observational hook with a fail-safe guard.

    If the hook is None, does nothing. If the hook raises, ONLY logs —
    never propagates (an observer failure must not affect the pipeline).
    """
    if on_stage is None:
        return
    try:
        await on_stage(stage, payload)
    except Exception:
        logger.exception(
            "on_stage hook failed (stage=%s) — pipeline continues without observability",
            stage,
        )


def _redflag_payload(red_flag: RedFlagResult) -> dict[str, Any]:
    return {
        "red_flag_activa": red_flag.red_flag_activa,
        "hits_count": len(red_flag.hits),
    }


def _differential_payload(differential: DifferentialResult) -> dict[str, Any]:
    top = [
        {"diagnosis": c.diagnosis.value, "score": c.score, "rule_ids": c.rule_ids}
        for c in differential.candidates[:5]
    ]
    return {"top_candidates": top}


def _ml_payload(ml: Any) -> dict[str, Any]:
    return {"available": ml is not None}


def _reasoning_payload(reasoning: Any) -> dict[str, Any]:
    return {
        "available": reasoning is not None,
        "model_used": reasoning.model_used if reasoning is not None else None,
    }


def _rails_payload(sealed: PipelineResult) -> dict[str, Any]:
    return {
        "urgency": sealed.urgency.value,
        "forced_actions": sorted(a.value for a in sealed.forced_actions),
        "applied_rails": list(sealed.applied_rails),
    }


async def evaluate(
    features: CaseFeatures,
    *,
    recording_mode: bool = False,
    grounding: Grounding | None = None,
    now: datetime | None = None,
    on_stage: StageHook | None = None,
    kill_reasoner: bool = False,
    lang: str = "es",
) -> PipelineResult:
    """Evaluates a complete clinical case — end-to-end Clinibrium pipeline.

    Args:
        features: de-identified CaseFeatures of the case.
        recording_mode: if True, forces Opus in the reasoner (quality > cost).
        grounding: injectable Grounding implementation (tests); if None,
                   uses `get_grounding()`.
        now: injectable timestamp for `AuditEvent.occurred_at` (deterministic tests).
        on_stage: optional observational hook, invoked after each pipeline
            step with (stage_name, payload). PURELY OBSERVATIONAL: a hook
            failure is logged and does NOT affect the result, urgency or
            audit. If None, the pipeline behaves identically to previous
            versions (INV-4 / existing tests unchanged).
        kill_reasoner: if True, forces reasoning=None and reasoner_status="degraded"
            (intentional INV-8). Does NOT affect urgency, red flags, forced_actions
            or the INV-4 guard. It is the already-existing INV-8 degradation,
            triggered on purpose (debug/demo).

    Returns:
        Sealed PipelineResult (post-rails) with audit_event_id populated.

    INV-4 guarantee: exactly 1 AuditEvent emitted. If an intermediate step
    fails, an error AuditEvent is emitted and the exception is re-raised.
    """
    occurred_at = now if now is not None else datetime.now(timezone.utc)
    reasoner_status = "degraded"
    audited = False
    result: PipelineResult | None = None

    try:
        # ── Step 1: RedFlagEngine (deterministic, separate) ─────────────────
        red_flag = redflag_evaluate(features)
        await _notify(on_stage, "redflag", _redflag_payload(red_flag))

        # ── Step 2: DifferentialEngine (ICVD rules) ────────────────────────
        differential = differential_evaluate(features)
        await _notify(on_stage, "differential", _differential_payload(differential))

        # ── Step 3: optional ML (track B) — None if B is down (INV-6) ──────
        ml = await _ml_client.predict(features)
        await _notify(on_stage, "ml", _ml_payload(ml))

        # ── Step 4: Grounding retrieval (AD-12) ─────────────────────────────
        _grounding = grounding if grounding is not None else get_grounding()
        chunks = _grounding.retrieve(differential, features, k=4)

        # ── Step 5: Claude reasoner — None if down (INV-8) or kill_reasoner ──
        if kill_reasoner:
            reasoning = None
        else:
            reasoning = await _reasoner.reason(
                features,
                red_flag,
                differential,
                ml,
                chunks,
                recording_mode=recording_mode,
                lang=lang,
            )
        if reasoning is not None:
            reasoner_status = "ok"
        await _notify(on_stage, "reasoning", _reasoning_payload(reasoning))

        # ── Step 6: Assemble preliminary PipelineResult ────────────────────
        case_id = _derive_case_id(features)
        result = PipelineResult(
            case_id=case_id,
            urgency=Urgency.ambulatoria,
            red_flag=red_flag,
            differential=differential,
            ml=ml,
            reasoning=reasoning,
            forced_actions=set(),
            applied_rails=[],
        )

        # ── Step 7: Rails — final urgency / forced_actions (THEY WIN) ─────
        sealed = _rails.apply_rails(result, features)
        await _notify(on_stage, "rails", _rails_payload(sealed))

        # ── Step 8: Emit AuditEvent ────────────────────────────────────────
        event = await _audit_engine.emit(
            sealed,
            features,
            reasoner_status=reasoner_status,
            outcome="evaluation",
            occurred_at=occurred_at,
            lang=lang,
        )
        # INV-4: the AuditEvent has ALREADY been emitted. Set `audited` BEFORE
        # any other operation (model_copy) so that a later failure does NOT
        # trigger a second emit in the except block (avoids the double AuditEvent).
        audited = True
        sealed = sealed.model_copy(
            update={"audit_event_id": event.id, "audit_event": event}
        )
        return sealed

    except Exception:
        # ── INV-4 guard: emit 1 fail-safe AuditEvent and re-raise ──────────
        if not audited:
            try:
                await _emit_error_event(
                    features=features,
                    red_flag=result.red_flag if result is not None else None,
                    reasoner_status=reasoner_status,
                    occurred_at=occurred_at,
                    lang=lang,
                )
            except Exception:
                logger.exception(
                    "INV-4 guard: could not emit error AuditEvent — "
                    "the pipeline failed without audit traceability"
                )
        raise


async def _emit_error_event(
    *,
    features: CaseFeatures,
    red_flag: RedFlagResult | None,
    reasoner_status: str,
    occurred_at: datetime,
    lang: str | None = None,
) -> None:
    """Emits a fail-safe AuditEvent when the pipeline fails.

    If the system fails, it is treated as immediate urgency (forced human
    review). `red_flag_activa` = whatever was computed, or False.
    """
    urgency = Urgency.inmediata

    if red_flag is None:
        red_flag = RedFlagResult(red_flag_activa=False, hits=[], forced_actions=set())

    error_result = PipelineResult(
        case_id="error",
        urgency=urgency,
        red_flag=red_flag,
        differential=DifferentialResult(candidates=[]),
        ml=None,
        reasoning=None,
        forced_actions={ForcedAction.DERIVAR_URGENTE, ForcedAction.ESCALAR},
        applied_rails=[],
    )

    await _audit_engine.emit(
        error_result,
        features,
        reasoner_status=reasoner_status,
        outcome="error",
        occurred_at=occurred_at,
        lang=lang,
    )


def _derive_case_id(features: CaseFeatures) -> str:
    """Derives a deterministic case_id from the features hash.

    No PII: the features are already de-identified by construction.
    """
    import hashlib
    import json

    payload = features.model_dump(mode="json")
    canonical = json.dumps(payload, sort_keys=True, default=str)
    h = hashlib.sha256(canonical.encode()).hexdigest()
    return h[:12]
