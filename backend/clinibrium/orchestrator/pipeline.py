"""Pipeline de evaluación VertigoDx — compone todos los módulos.

Es el ÚNICO módulo que conoce el grafo completo:
    orchestrator → engines + reasoner + rails + ml_client + audit + grounding + contracts

INV-4: exactamente 1 AuditEvent por invocación, incluso bajo fallo parcial
(ml-down + reasoner-down) o excepción inesperada. El flag `audited`
garantiza que nunca se emitan 0 ni 2 eventos.

INV-6 / INV-8: ml_client y reasoner degradan a None sin romper el pipeline.
La seguridad (urgencia / red_flag / forced_actions) la sellan capas
deterministas (RedFlagEngine + rails).

INV-5: PROHIBIDO que orchestrator sea importado por engines / reasoner / rails.
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
"""Hook observacional: recibe (stage_name, payload) tras cada paso del pipeline.
El LLM/cliente NUNCA debe fijar urgencia vinculante — esto es puramente
observacional (AD-6 / regla dura 2). Un fallo del hook se loguea y NO
rompe el pipeline ni afecta INV-4."""


async def _notify(
    on_stage: StageHook | None,
    stage: str,
    payload: Any,
) -> None:
    """Invoca el hook observacional con guarda fail-safe.

    Si el hook es None, no hace nada. Si el hook levanta, SOLO loguea —
    nunca propaga (un fallo del observador no debe afectar el pipeline).
    """
    if on_stage is None:
        return
    try:
        await on_stage(stage, payload)
    except Exception:
        logger.exception(
            "on_stage hook failed (stage=%s) — pipeline continúa sin observabilidad",
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
) -> PipelineResult:
    """Evalúa un caso clínico completo — pipeline end-to-end de VertigoDx.

    Args:
        features: CaseFeatures desidentificadas del caso.
        recording_mode: si True, fuerza Opus en el razonador (calidad > costo).
        grounding: implementación de Grounding inyectable (tests); si None,
                   usa `get_grounding()`.
        now: timestamp inyectable para `AuditEvent.occurred_at` (tests determinísticos).
        on_stage: hook observacional opcional, invocado tras cada paso del
            pipeline con (stage_name, payload). PURAMENTE OBSERVACIONAL:
            un fallo del hook se loguea y NO afecta el resultado, urgencia
            ni auditoría. Si es None, el pipeline se comporta idéntico a
            versiones anteriores (INV-4 / tests existentes sin cambios).

    Returns:
        PipelineResult sellado (post-rails) con audit_event_id poblado.

    INV-4 garantía: exactamente 1 AuditEvent emitido. Si un paso intermedio
    falla, se emite un AuditEvent de error y se re-lanza la excepción.
    """
    occurred_at = now if now is not None else datetime.now(timezone.utc)
    reasoner_status = "degraded"
    audited = False
    result: PipelineResult | None = None

    try:
        # ── Paso 1: RedFlagEngine (determinista, separado) ──────────────────
        red_flag = redflag_evaluate(features)
        await _notify(on_stage, "redflag", _redflag_payload(red_flag))

        # ── Paso 2: DifferentialEngine (reglas ICVD) ───────────────────────
        differential = differential_evaluate(features)
        await _notify(on_stage, "differential", _differential_payload(differential))

        # ── Paso 3: ML opcional (track B) — None si B down (INV-6) ─────────
        ml = await _ml_client.predict(features)
        await _notify(on_stage, "ml", _ml_payload(ml))

        # ── Paso 4: Grounding retrieval (AD-12) ─────────────────────────────
        _grounding = grounding if grounding is not None else get_grounding()
        chunks = _grounding.retrieve(differential, features, k=4)

        # ── Paso 5: Razonador Claude — None si down (INV-8) ─────────────────
        reasoning = await _reasoner.reason(
            features,
            red_flag,
            differential,
            ml,
            chunks,
            recording_mode=recording_mode,
        )
        if reasoning is not None:
            reasoner_status = "ok"
        await _notify(on_stage, "reasoning", _reasoning_payload(reasoning))

        # ── Paso 6: Ensamblar PipelineResult preliminar ────────────────────
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

        # ── Paso 7: Rieles — urgencia / forced_actions finales (GANAN) ────
        sealed = _rails.apply_rails(result, features)
        await _notify(on_stage, "rails", _rails_payload(sealed))

        # ── Paso 8: Emitir AuditEvent ──────────────────────────────────────
        event = await _audit_engine.emit(
            sealed,
            features,
            reasoner_status=reasoner_status,
            outcome="evaluation",
            occurred_at=occurred_at,
        )
        # INV-4: el AuditEvent YA se emitió. Marcá `audited` ANTES de cualquier
        # otra operación (model_copy) para que un fallo posterior NO gatille un
        # segundo emit en el except (evita el doble-AuditEvent).
        audited = True
        sealed = sealed.model_copy(
            update={"audit_event_id": event.id, "audit_event": event}
        )
        return sealed

    except Exception:
        # ── Guard INV-4: emitir 1 AuditEvent fail-safe y re-lanzar ─────────
        if not audited:
            try:
                await _emit_error_event(
                    features=features,
                    red_flag=result.red_flag if result is not None else None,
                    reasoner_status=reasoner_status,
                    occurred_at=occurred_at,
                )
            except Exception:
                logger.exception(
                    "INV-4 guard: no se pudo emitir AuditEvent de error — "
                    "el pipeline falló sin trazabilidad de auditoría"
                )
        raise


async def _emit_error_event(
    *,
    features: CaseFeatures,
    red_flag: RedFlagResult | None,
    reasoner_status: str,
    occurred_at: datetime,
) -> None:
    """Emite un AuditEvent fail-safe cuando el pipeline falla.

    Si el sistema falla, se trata como urgencia inmediata (revisión humana
    forzada). `red_flag_activa` = lo que se haya calculado o False.
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
    )


def _derive_case_id(features: CaseFeatures) -> str:
    """Deriva un case_id determinista del hash de features.

    Sin PII: las features ya son desidentificadas por construcción.
    """
    import hashlib
    import json

    payload = features.model_dump(mode="json")
    canonical = json.dumps(payload, sort_keys=True, default=str)
    h = hashlib.sha256(canonical.encode()).hexdigest()
    return h[:12]
