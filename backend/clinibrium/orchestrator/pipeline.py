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


async def evaluate(
    features: CaseFeatures,
    *,
    recording_mode: bool = False,
    grounding: Grounding | None = None,
    now: datetime | None = None,
) -> PipelineResult:
    """Evalúa un caso clínico completo — pipeline end-to-end de VertigoDx.

    Args:
        features: CaseFeatures desidentificadas del caso.
        recording_mode: si True, fuerza Opus en el razonador (calidad > costo).
        grounding: implementación de Grounding inyectable (tests); si None,
                   usa `get_grounding()`.
        now: timestamp inyectable para `AuditEvent.occurred_at` (tests determinísticos).

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

        # ── Paso 2: DifferentialEngine (reglas ICVD) ───────────────────────
        differential = differential_evaluate(features)

        # ── Paso 3: ML opcional (track B) — None si B down (INV-6) ─────────
        ml = await _ml_client.predict(features)

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
        sealed = sealed.model_copy(update={"audit_event_id": event.id})
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
