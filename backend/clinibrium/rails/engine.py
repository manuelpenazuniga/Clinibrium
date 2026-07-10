"""Rieles deterministas — la red dura que se aplica DESPUÉS de Claude.

Cada riel es una función pura. `apply_rails` los compone en orden.
El LLM nunca fija urgencia vinculante (INV-3); los rieles ganan siempre.

INV-1 : red_flag_activa == True ⇒ urgencia = inmediata
INV-3 : urgencia NUNCA del LLM, siempre de capas deterministas
INV-5 : rails PROHIBIDO importar reasoner/redflag_engine/differential_engine/
        ml_client/orchestrator/api (solo contracts)
INV-7 : monotonía de seguridad (solo SUBE urgencia), idempotencia, totalidad,
        trazabilidad
"""
from __future__ import annotations

from clinibrium.contracts.enums import ForcedAction, NystagmusDirection, Urgency
from clinibrium.contracts.features import CaseFeatures
from clinibrium.contracts.results import PipelineResult
from clinibrium.rails.ordering import _URGENCY_RANK, urgency_max
from clinibrium.rails.thresholds import (
    AMBIGUITY_EPSILON,
    BPPV_EPLEY_CONFIDENCE_FLOOR,
    DIFFERENTIAL_UNCERTAINTY_FLOOR,
)

_RailResult = tuple[set[ForcedAction], str | None]
"""Cada riel devuelve (forced_actions_que_agrega, rail_id_si_disparó)."""


def _rail_inv1(result: PipelineResult, _features: CaseFeatures) -> _RailResult:
    """R-INV1: red flag gana — urgencia inmediata, propaga forced_actions."""
    if result.red_flag.red_flag_activa:
        return set(result.red_flag.forced_actions), "R-INV1"
    return set(), None


def _rail_epley_d(
    result: PipelineResult, features: CaseFeatures, accumulated_forced: set[ForcedAction]
) -> _RailResult:
    """R-EPLEY-D: Bloque D — cuándo NO recomendar Epley."""
    actions: set[ForcedAction] = set()

    red_flag_activa = result.red_flag.red_flag_activa
    precaucion_presente = ForcedAction.PRECAUCION_EXAMEN in accumulated_forced

    top_candidate = (
        result.differential.candidates[0] if result.differential.candidates else None
    )
    top_is_bppv_posterior = (
        top_candidate is not None
        and top_candidate.diagnosis.value == "bppv_posterior"
    )
    top_is_bppv_horizontal = (
        top_candidate is not None
        and top_candidate.diagnosis.value == "bppv_horizontal"
    )
    top_score_low = (
        top_candidate is not None
        and top_candidate.score < BPPV_EPLEY_CONFIDENCE_FLOOR
    )

    nystagmus_duration_long = (
        features.nystagmus_duration_s is not None
        and features.nystagmus_duration_s > 60
    )
    nystagmus_not_fatigable = features.nystagmus_fatigable is False
    nystagmus_atypical_dir = features.nystagmus_direction in {
        NystagmusDirection.vertical_pure,
        NystagmusDirection.torsional_pure,
    }
    atypical_nystagmus = (
        nystagmus_duration_long or nystagmus_not_fatigable or nystagmus_atypical_dir
    )

    bloque_epley = (
        red_flag_activa
        or precaucion_presente
        or (top_candidate is not None and not top_is_bppv_posterior)
        or top_score_low
        or atypical_nystagmus
    )

    if bloque_epley:
        actions.add(ForcedAction.BLOQUEAR_EPLEY)

    if top_is_bppv_horizontal:
        actions.add(ForcedAction.ESCALAR)

    if atypical_nystagmus:
        actions.add(ForcedAction.NO_BENIGNO)

    if actions:
        return actions, "R-EPLEY-D"
    return set(), None


def _rail_e2(result: PipelineResult, _features: CaseFeatures) -> _RailResult:
    """R-E2: epistémico — incertidumbre ⇒ ESCALAR (fail-safe)."""
    candidates = result.differential.candidates
    if not candidates:
        return {ForcedAction.ESCALAR}, "R-E2"

    top = candidates[0]
    if top.score < DIFFERENTIAL_UNCERTAINTY_FLOOR:
        return {ForcedAction.ESCALAR}, "R-E2"

    if len(candidates) >= 2:
        top1_score = top.score
        top2_score = candidates[1].score
        if top1_score - top2_score < AMBIGUITY_EPSILON:
            return {ForcedAction.ESCALAR}, "R-E2"

    return set(), None


def _rail_divergencia(
    result: PipelineResult,
    _features: CaseFeatures,
    current_deterministic_urgency: Urgency,
) -> _RailResult:
    """R-DIVERGENCIA: reasoner sugiere más urgencia que la determinista.

    INV-3: NUNCA adoptamos el valor del LLM. Si el reasoner sugiere
    más urgencia, respondemos con ESCALAR (determinista), no con el
    valor del LLM.  Si sugiere menor o igual, se ignora.
    """
    if result.reasoning is None:
        return set(), None

    suggested = result.reasoning.reasoner_suggested_urgency
    if suggested is None:
        return set(), None

    if _URGENCY_RANK[suggested] < _URGENCY_RANK[current_deterministic_urgency]:
        return {ForcedAction.ESCALAR}, "R-DIVERGENCIA"

    return set(), None


def _compute_urgency(
    red_flag_activa: bool,
    forced_actions: set[ForcedAction],
    current_urgency: Urgency,
) -> Urgency:
    """Calcula la urgencia final a partir de forced_actions acumuladas.

    Solo `DERIVAR_URGENTE` / `red_flag_activa` fijan `inmediata`;
    solo `ESCALAR` fija `prioritaria`; el resto no fija urgencia.
    Siempre devuelve un valor (totalidad → default ambulatoria).
    Monotonía: nunca baja respecto de `current_urgency`.
    """
    urgency: Urgency = Urgency.ambulatoria

    if red_flag_activa or ForcedAction.DERIVAR_URGENTE in forced_actions:
        urgency = urgency_max(urgency, Urgency.inmediata)
    if ForcedAction.ESCALAR in forced_actions:
        urgency = urgency_max(urgency, Urgency.prioritaria)

    urgency = urgency_max(urgency, current_urgency)
    return urgency


def apply_rails(result: PipelineResult, features: CaseFeatures) -> PipelineResult:
    """Aplica todos los rieles deterministas y devuelve un PipelineResult sellado.

    NO muta `result` ni `features`.  Cada riel agrega forced_actions y su
    id a `applied_rails`.  La urgencia final se calcula a partir de las
    forced_actions acumuladas con monotonía garantizada (INV-7).

    Orden de aplicación:
      1. R-INV1 — red flag gana
      2. R-EPLEY-D — bloque D (NO Epley)
      3. R-E2 — epistémico (incertidumbre ⇒ ESCALAR)
      4. R-DIVERGENCIA — divergencia reasoner vs determinista
    """
    forced_actions: set[ForcedAction] = set(result.forced_actions)
    applied_rails: list[str] = list(result.applied_rails)
    applied_set: set[str] = set(result.applied_rails)

    features_input: CaseFeatures = features
    result_input: PipelineResult = result

    def _apply_rail(rail_result: _RailResult) -> None:
        actions, rail_id = rail_result
        if actions and rail_id is not None and rail_id not in applied_set:
            forced_actions.update(actions)
            applied_rails.append(rail_id)
            applied_set.add(rail_id)

    _apply_rail(_rail_inv1(result_input, features_input))
    _apply_rail(_rail_epley_d(result_input, features_input, forced_actions))
    _apply_rail(_rail_e2(result_input, features_input))

    pre_divergence_urgency = _compute_urgency(
        result_input.red_flag.red_flag_activa, forced_actions, result_input.urgency
    )
    _apply_rail(_rail_divergencia(result_input, features_input, pre_divergence_urgency))

    final_urgency = _compute_urgency(
        result_input.red_flag.red_flag_activa, forced_actions, result_input.urgency
    )

    return PipelineResult(
        case_id=result_input.case_id,
        urgency=final_urgency,
        red_flag=result_input.red_flag,
        differential=result_input.differential,
        ml=result_input.ml,
        reasoning=result_input.reasoning,
        forced_actions=forced_actions,
        applied_rails=applied_rails,
        audit_event_id=result_input.audit_event_id,
    )
