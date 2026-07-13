"""Deterministic rails — the hard safety net applied AFTER Claude.

Each rail is a pure function. `apply_rails` composes them in order.
The LLM never sets binding urgency (INV-3); the rails always win.

INV-1 : red_flag_activa == True ⇒ urgency = inmediata
INV-3 : urgency NEVER comes from the LLM, always from deterministic layers
INV-5 : rails are FORBIDDEN from importing reasoner/redflag_engine/
        differential_engine/ml_client/orchestrator/api (contracts only)
INV-7 : safety monotonicity (urgency only goes UP), idempotence, totality,
        traceability
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
"""Each rail returns (forced_actions_it_adds, rail_id_if_fired)."""


def _rail_inv1(result: PipelineResult, _features: CaseFeatures) -> _RailResult:
    """R-INV1: red flag wins — immediate urgency, propagates forced_actions."""
    if result.red_flag.red_flag_activa:
        actions = set(result.red_flag.forced_actions)
        # red_flag_activa ⇒ urgent referral forced by definition of the RedFlagEngine.
        # We re-assert it here (defensive + INV-7 traceability): R-INV1 is always
        # recorded and DERIVAR_URGENTE present even if forced_actions came in empty.
        actions.add(ForcedAction.DERIVAR_URGENTE)
        return actions, "R-INV1"
    return set(), None


def _rail_epley_d(
    result: PipelineResult, features: CaseFeatures, accumulated_forced: set[ForcedAction]
) -> _RailResult:
    """R-EPLEY-D: Block D — when NOT to recommend Epley."""
    actions: set[ForcedAction] = set()

    red_flag_activa = result.red_flag.red_flag_activa
    # Direct check of exam contraindications (defensive): we do not rely solely
    # on RedFlagEngine having already propagated PRECAUCION_EXAMEN into accumulated_forced.
    precaucion_presente = (
        ForcedAction.PRECAUCION_EXAMEN in accumulated_forced
        or features.cervical_pathology
        or features.known_carotid_vertebrobasilar_disease
        or features.cardiovascular_instability
    )

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
    nystagmus_atypical_dir = (
        features.nystagmus_direction
        in {
            NystagmusDirection.vertical_pure,
            NystagmusDirection.torsional_pure,
            NystagmusDirection.direction_changing,  # central sign: block Epley
        }
        or features.nystagmus_direction_changing_gaze
    )
    atypical_nystagmus = (
        nystagmus_duration_long or nystagmus_not_fatigable or nystagmus_atypical_dir
    )

    bloque_epley = (
        red_flag_activa
        or precaucion_presente
        or (top_candidate is None or not top_is_bppv_posterior)
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
    """R-E2: epistemic — uncertainty ⇒ ESCALAR (fail-safe)."""
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
    """R-DIVERGENCIA: reasoner suggests higher urgency than the deterministic one.

    INV-3: we NEVER adopt the LLM's value. If the reasoner suggests
    higher urgency, we respond with ESCALAR (deterministic), not with
    the LLM's value.  If it suggests lower or equal, it is ignored.
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
    """Computes the final urgency from the accumulated forced_actions.

    Only `DERIVAR_URGENTE` / `red_flag_activa` set `inmediata`;
    only `ESCALAR` sets `prioritaria`; the rest do not set urgency.
    Always returns a value (totality → default ambulatoria).
    Monotonicity: never goes below `current_urgency`.
    """
    urgency: Urgency = Urgency.ambulatoria

    if red_flag_activa or ForcedAction.DERIVAR_URGENTE in forced_actions:
        urgency = urgency_max(urgency, Urgency.inmediata)
    if ForcedAction.ESCALAR in forced_actions:
        urgency = urgency_max(urgency, Urgency.prioritaria)

    urgency = urgency_max(urgency, current_urgency)
    return urgency


def apply_rails(result: PipelineResult, features: CaseFeatures) -> PipelineResult:
    """Applies all deterministic rails and returns a sealed PipelineResult.

    Does NOT mutate `result` or `features`.  Each rail adds forced_actions
    and its id to `applied_rails`.  The final urgency is computed from the
    accumulated forced_actions with guaranteed monotonicity (INV-7).

    Application order:
      1. R-INV1 — red flag wins
      2. R-EPLEY-D — block D (NO Epley)
      3. R-E2 — epistemic (uncertainty ⇒ ESCALAR)
      4. R-DIVERGENCIA — reasoner vs deterministic divergence
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
