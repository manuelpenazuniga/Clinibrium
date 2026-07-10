"""Tests de la capa de RIELES (el corazón determinista de Clinibrium).

Cubre:
  - Unidad: ordering (urgency_max), thresholds, cada rail individual.
  - Integración: apply_rails con casos clínicos realistas.
  - TEST ADVERSARIAL INV-1 (el demo): aunque todo diga BPPV benigno,
    red_flag_activa=True fuerza inmediata.
  - INV-7: monotonía, idempotencia, totalidad, trazabilidad.
  - Pureza: apply_rails no muta sus argumentos.
  - INV-5: rails no importa módulos prohibidos (AST).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from clinibrium.contracts import (
    CaseFeatures,
    Diagnosis,
    DifferentialCandidate,
    DifferentialResult,
    ForcedAction,
    NystagmusDirection,
    PipelineResult,
    PredictResponse,
    ReasonerOutput,
    RedFlagHit,
    RedFlagResult,
    Urgency,
)
from clinibrium.rails import (
    AMBIGUITY_EPSILON,
    BPPV_EPLEY_CONFIDENCE_FLOOR,
    DIFFERENTIAL_UNCERTAINTY_FLOOR,
    apply_rails,
    urgency_max,
)
from clinibrium.rails.engine import (
    _URGENCY_RANK,
    _rail_inv1,
    _rail_epley_d,
    _rail_e2,
    _rail_divergencia,
    _compute_urgency,
)


# =========================================================================
# Helpers
# =========================================================================


def _make_features(**overrides: object) -> CaseFeatures:
    defaults: dict[str, object] = {
        "nystagmus_direction": NystagmusDirection.none,
    }
    defaults.update(overrides)
    return CaseFeatures(**defaults)  # type: ignore[arg-type]


def _make_result(
    *,
    case_id: str = "test-1",
    red_flag_activa: bool = False,
    red_flag_actions: set[ForcedAction] | None = None,
    red_flag_hits: list[RedFlagHit] | None = None,
    candidates: list[DifferentialCandidate] | None = None,
    forced_actions: set[ForcedAction] | None = None,
    applied_rails: list[str] | None = None,
    urgency: Urgency = Urgency.ambulatoria,
    reasoning: ReasonerOutput | None = None,
    ml: PredictResponse | None = None,
) -> PipelineResult:
    rf_actions = red_flag_actions or set()
    if bool(rf_actions) and not red_flag_activa:
        red_flag_activa = True
    return PipelineResult(
        case_id=case_id,
        urgency=urgency,
        red_flag=RedFlagResult(
            red_flag_activa=red_flag_activa,
            hits=red_flag_hits or [],
            forced_actions=rf_actions,
        ),
        differential=DifferentialResult(candidates=candidates or []),
        forced_actions=forced_actions or set(),
        applied_rails=applied_rails or [],
        reasoning=reasoning,
        ml=ml,
    )


# =========================================================================
# ordering.py
# =========================================================================


class TestUrgencyOrdering:
    def test_rank_values(self) -> None:
        assert _URGENCY_RANK[Urgency.inmediata] == 0
        assert _URGENCY_RANK[Urgency.prioritaria] == 1
        assert _URGENCY_RANK[Urgency.ambulatoria] == 2

    def test_max_inmediata_over_prioritaria(self) -> None:
        assert urgency_max(Urgency.inmediata, Urgency.prioritaria) == Urgency.inmediata

    def test_max_inmediata_over_ambulatoria(self) -> None:
        assert urgency_max(Urgency.inmediata, Urgency.ambulatoria) == Urgency.inmediata

    def test_max_prioritaria_over_ambulatoria(self) -> None:
        assert urgency_max(Urgency.prioritaria, Urgency.ambulatoria) == Urgency.prioritaria

    def test_max_same(self) -> None:
        assert urgency_max(Urgency.ambulatoria, Urgency.ambulatoria) == Urgency.ambulatoria
        assert urgency_max(Urgency.inmediata, Urgency.inmediata) == Urgency.inmediata

    def test_max_commutative(self) -> None:
        assert urgency_max(Urgency.ambulatoria, Urgency.inmediata) == Urgency.inmediata
        assert urgency_max(Urgency.prioritaria, Urgency.inmediata) == Urgency.inmediata
        assert urgency_max(Urgency.ambulatoria, Urgency.prioritaria) == Urgency.prioritaria


# =========================================================================
# thresholds.py
# =========================================================================


class TestThresholds:
    def test_constants_exist_and_values(self) -> None:
        assert isinstance(BPPV_EPLEY_CONFIDENCE_FLOOR, float)
        assert BPPV_EPLEY_CONFIDENCE_FLOOR == 0.6

        assert isinstance(DIFFERENTIAL_UNCERTAINTY_FLOOR, float)
        assert DIFFERENTIAL_UNCERTAINTY_FLOOR == 0.4

        assert isinstance(AMBIGUITY_EPSILON, float)
        assert AMBIGUITY_EPSILON == 0.1

    def test_constants_fail_safe_direction(self) -> None:
        """Los umbrales empujan hacia MÁS seguridad (escalar/bloquear)."""
        assert BPPV_EPLEY_CONFIDENCE_FLOOR > 0.5, (
            "Un floor bajo permitiría Epley con poca confianza — inseguro"
        )
        assert DIFFERENTIAL_UNCERTAINTY_FLOOR > 0.3, (
            "Un floor bajo no escalaría casos inciertos — inseguro"
        )
        assert AMBIGUITY_EPSILON > 0.05, (
            "Un epsilon pequeño ignoraría ambigüedades — inseguro"
        )


# =========================================================================
# Individual rail functions
# =========================================================================


class TestRailInv1:
    def test_red_flag_activa_fires(self) -> None:
        result = _make_result(red_flag_activa=True, red_flag_actions={ForcedAction.DERIVAR_URGENTE})
        actions, rail_id = _rail_inv1(result, _make_features())
        assert rail_id == "R-INV1"
        assert ForcedAction.DERIVAR_URGENTE in actions

    def test_red_flag_inactiva_does_not_fire(self) -> None:
        result = _make_result(red_flag_activa=False)
        actions, rail_id = _rail_inv1(result, _make_features())
        assert rail_id is None
        assert actions == set()

    def test_propagates_all_red_flag_forced_actions(self) -> None:
        rf_actions = {ForcedAction.DERIVAR_URGENTE, ForcedAction.NO_BENIGNO, ForcedAction.PRECAUCION_EXAMEN}
        result = _make_result(red_flag_activa=True, red_flag_actions=rf_actions)
        actions, rail_id = _rail_inv1(result, _make_features())
        assert actions == rf_actions


class TestRailEpleyD:
    def test_red_flag_activa_blocks_epley(self) -> None:
        result = _make_result(red_flag_activa=True, red_flag_actions={ForcedAction.DERIVAR_URGENTE},
                              candidates=[
                                  DifferentialCandidate(diagnosis=Diagnosis.bppv_posterior, score=0.95)
                              ])
        actions, rail_id = _rail_epley_d(result, _make_features(), set())
        assert rail_id == "R-EPLEY-D"
        assert ForcedAction.BLOQUEAR_EPLEY in actions

    def test_precaucion_examen_in_accumulated_blocks_epley(self) -> None:
        result = _make_result(red_flag_activa=False, candidates=[
            DifferentialCandidate(diagnosis=Diagnosis.bppv_posterior, score=0.95),
        ])
        accumulated = {ForcedAction.PRECAUCION_EXAMEN}
        actions, rail_id = _rail_epley_d(result, _make_features(), accumulated)
        assert rail_id == "R-EPLEY-D"
        assert ForcedAction.BLOQUEAR_EPLEY in actions

    def test_top_not_bppv_posterior_blocks_epley(self) -> None:
        result = _make_result(red_flag_activa=False, candidates=[
            DifferentialCandidate(diagnosis=Diagnosis.vestibular_neuritis, score=0.8),
        ])
        actions, rail_id = _rail_epley_d(result, _make_features(), set())
        assert rail_id == "R-EPLEY-D"
        assert ForcedAction.BLOQUEAR_EPLEY in actions

    def test_top_bppv_posterior_below_confidence_floor_blocks_epley(self) -> None:
        result = _make_result(red_flag_activa=False, candidates=[
            DifferentialCandidate(diagnosis=Diagnosis.bppv_posterior, score=0.5),
        ])
        actions, rail_id = _rail_epley_d(result, _make_features(), set())
        assert rail_id == "R-EPLEY-D"
        assert ForcedAction.BLOQUEAR_EPLEY in actions

    def test_bppv_horizontal_top_blocks_epley_and_escalar(self) -> None:
        result = _make_result(red_flag_activa=False, candidates=[
            DifferentialCandidate(diagnosis=Diagnosis.bppv_horizontal, score=0.9),
        ])
        actions, rail_id = _rail_epley_d(result, _make_features(), set())
        assert rail_id == "R-EPLEY-D"
        assert ForcedAction.BLOQUEAR_EPLEY in actions
        assert ForcedAction.ESCALAR in actions

    def test_atypical_nystagmus_duration_blocks_epley_and_no_benigno(self) -> None:
        features = _make_features(nystagmus_duration_s=90.0)
        result = _make_result(red_flag_activa=False, candidates=[
            DifferentialCandidate(diagnosis=Diagnosis.bppv_posterior, score=0.95),
        ])
        actions, rail_id = _rail_epley_d(result, features, set())
        assert rail_id == "R-EPLEY-D"
        assert ForcedAction.BLOQUEAR_EPLEY in actions
        assert ForcedAction.NO_BENIGNO in actions

    def test_atypical_nystagmus_not_fatigable_blocks_epley_and_no_benigno(self) -> None:
        features = _make_features(nystagmus_fatigable=False)
        result = _make_result(red_flag_activa=False, candidates=[
            DifferentialCandidate(diagnosis=Diagnosis.bppv_posterior, score=0.95),
        ])
        actions, rail_id = _rail_epley_d(result, features, set())
        assert rail_id == "R-EPLEY-D"
        assert ForcedAction.BLOQUEAR_EPLEY in actions
        assert ForcedAction.NO_BENIGNO in actions

    def test_atypical_nystagmus_direction_blocks_epley_and_no_benigno(self) -> None:
        for direction in (NystagmusDirection.vertical_pure, NystagmusDirection.torsional_pure):
            features = _make_features(nystagmus_direction=direction)
            result = _make_result(red_flag_activa=False, candidates=[
                DifferentialCandidate(diagnosis=Diagnosis.bppv_posterior, score=0.95),
            ])
            actions, rail_id = _rail_epley_d(result, features, set())
            assert rail_id == "R-EPLEY-D", f"failed for direction={direction}"
            assert ForcedAction.BLOQUEAR_EPLEY in actions
            assert ForcedAction.NO_BENIGNO in actions

    def test_no_epley_block_when_benign_bppv_posterior_high_confidence(self) -> None:
        features = _make_features(
            nystagmus_duration_s=20.0,
            nystagmus_fatigable=True,
            nystagmus_direction=NystagmusDirection.mixed,
        )
        result = _make_result(red_flag_activa=False, candidates=[
            DifferentialCandidate(diagnosis=Diagnosis.bppv_posterior, score=0.95),
        ])
        actions, rail_id = _rail_epley_d(result, features, set())
        assert rail_id is None
        assert actions == set()

    def test_nystagmus_fatigable_none_does_not_trigger(self) -> None:
        """nystagmus_fatigable=None no es explícitamente False, no dispara."""
        features = _make_features(nystagmus_fatigable=None)
        result = _make_result(red_flag_activa=False, candidates=[
            DifferentialCandidate(diagnosis=Diagnosis.bppv_posterior, score=0.95),
        ])
        actions, rail_id = _rail_epley_d(result, features, set())
        assert rail_id is None

    def test_nystagmus_duration_none_does_not_trigger(self) -> None:
        features = _make_features(nystagmus_duration_s=None)
        result = _make_result(red_flag_activa=False, candidates=[
            DifferentialCandidate(diagnosis=Diagnosis.bppv_posterior, score=0.95),
        ])
        actions, rail_id = _rail_epley_d(result, features, set())
        assert rail_id is None

    def test_empty_differential_blocks_epley(self) -> None:
        """Fix auditoría (D1): sin candidatos NO hay BPPV confiable ⇒ BLOQUEAR_EPLEY.
        (Antes se dejaba pasar — falso negativo detectado por auditoría Gemini.)"""
        result = _make_result(red_flag_activa=False, candidates=[])
        actions, rail_id = _rail_epley_d(result, _make_features(), set())
        assert ForcedAction.BLOQUEAR_EPLEY in actions
        assert rail_id == "R-EPLEY-D"


class TestRailE2:
    def test_empty_differential_escalar(self) -> None:
        result = _make_result(red_flag_activa=False, candidates=[])
        actions, rail_id = _rail_e2(result, _make_features())
        assert rail_id == "R-E2"
        assert actions == {ForcedAction.ESCALAR}

    def test_top_below_uncertainty_floor_escalar(self) -> None:
        result = _make_result(red_flag_activa=False, candidates=[
            DifferentialCandidate(diagnosis=Diagnosis.undetermined, score=0.3),
        ])
        actions, rail_id = _rail_e2(result, _make_features())
        assert rail_id == "R-E2"
        assert actions == {ForcedAction.ESCALAR}

    def test_ambiguity_escalar(self) -> None:
        result = _make_result(red_flag_activa=False, candidates=[
            DifferentialCandidate(diagnosis=Diagnosis.bppv_posterior, score=0.65),
            DifferentialCandidate(diagnosis=Diagnosis.vestibular_neuritis, score=0.56),
        ])
        # diff = 0.09 < 0.1
        actions, rail_id = _rail_e2(result, _make_features())
        assert rail_id == "R-E2"
        assert actions == {ForcedAction.ESCALAR}

    def test_single_candidate_no_ambiguity_trigger(self) -> None:
        """Con un solo candidato no hay top2 con que comparar."""
        result = _make_result(red_flag_activa=False, candidates=[
            DifferentialCandidate(diagnosis=Diagnosis.bppv_posterior, score=0.7),
        ])
        actions, rail_id = _rail_e2(result, _make_features())
        # score=0.7 > 0.4 floor, single candidate → no ambiguity
        assert rail_id is None
        assert actions == set()

    def test_no_escalar_when_confident(self) -> None:
        result = _make_result(red_flag_activa=False, candidates=[
            DifferentialCandidate(diagnosis=Diagnosis.bppv_posterior, score=0.9),
            DifferentialCandidate(diagnosis=Diagnosis.vestibular_neuritis, score=0.2),
        ])
        actions, rail_id = _rail_e2(result, _make_features())
        assert rail_id is None
        assert actions == set()


class TestRailDivergencia:
    def test_reasoning_none_no_effect(self) -> None:
        result = _make_result(red_flag_activa=False, reasoning=None)
        actions, rail_id = _rail_divergencia(result, _make_features(), Urgency.ambulatoria)
        assert rail_id is None
        assert actions == set()

    def test_suggested_urgency_none_no_effect(self) -> None:
        reasoning = ReasonerOutput(
            model_used="test-model",
            explanation="test",
            reconciliation="test",
            reasoner_suggested_urgency=None,
        )
        result = _make_result(red_flag_activa=False, reasoning=reasoning)
        actions, rail_id = _rail_divergencia(result, _make_features(), Urgency.ambulatoria)
        assert rail_id is None
        assert actions == set()

    def test_suggested_higher_than_deterministic_adds_escalar(self) -> None:
        reasoning = ReasonerOutput(
            model_used="test-model",
            explanation="test",
            reconciliation="test",
            reasoner_suggested_urgency=Urgency.inmediata,
        )
        result = _make_result(red_flag_activa=False, reasoning=reasoning)
        actions, rail_id = _rail_divergencia(result, _make_features(), Urgency.ambulatoria)
        assert rail_id == "R-DIVERGENCIA"
        assert actions == {ForcedAction.ESCALAR}

    def test_suggested_lower_than_deterministic_ignored(self) -> None:
        reasoning = ReasonerOutput(
            model_used="test-model",
            explanation="test",
            reconciliation="test",
            reasoner_suggested_urgency=Urgency.ambulatoria,
        )
        result = _make_result(red_flag_activa=False, reasoning=reasoning)
        actions, rail_id = _rail_divergencia(result, _make_features(), Urgency.prioritaria)
        assert rail_id is None
        assert actions == set()

    def test_suggested_equal_to_deterministic_ignored(self) -> None:
        reasoning = ReasonerOutput(
            model_used="test-model",
            explanation="test",
            reconciliation="test",
            reasoner_suggested_urgency=Urgency.prioritaria,
        )
        result = _make_result(red_flag_activa=False, reasoning=reasoning)
        actions, rail_id = _rail_divergencia(result, _make_features(), Urgency.prioritaria)
        assert rail_id is None
        assert actions == set()

    def test_suggested_higher_but_deterministic_already_inmediata_ignored(self) -> None:
        """Si determinista ya es inmediata, reasoner no puede sugerir algo más alto."""
        reasoning = ReasonerOutput(
            model_used="test-model",
            explanation="test",
            reconciliation="test",
            reasoner_suggested_urgency=Urgency.inmediata,
        )
        result = _make_result(red_flag_activa=True, red_flag_actions={ForcedAction.DERIVAR_URGENTE},
                              reasoning=reasoning)
        actions, rail_id = _rail_divergencia(result, _make_features(), Urgency.inmediata)
        # inmediata rank 0, inmediata rank 0 → iguales → ignorado
        assert rail_id is None
        assert actions == set()


class TestComputeUrgency:
    def test_no_forced_actions_defaults_ambulatoria(self) -> None:
        urgencia = _compute_urgency(False, set(), Urgency.ambulatoria)
        assert urgencia == Urgency.ambulatoria

    def test_escalar_drives_prioritaria(self) -> None:
        urgencia = _compute_urgency(False, {ForcedAction.ESCALAR}, Urgency.ambulatoria)
        assert urgencia == Urgency.prioritaria

    def test_derivar_urgente_drives_inmediata(self) -> None:
        urgencia = _compute_urgency(False, {ForcedAction.DERIVAR_URGENTE}, Urgency.ambulatoria)
        assert urgencia == Urgency.inmediata

    def test_red_flag_activa_drives_inmediata(self) -> None:
        urgencia = _compute_urgency(True, set(), Urgency.ambulatoria)
        assert urgencia == Urgency.inmediata

    def test_bloquear_epley_no_drives_urgency(self) -> None:
        urgencia = _compute_urgency(False, {ForcedAction.BLOQUEAR_EPLEY}, Urgency.ambulatoria)
        assert urgencia == Urgency.ambulatoria

    def test_precaucion_examen_no_drives_urgency(self) -> None:
        urgencia = _compute_urgency(False, {ForcedAction.PRECAUCION_EXAMEN}, Urgency.ambulatoria)
        assert urgencia == Urgency.ambulatoria

    def test_no_benigno_no_drives_urgency(self) -> None:
        urgencia = _compute_urgency(False, {ForcedAction.NO_BENIGNO}, Urgency.ambulatoria)
        assert urgencia == Urgency.ambulatoria

    def test_monotonia_respects_input_urgency(self) -> None:
        """Nunca baja respecto de current_urgency."""
        urgencia = _compute_urgency(False, set(), Urgency.prioritaria)
        assert urgencia == Urgency.prioritaria

    def test_inmediata_wins_over_prioritaria(self) -> None:
        urgencia = _compute_urgency(True, {ForcedAction.ESCALAR}, Urgency.ambulatoria)
        assert urgencia == Urgency.inmediata


# =========================================================================
# apply_rails — integración
# =========================================================================


class TestApplyRailsAdversarialInv1:
    """TEST ADVERSARIAL INV-1 — el que ES el demo.

    PipelineResult con red_flag_activa=True PERO differential top=bppv_posterior
    score 0.95, ml probabilities bppv 0.95, reasoning con
    reasoner_suggested_urgency=ambulatoria →
    apply_rails(...).urgency == inmediata.

    'Aunque todo diga BPPV benigno, el riel fuerza inmediata.'
    """

    def test_adversarial_red_flag_trumps_all(self) -> None:
        features = _make_features(
            nystagmus_duration_s=20.0,
            nystagmus_fatigable=True,
            nystagmus_direction=NystagmusDirection.mixed,
        )
        result = _make_result(
            red_flag_activa=True,
            red_flag_actions={ForcedAction.DERIVAR_URGENTE, ForcedAction.NO_BENIGNO},
            candidates=[
                DifferentialCandidate(diagnosis=Diagnosis.bppv_posterior, score=0.95),
            ],
            ml=PredictResponse(
                probabilities={"bppv_posterior": 0.95},
                model_version="catboost-v0.1",
            ),
            reasoning=ReasonerOutput(
            model_used="test-model",
                explanation="Caso benigno clásico de VPPB posterior.",
                reconciliation="Concuerda con hallazgos típicos.",
                reasoner_suggested_urgency=Urgency.ambulatoria,
            ),
            urgency=Urgency.ambulatoria,
        )
        output = apply_rails(result, features)
        assert output.urgency == Urgency.inmediata, (
            "INV-1 violada: red_flag_activa=True debería forzar inmediata"
        )
        assert "R-INV1" in output.applied_rails
        assert ForcedAction.DERIVAR_URGENTE in output.forced_actions
        assert ForcedAction.BLOQUEAR_EPLEY in output.forced_actions  # red flag bloquea Epley
        assert ForcedAction.NO_BENIGNO in output.forced_actions


class TestApplyRailsMonotoniaInv7:
    """INV-7: apply_rails solo SUBE urgencia, nunca la baja."""

    _CASES: list[tuple[PipelineResult, CaseFeatures]] = []

    @classmethod
    def setup_class(cls) -> None:
        feats = _make_features()
        # Caso 1: red flag activa → inmediata (sube desde ambulatoria)
        cls._CASES.append((
            _make_result(red_flag_activa=True, red_flag_actions={ForcedAction.DERIVAR_URGENTE},
                         urgency=Urgency.ambulatoria),
            feats,
        ))
        # Caso 2: ESCALAR → prioritaria (sube desde ambulatoria)
        cls._CASES.append((
            _make_result(red_flag_activa=False, candidates=[
                DifferentialCandidate(diagnosis=Diagnosis.undetermined, score=0.3),
            ], urgency=Urgency.ambulatoria),
            feats,
        ))
        # Caso 3: ya es prioritaria, se mantiene
        cls._CASES.append((
            _make_result(red_flag_activa=False, candidates=[
                DifferentialCandidate(diagnosis=Diagnosis.undetermined, score=0.3),
            ], urgency=Urgency.prioritaria),
            feats,
        ))
        # Caso 4: ya es inmediata, se mantiene
        cls._CASES.append((
            _make_result(red_flag_activa=True, red_flag_actions={ForcedAction.DERIVAR_URGENTE},
                         urgency=Urgency.inmediata),
            feats,
        ))
        # Caso 5: BPPV benigno, ambulatoria se mantiene
        cls._CASES.append((
            _make_result(red_flag_activa=False, candidates=[
                DifferentialCandidate(diagnosis=Diagnosis.bppv_posterior, score=0.95),
                DifferentialCandidate(diagnosis=Diagnosis.vestibular_neuritis, score=0.2),
            ], urgency=Urgency.ambulatoria),
            feats,
        ))

    @pytest.mark.parametrize("result,features", _CASES)
    def test_urgency_never_decreases(self, result: PipelineResult, features: CaseFeatures) -> None:
        input_urgency = result.urgency
        output = apply_rails(result, features)
        assert _URGENCY_RANK[output.urgency] <= _URGENCY_RANK[input_urgency], (
            f"Urgency bajó: {input_urgency} → {output.urgency}"
        )


class TestApplyRailsIdempotencia:
    """Segundo pase no cambia urgency/forced_actions/applied_rails."""

    def test_idempotent_red_flag_case(self) -> None:
        result = _make_result(red_flag_activa=True, red_flag_actions={ForcedAction.DERIVAR_URGENTE})
        features = _make_features()
        first = apply_rails(result, features)
        second = apply_rails(first, features)
        assert second.urgency == first.urgency
        assert second.forced_actions == first.forced_actions
        assert second.applied_rails == first.applied_rails

    def test_idempotent_epistemic_escalar_case(self) -> None:
        result = _make_result(red_flag_activa=False, candidates=[
            DifferentialCandidate(diagnosis=Diagnosis.undetermined, score=0.3),
        ])
        features = _make_features()
        first = apply_rails(result, features)
        second = apply_rails(first, features)
        assert second.urgency == first.urgency
        assert second.forced_actions == first.forced_actions
        assert second.applied_rails == first.applied_rails

    def test_idempotent_benign_case(self) -> None:
        result = _make_result(red_flag_activa=False, candidates=[
            DifferentialCandidate(diagnosis=Diagnosis.bppv_posterior, score=0.95),
            DifferentialCandidate(diagnosis=Diagnosis.vestibular_neuritis, score=0.2),
        ])
        features = _make_features(
            nystagmus_duration_s=20.0,
            nystagmus_fatigable=True,
            nystagmus_direction=NystagmusDirection.mixed,
        )
        first = apply_rails(result, features)
        second = apply_rails(first, features)
        assert second.urgency == first.urgency
        assert second.forced_actions == first.forced_actions
        assert second.applied_rails == first.applied_rails


class TestApplyRailsTotalidad:
    """Un PipelineResult mínimo siempre obtiene urgencia (nunca None)."""

    def test_minimal_result_gets_urgency(self) -> None:
        result = _make_result(red_flag_activa=False, candidates=[])
        output = apply_rails(result, _make_features())
        assert output.urgency is not None
        assert isinstance(output.urgency, Urgency)

    def test_empty_differential_escalar(self) -> None:
        """Differential vacío ⇒ R-E2 dispara ESCALAR."""
        result = _make_result(red_flag_activa=False, candidates=[])
        output = apply_rails(result, _make_features())
        assert output.urgency == Urgency.prioritaria  # ESCALAR sube a prioritaria
        assert ForcedAction.ESCALAR in output.forced_actions
        assert "R-E2" in output.applied_rails

    def test_default_ambulatoria_when_all_clear(self) -> None:
        result = _make_result(red_flag_activa=False, candidates=[
            DifferentialCandidate(diagnosis=Diagnosis.bppv_posterior, score=0.95),
            DifferentialCandidate(diagnosis=Diagnosis.vestibular_neuritis, score=0.2),
        ])
        features = _make_features(
            nystagmus_duration_s=20.0,
            nystagmus_fatigable=True,
            nystagmus_direction=NystagmusDirection.mixed,
        )
        output = apply_rails(result, features)
        assert output.urgency == Urgency.ambulatoria


class TestApplyRailsBloqueD:
    """Escenarios del Bloque D (cuándo NO Epley)."""

    def test_red_flag_blocks_epley(self) -> None:
        result = _make_result(red_flag_activa=True, red_flag_actions={ForcedAction.DERIVAR_URGENTE},
                              candidates=[
                                  DifferentialCandidate(diagnosis=Diagnosis.bppv_posterior, score=0.95),
                              ])
        output = apply_rails(result, _make_features())
        assert ForcedAction.BLOQUEAR_EPLEY in output.forced_actions
        assert "R-EPLEY-D" in output.applied_rails

    def test_top_non_bppv_posterior_blocks_epley(self) -> None:
        result = _make_result(red_flag_activa=False, candidates=[
            DifferentialCandidate(diagnosis=Diagnosis.vestibular_neuritis, score=0.8),
        ])
        output = apply_rails(result, _make_features())
        assert ForcedAction.BLOQUEAR_EPLEY in output.forced_actions

    def test_top_score_below_floor_blocks_epley(self) -> None:
        result = _make_result(red_flag_activa=False, candidates=[
            DifferentialCandidate(diagnosis=Diagnosis.bppv_posterior, score=0.5),
        ])
        output = apply_rails(result, _make_features())
        assert ForcedAction.BLOQUEAR_EPLEY in output.forced_actions

    def test_bppv_horizontal_top_blocks_epley_and_escalar(self) -> None:
        result = _make_result(red_flag_activa=False, candidates=[
            DifferentialCandidate(diagnosis=Diagnosis.bppv_horizontal, score=0.9),
        ])
        output = apply_rails(result, _make_features())
        assert ForcedAction.BLOQUEAR_EPLEY in output.forced_actions
        assert ForcedAction.ESCALAR in output.forced_actions

    def test_atypical_nystagmus_duration_blocks_epley_and_no_benigno(self) -> None:
        features = _make_features(nystagmus_duration_s=90.0)
        result = _make_result(red_flag_activa=False, candidates=[
            DifferentialCandidate(diagnosis=Diagnosis.bppv_posterior, score=0.95),
        ])
        output = apply_rails(result, features)
        assert ForcedAction.BLOQUEAR_EPLEY in output.forced_actions
        assert ForcedAction.NO_BENIGNO in output.forced_actions

    def test_atypical_nystagmus_not_fatigable_blocks_epley_and_no_benigno(self) -> None:
        features = _make_features(nystagmus_fatigable=False)
        result = _make_result(red_flag_activa=False, candidates=[
            DifferentialCandidate(diagnosis=Diagnosis.bppv_posterior, score=0.95),
        ])
        output = apply_rails(result, features)
        assert ForcedAction.BLOQUEAR_EPLEY in output.forced_actions
        assert ForcedAction.NO_BENIGNO in output.forced_actions

    def test_atypical_nystagmus_direction_blocks_epley_and_no_benigno(self) -> None:
        for direction in (NystagmusDirection.vertical_pure, NystagmusDirection.torsional_pure):
            features = _make_features(nystagmus_direction=direction)
            result = _make_result(red_flag_activa=False, candidates=[
                DifferentialCandidate(diagnosis=Diagnosis.bppv_posterior, score=0.95),
            ])
            output = apply_rails(result, features)
            assert ForcedAction.BLOQUEAR_EPLEY in output.forced_actions, f"direction={direction}"
            assert ForcedAction.NO_BENIGNO in output.forced_actions, f"direction={direction}"

    def test_benign_bppv_posterior_no_epley_block(self) -> None:
        features = _make_features(
            nystagmus_duration_s=20.0,
            nystagmus_fatigable=True,
            nystagmus_direction=NystagmusDirection.mixed,
        )
        result = _make_result(red_flag_activa=False, candidates=[
            DifferentialCandidate(diagnosis=Diagnosis.bppv_posterior, score=0.95),
            DifferentialCandidate(diagnosis=Diagnosis.vestibular_neuritis, score=0.2),
        ])
        output = apply_rails(result, features)
        assert ForcedAction.BLOQUEAR_EPLEY not in output.forced_actions


class TestApplyRailsDivergencia:
    """R-DIVERGENCIA: reasoner sugiere más urgencia ⇒ ESCALAR, no adopta valor LLM."""

    def test_reasoner_suggests_inmediata_deterministic_ambulatoria_becomes_prioritaria(self) -> None:
        """INV-3: la urgencia del LLM no se adopta; se traduce a ESCALAR."""
        reasoning = ReasonerOutput(
            model_used="test-model",
            explanation="Algo no cuadra.",
            reconciliation="Podría ser central.",
            reasoner_suggested_urgency=Urgency.inmediata,
        )
        result = _make_result(
            red_flag_activa=False,
            candidates=[
                DifferentialCandidate(diagnosis=Diagnosis.bppv_posterior, score=0.9),
                DifferentialCandidate(diagnosis=Diagnosis.vestibular_neuritis, score=0.2),
            ],
            reasoning=reasoning,
        )
        features = _make_features(
            nystagmus_duration_s=20.0,
            nystagmus_fatigable=True,
            nystagmus_direction=NystagmusDirection.mixed,
        )
        output = apply_rails(result, features)
        assert output.urgency == Urgency.prioritaria, (
            "Debe subir a prioritaria (vía ESCALAR), NO a inmediata"
        )
        assert output.urgency != Urgency.inmediata, (
            "INV-3 violada: NO se adopta el valor inmediata del LLM"
        )
        assert ForcedAction.ESCALAR in output.forced_actions
        assert "R-DIVERGENCIA" in output.applied_rails

    def test_reasoner_suggests_lower_urgency_ignored(self) -> None:
        reasoning = ReasonerOutput(
            model_used="test-model",
            explanation="Todo ok.",
            reconciliation="BPPV típico.",
            reasoner_suggested_urgency=Urgency.ambulatoria,
        )
        result = _make_result(
            red_flag_activa=True,
            red_flag_actions={ForcedAction.DERIVAR_URGENTE},
            candidates=[
                DifferentialCandidate(diagnosis=Diagnosis.bppv_posterior, score=0.95),
            ],
            reasoning=reasoning,
        )
        output = apply_rails(result, _make_features())
        assert output.urgency == Urgency.inmediata  # red flag wins
        assert "R-DIVERGENCIA" not in output.applied_rails

    def test_no_reasoning_no_divergencia(self) -> None:
        result = _make_result(red_flag_activa=False, reasoning=None, candidates=[
            DifferentialCandidate(diagnosis=Diagnosis.bppv_posterior, score=0.95),
            DifferentialCandidate(diagnosis=Diagnosis.vestibular_neuritis, score=0.2),
        ])
        features = _make_features(
            nystagmus_duration_s=20.0,
            nystagmus_fatigable=True,
            nystagmus_direction=NystagmusDirection.mixed,
        )
        output = apply_rails(result, features)
        assert "R-DIVERGENCIA" not in output.applied_rails


class TestApplyRailsTrazabilidad:
    """Todo forced_action de salida tiene su riel en applied_rails."""

    def _all_forced_actions_traced(self, output: PipelineResult) -> None:
        """Verifica que cada forced_action en output es producido por algún riel."""
        # Reconstruir qué acciones puede producir cada riel
        # Este es un sanity check estructural: si hay forced_actions, debe haber
        # al menos un riel aplicado.
        if output.forced_actions:
            assert output.applied_rails, (
                f"forced_actions={output.forced_actions} sin applied_rails"
            )

    def test_actions_from_inv1_have_rail_id(self) -> None:
        result = _make_result(red_flag_activa=True, red_flag_actions={ForcedAction.DERIVAR_URGENTE})
        output = apply_rails(result, _make_features())
        self._all_forced_actions_traced(output)
        assert "R-INV1" in output.applied_rails

    def test_actions_from_epley_d_have_rail_id(self) -> None:
        result = _make_result(red_flag_activa=False, candidates=[
            DifferentialCandidate(diagnosis=Diagnosis.vestibular_neuritis, score=0.8),
        ])
        output = apply_rails(result, _make_features())
        self._all_forced_actions_traced(output)
        assert "R-EPLEY-D" in output.applied_rails
        assert ForcedAction.BLOQUEAR_EPLEY in output.forced_actions

    def test_actions_from_e2_have_rail_id(self) -> None:
        result = _make_result(red_flag_activa=False, candidates=[])
        output = apply_rails(result, _make_features())
        self._all_forced_actions_traced(output)
        assert "R-E2" in output.applied_rails
        assert ForcedAction.ESCALAR in output.forced_actions

    def test_actions_from_divergencia_have_rail_id(self) -> None:
        reasoning = ReasonerOutput(
            model_used="test-model",
            explanation="test",
            reconciliation="test",
            reasoner_suggested_urgency=Urgency.inmediata,
        )
        result = _make_result(red_flag_activa=False, candidates=[
            DifferentialCandidate(diagnosis=Diagnosis.bppv_posterior, score=0.9),
            DifferentialCandidate(diagnosis=Diagnosis.vestibular_neuritis, score=0.2),
        ], reasoning=reasoning)
        features = _make_features(
            nystagmus_duration_s=20.0,
            nystagmus_fatigable=True,
            nystagmus_direction=NystagmusDirection.mixed,
        )
        output = apply_rails(result, features)
        self._all_forced_actions_traced(output)
        assert "R-DIVERGENCIA" in output.applied_rails
        assert ForcedAction.ESCALAR in output.forced_actions


class TestApplyRailsPureza:
    """apply_rails NO muta sus argumentos (result ni features)."""

    def test_pipeline_result_not_mutated(self) -> None:
        features = _make_features()
        result = _make_result(red_flag_activa=True, red_flag_actions={ForcedAction.DERIVAR_URGENTE})
        orig_forced = set(result.forced_actions)
        orig_applied = list(result.applied_rails)
        orig_urgency = result.urgency

        output = apply_rails(result, features)

        # El input NO cambió
        assert result.forced_actions == orig_forced
        assert result.applied_rails == orig_applied
        assert result.urgency == orig_urgency

        # La salida es distinta (nuevo objeto)
        assert output is not result
        assert output is not result

    def test_case_features_not_mutated(self) -> None:
        features = _make_features(nystagmus_duration_s=90.0)
        orig = features.model_copy(deep=True)
        result = _make_result(red_flag_activa=False, candidates=[
            DifferentialCandidate(diagnosis=Diagnosis.bppv_posterior, score=0.95),
        ])
        apply_rails(result, features)
        assert features == orig

    def test_precaucion_examen_propagation_preserves_precaucion(self) -> None:
        """Cuando R-INV1 propaga PRECAUCION_EXAMEN desde red_flag, R-EPLEY-D la ve."""
        result = _make_result(
            red_flag_activa=True,
            red_flag_actions={ForcedAction.DERIVAR_URGENTE, ForcedAction.PRECAUCION_EXAMEN},
            candidates=[
                DifferentialCandidate(diagnosis=Diagnosis.bppv_posterior, score=0.95),
            ],
        )
        output = apply_rails(result, _make_features(
            nystagmus_duration_s=20.0,
            nystagmus_fatigable=True,
            nystagmus_direction=NystagmusDirection.mixed,
        ))
        assert ForcedAction.PRECAUCION_EXAMEN in output.forced_actions
        assert ForcedAction.BLOQUEAR_EPLEY in output.forced_actions

    def test_input_reasoning_preserved_after_apply(self) -> None:
        reasoning = ReasonerOutput(
            model_used="test-model",
            explanation="Caso benigno.",
            reconciliation="BPPV típico.",
            reasoner_suggested_urgency=Urgency.ambulatoria,
        )
        result = _make_result(red_flag_activa=False, reasoning=reasoning, candidates=[
            DifferentialCandidate(diagnosis=Diagnosis.bppv_posterior, score=0.95),
            DifferentialCandidate(diagnosis=Diagnosis.vestibular_neuritis, score=0.2),
        ])
        features = _make_features(
            nystagmus_duration_s=20.0,
            nystagmus_fatigable=True,
            nystagmus_direction=NystagmusDirection.mixed,
        )
        apply_rails(result, features)
        assert result.reasoning is reasoning
        assert result.reasoning.reasoner_suggested_urgency == Urgency.ambulatoria


# =========================================================================
# INV-5 — rails solo importa contracts
# =========================================================================


_RAILS_FORBIDDEN_IMPORTS = {
    "clinibrium.reasoner",
    "clinibrium.redflag_engine",
    "clinibrium.differential_engine",
    "clinibrium.ml_client",
    "clinibrium.orchestrator",
    "clinibrium.api",
}


def _iter_imports(py_file: Path) -> list[tuple[int, str]]:
    """Devuelve (line_no, module) de cada import de `clinibrium.*` encontrado."""
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                out.append((node.lineno, node.module))
    return out


def test_rails_does_not_import_forbidden_modules() -> None:
    """INV-5: rails NO puede importar reasoner/engines/ml_client/orchestrator/api."""
    pkg_root = Path(__file__).resolve().parents[1] / "clinibrium" / "rails"
    offenders: list[str] = []
    for py in sorted(pkg_root.glob("*.py")):
        for lineno, mod in _iter_imports(py):
            for forbidden in _RAILS_FORBIDDEN_IMPORTS:
                if mod == forbidden or mod.startswith(forbidden + "."):
                    offenders.append(f"{py.name}:{lineno} → {mod}")
    assert not offenders, (
        "INV-5 violada — imports prohibidos desde rails:\n  "
        + "\n  ".join(offenders)
    )


def test_rails_only_imports_from_contracts_and_self() -> None:
    """Refuerzo: solo `clinibrium.contracts` y `clinibrium.rails` están permitidos."""
    pkg_root = Path(__file__).resolve().parents[1] / "clinibrium" / "rails"
    allowed_roots = {"clinibrium.contracts", "clinibrium.rails"}
    offenders: list[str] = []
    for py in sorted(pkg_root.glob("*.py")):
        for lineno, mod in _iter_imports(py):
            if not mod.startswith("clinibrium."):
                continue
            if not any(mod == a or mod.startswith(a + ".") for a in allowed_roots):
                offenders.append(f"{py.name}:{lineno} → {mod}")
    assert not offenders, (
        "Import cross-module no permitido desde rails:\n  "
        + "\n  ".join(offenders)
    )


# =========================================================================
# Sanity: _URGENCY_RANK usado internamente es accesible para tests
# =========================================================================


def test_urgency_rank_has_three_entries() -> None:
    assert len(_URGENCY_RANK) == 3
    for u in Urgency:
        assert u in _URGENCY_RANK


# ---------------------------------------------------------------------------
# Fixes de auditoría Gemini T8 (Bloque D más conservador + trazabilidad R-INV1)
# ---------------------------------------------------------------------------
class TestAuditFixesT8:
    def test_rinv1_traceable_even_if_forced_actions_empty(self) -> None:
        """Fix trazabilidad: red_flag_activa con forced_actions vacío igual
        registra R-INV1 y garantiza DERIVAR_URGENTE + urgencia inmediata."""
        result = _make_result(red_flag_activa=True, red_flag_actions=set())
        sealed = apply_rails(result, _make_features())
        assert sealed.urgency == Urgency.inmediata
        assert "R-INV1" in sealed.applied_rails
        assert ForcedAction.DERIVAR_URGENTE in sealed.forced_actions

    def test_empty_differential_blocks_epley(self) -> None:
        """Fix D1: diferencial vacío (sin BPPV confiable) ⇒ BLOQUEAR_EPLEY."""
        result = _make_result(candidates=[])
        sealed = apply_rails(result, _make_features())
        assert ForcedAction.BLOQUEAR_EPLEY in sealed.forced_actions

    def test_cervical_feature_blocks_epley(self) -> None:
        """Fix D4: contraindicación cervical directa en features ⇒ BLOQUEAR_EPLEY,
        aunque el top sea un bppv_posterior de score alto."""
        result = _make_result(
            candidates=[
                DifferentialCandidate(diagnosis=Diagnosis.bppv_posterior, score=0.95)
            ]
        )
        sealed = apply_rails(result, _make_features(cervical_pathology=True))
        assert ForcedAction.BLOQUEAR_EPLEY in sealed.forced_actions

    def test_direction_changing_nystagmus_blocks_epley_and_no_benigno(self) -> None:
        """Fix D2/central: nistagmo cambiante de dirección ⇒ BLOQUEAR_EPLEY + NO_BENIGNO."""
        result = _make_result(
            candidates=[
                DifferentialCandidate(diagnosis=Diagnosis.bppv_posterior, score=0.95)
            ]
        )
        sealed = apply_rails(
            result,
            _make_features(nystagmus_direction=NystagmusDirection.direction_changing),
        )
        assert ForcedAction.BLOQUEAR_EPLEY in sealed.forced_actions
        assert ForcedAction.NO_BENIGNO in sealed.forced_actions

    def test_direction_changing_gaze_bool_blocks_epley(self) -> None:
        """Fix central: el bool nystagmus_direction_changing_gaze también cuenta como atípico."""
        result = _make_result(
            candidates=[
                DifferentialCandidate(diagnosis=Diagnosis.bppv_posterior, score=0.95)
            ]
        )
        sealed = apply_rails(
            result, _make_features(nystagmus_direction_changing_gaze=True)
        )
        assert ForcedAction.BLOQUEAR_EPLEY in sealed.forced_actions
        assert ForcedAction.NO_BENIGNO in sealed.forced_actions
