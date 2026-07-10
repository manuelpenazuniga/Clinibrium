"""Tests del módulo `contracts` (hoja del grafo `clinibrium.*`).

Cubre los 4 criterios de aceptación de la tarea:
  (a) instanciar cada modelo con valores mínimos,
  (b) `CaseFeatures(extra_field=...)` levanta `ValidationError` (extra=forbid),
  (c) `NETWORK_SAFE_FIELDS` NO contiene PII keys (test negativo, INV-2),
  (d) `AuditEvent` es inmutable (`frozen=True`).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from clinibrium.contracts import (
    NETWORK_SAFE_FIELDS,
    ActorType,
    AuditEvent,
    CaseFeatures,
    Diagnosis,
    DifferentialCandidate,
    DifferentialResult,
    DixHallpikeResult,
    FocalSign,
    ForcedAction,
    HeadImpulse,
    HearingLoss,
    NystagmusDirection,
    Onset,
    PipelineResult,
    PredictResponse,
    ReasonerOutput,
    RedFlagHit,
    RedFlagResult,
    SymptomDuration,
    TimingPattern,
    Trigger,
    Urgency,
    VascularRiskFactor,
)

# =========================================================================
# Enums
# =========================================================================


def test_urgency_values() -> None:
    assert {u.value for u in Urgency} == {"inmediata", "prioritaria", "ambulatoria"}


def test_diagnosis_values() -> None:
    assert Diagnosis.central_suspected.value == "central_suspected"
    assert Diagnosis.undetermined.value == "undetermined"


def test_forced_action_values() -> None:
    assert ForcedAction.DERIVAR_URGENTE.value == "DERIVAR_URGENTE"
    assert ForcedAction.BLOQUEAR_EPLEY.value == "BLOQUEAR_EPLEY"


def test_actor_type_values() -> None:
    assert ActorType.system.value == "system"
    assert ActorType.clinician.value == "clinician"


# =========================================================================
# CaseFeatures — (a) instancia mínima; (b) extra=forbid; (c) allowlist sin PII
# =========================================================================


def test_case_features_minimal_defaults() -> None:
    f = CaseFeatures()
    assert f.nystagmus_direction == NystagmusDirection.none
    assert f.head_impulse == HeadImpulse.not_done
    assert f.hearing_loss == HearingLoss.none
    assert f.dix_hallpike == DixHallpikeResult.not_done
    assert f.focal_signs == set()
    assert f.vascular_risk_factors == set()
    assert f.worsening_during_flow is False
    # los opcionales quedan en None
    assert f.duration is None
    assert f.onset is None
    assert f.trigger is None
    assert f.timing_pattern is None
    assert f.age_years is None
    assert f.episode_count is None


def test_case_features_full_instantiation() -> None:
    f = CaseFeatures(
        duration=SymptomDuration.seconds,
        onset=Onset.sudden,
        trigger=Trigger.positional_head,
        timing_pattern=TimingPattern.episodic_triggered,
        nystagmus_direction=NystagmusDirection.torsional_pure,
        nystagmus_direction_changing_gaze=False,
        nystagmus_latency_s=2.0,
        nystagmus_duration_s=20.0,
        nystagmus_fatigable=True,
        nystagmus_suppressed_by_fixation=True,
        head_impulse=HeadImpulse.normal,
        skew_deviation=False,
        hearing_loss=HearingLoss.none,
        tinnitus=False,
        aural_fullness=False,
        focal_signs={FocalSign.dysarthria, FocalSign.diplopia},
        truncal_ataxia_severe=False,
        headache_neck_pain_sudden_severe=False,
        migrainous_features=False,
        age_years=60,
        vascular_risk_factors={VascularRiskFactor.hypertension},
        fever=False,
        neck_stiffness=False,
        altered_consciousness=False,
        presyncope_syncope=False,
        palpitations=False,
        chest_pain=False,
        otitis_mastoiditis=False,
        recent_head_neck_trauma=False,
        cervical_pathology=False,
        known_carotid_vertebrobasilar_disease=False,
        cardiovascular_instability=False,
        dix_hallpike=DixHallpikeResult.right_positive,
        torsion_confirmed_by_clinician=True,
        episode_count=3,
        episode_duration=SymptomDuration.hours,
        worsening_during_flow=False,
    )
    assert f.duration == SymptomDuration.seconds
    assert f.dix_hallpike == DixHallpikeResult.right_positive
    assert FocalSign.diplopia in f.focal_signs
    assert VascularRiskFactor.hypertension in f.vascular_risk_factors
    assert f.torsion_confirmed_by_clinician is True


def test_case_features_rejects_extra_field() -> None:
    """(b) extra=forbid — INV-2: cualquier campo fuera del allowlist explota."""
    with pytest.raises(ValidationError):
        CaseFeatures(extra_field=1)


def test_case_features_rejects_pii_keys() -> None:
    """INV-2: PII keys explotas en construcción, aunque sean del allowlist."""
    for pii_key in ("name", "rut", "dob", "address", "notes", "patient_id"):
        with pytest.raises(ValidationError):
            CaseFeatures(**{pii_key: "leaked"})  # type: ignore[arg-type]


def test_network_safe_fields_excludes_pii_keys() -> None:
    """(c) test negativo de allowlist (INV-2)."""
    forbidden = {
        "name",
        "rut",
        "dob",
        "address",
        "notes",
        "free_text",
        "video",
        "frames",
        "patient_id",
    }
    leaked = forbidden & set(NETWORK_SAFE_FIELDS)
    assert not leaked, f"NETWORK_SAFE_FIELDS filtra keys de PII: {leaked}"


def test_network_safe_fields_matches_case_features_fields() -> None:
    assert NETWORK_SAFE_FIELDS == frozenset(CaseFeatures.model_fields.keys())


# =========================================================================
# Results
# =========================================================================


def test_red_flag_result_minimal() -> None:
    r = RedFlagResult(red_flag_activa=False)
    assert r.red_flag_activa is False
    assert r.hits == []
    assert r.forced_actions == set()


def test_red_flag_hit_minimal() -> None:
    h = RedFlagHit(
        id="A1",
        label="AVS con focal signs",
        forced_actions=[ForcedAction.DERIVAR_URGENTE, ForcedAction.NO_BENIGNO],
        severity="high",
    )
    assert h.id == "A1"
    assert h.severity == "high"
    assert ForcedAction.DERIVAR_URGENTE in h.forced_actions


def test_red_flag_hit_rejects_unknown_severity() -> None:
    with pytest.raises(ValidationError):
        RedFlagHit(
            id="A1",
            label="x",
            forced_actions=[ForcedAction.ESCALAR],
            severity="critical",  # type: ignore[arg-type]
        )


def test_differential_result_candidates_ordered() -> None:
    d = DifferentialResult(
        candidates=[
            DifferentialCandidate(
                diagnosis=Diagnosis.bppv_posterior, score=0.9, rule_ids=["R1", "R2"]
            ),
            DifferentialCandidate(diagnosis=Diagnosis.vestibular_neuritis, score=0.4),
            DifferentialCandidate(diagnosis=Diagnosis.undetermined, score=0.1),
        ]
    )
    assert len(d.candidates) == 3
    assert d.candidates[0].score >= d.candidates[1].score >= d.candidates[2].score
    assert d.candidates[0].diagnosis == Diagnosis.bppv_posterior
    assert d.candidates[0].rule_ids == ["R1", "R2"]


def test_predict_response_minimal() -> None:
    p = PredictResponse(
        probabilities={"bppv_posterior": 0.8, "meniere": 0.1},
        model_version="catboost-v0.1",
    )
    assert p.shap is None
    assert p.model_version == "catboost-v0.1"
    assert p.probabilities["bppv_posterior"] == 0.8


def test_reasoner_output_minimal() -> None:
    r = ReasonerOutput(
        explanation="...",
        reconciliation="...",
        model_used="claude-opus-4-8",
    )
    assert r.suggested_next_steps == []
    assert r.model_used == "claude-opus-4-8"
    assert r.reasoner_suggested_urgency is None


def test_reasoner_output_with_suggested_urgency() -> None:
    """AD-11: `reasoner_suggested_urgency` es un enum Urgency estructurado."""
    r = ReasonerOutput(
        explanation="Caso dudoso.",
        reconciliation="Las features no cuadran con VPPB típico.",
        model_used="claude-opus-4-8",
        reasoner_suggested_urgency=Urgency.inmediata,
    )
    assert r.reasoner_suggested_urgency == Urgency.inmediata


def test_pipeline_result_minimal() -> None:
    pr = PipelineResult(
        case_id="case-1",
        urgency=Urgency.ambulatoria,
        red_flag=RedFlagResult(red_flag_activa=False),
        differential=DifferentialResult(
            candidates=[DifferentialCandidate(diagnosis=Diagnosis.bppv_posterior, score=0.7)],
        ),
    )
    assert pr.case_id == "case-1"
    assert pr.urgency == Urgency.ambulatoria
    assert pr.ml is None
    assert pr.reasoning is None
    assert pr.audit_event_id is None
    assert pr.forced_actions == set()
    assert pr.applied_rails == []


# =========================================================================
# AuditEvent — (d) inmutable (INV-4)
# =========================================================================


def test_audit_event_minimal() -> None:
    e = AuditEvent(
        id="evt-1",
        occurred_at=datetime(2026, 7, 10, 12, 0, 0, tzinfo=timezone.utc),
        event_type="pipeline_evaluation",
        input_features_hash="sha256:deadbeef",
        urgency=Urgency.ambulatoria,
        red_flag_activa=False,
        outcome_summary="BPPV posterior probable; sin red flags.",
    )
    assert e.actor == ActorType.system
    assert e.model_used is None
    assert e.forced_actions == []
    assert e.red_flag_activa is False


def test_audit_event_full() -> None:
    e = AuditEvent(
        id="evt-2",
        occurred_at=datetime(2026, 7, 10, 12, 0, 0, tzinfo=timezone.utc),
        event_type="pipeline_evaluation",
        actor=ActorType.clinician,
        model_used="claude-opus-4-8",
        input_features_hash="sha256:abc",
        urgency=Urgency.inmediata,
        forced_actions=[ForcedAction.DERIVAR_URGENTE, ForcedAction.NO_BENIGNO],
        red_flag_activa=True,
        outcome_summary="AVS + focal signs → derivar.",
    )
    assert e.actor == ActorType.clinician
    assert e.urgency == Urgency.inmediata
    assert e.red_flag_activa is True
    assert ForcedAction.DERIVAR_URGENTE in e.forced_actions


def test_audit_event_is_frozen() -> None:
    """(d) AuditEvent es inmutable (frozen=True, INV-4)."""
    e = AuditEvent(
        id="evt-3",
        occurred_at=datetime(2026, 7, 10, 12, 0, 0, tzinfo=timezone.utc),
        event_type="pipeline_evaluation",
        input_features_hash="sha256:abc",
        urgency=Urgency.ambulatoria,
        red_flag_activa=False,
        outcome_summary="ok",
    )
    with pytest.raises(ValidationError):
        e.urgency = Urgency.inmediata  # type: ignore[misc]
