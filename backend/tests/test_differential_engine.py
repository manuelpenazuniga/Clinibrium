"""Tests for the DifferentialEngine — ICVD rules as data, deterministic scoring.

Covers the task's acceptance criteria:
  - Typical benign BPPV ⇒ top == bppv_posterior with a high score.
  - Spontaneous AVS (abnormal head-impulse + horizontal nystagmus + no
    hearing loss) ⇒ top == vestibular_neuritis.
  - Spontaneous episodic + fluctuating hearing loss + tinnitus + fullness
    ⇒ top == meniere.
  - Central pattern (normal head_impulse + pure vertical nystagmus +
    skew) ⇒ central_suspected among the top.
  - Determinism: two identical evaluations.
  - Candidates ordered descending; none with score 0.
  - INV-5: the module does NOT import the forbidden ones (redflag_engine,
    reasoner, ml_client, orchestrator).
"""
from __future__ import annotations

import inspect
from pathlib import Path

from clinibrium.contracts import (
    CaseFeatures,
    Diagnosis,
    DifferentialResult,
    DixHallpikeResult,
    FocalSign,
    HeadImpulse,
    HearingLoss,
    NystagmusDirection,
    SymptomDuration,
    TimingPattern,
    Trigger,
)
from clinibrium.differential_engine import CRITERIA, DiagnosisCriterion, evaluate

# =========================================================================
# Synthetic clinical case helpers
# =========================================================================


def _bppv_posterior_typical() -> CaseFeatures:
    """Benign posterior-canal BPPV: positional, <1min, Dix-Hallpike +,
    fatigable, 5s latency, torsion confirmed by clinician.
    """
    return CaseFeatures(
        trigger=Trigger.positional_head,
        duration=SymptomDuration.under_1min,
        dix_hallpike=DixHallpikeResult.right_positive,
        nystagmus_fatigable=True,
        nystagmus_latency_s=5.0,
        torsion_confirmed_by_clinician=True,
    )


def _avs_periferico_typical() -> CaseFeatures:
    """Spontaneous AVS: acute timing, abnormal head-impulse, horizontal
    nystagmus, no hearing loss. ⇒ vestibular_neuritis.
    """
    return CaseFeatures(
        timing_pattern=TimingPattern.acute_continuous,
        trigger=Trigger.spontaneous,
        nystagmus_direction=NystagmusDirection.horizontal,
        head_impulse=HeadImpulse.abnormal_corrective_saccade,
        hearing_loss=HearingLoss.none,
    )


def _meniere_typical() -> CaseFeatures:
    """Ménière: spontaneous episodic, hours-long episodes, fluctuating
    hearing loss, tinnitus, aural fullness.
    """
    return CaseFeatures(
        timing_pattern=TimingPattern.episodic_spontaneous,
        episode_duration=SymptomDuration.hours,
        hearing_loss=HearingLoss.fluctuating,
        tinnitus=True,
        aural_fullness=True,
    )


def _central_suspected_typical() -> CaseFeatures:
    """Central pattern: AVS + normal head_impulse + pure vertical nystagmus
    + skew deviation + focal sign (dysarthria).
    """
    return CaseFeatures(
        timing_pattern=TimingPattern.acute_continuous,
        head_impulse=HeadImpulse.normal,
        nystagmus_direction=NystagmusDirection.vertical_pure,
        skew_deviation=True,
        focal_signs={FocalSign.dysarthria},
    )


# =========================================================================
# Clinical scenarios (acceptance criteria)
# =========================================================================


def test_bppv_posterior_top_candidate() -> None:
    result = evaluate(_bppv_posterior_typical())
    assert result.candidates, "typical BPPV must yield at least one candidate"
    top = result.candidates[0]
    assert top.diagnosis == Diagnosis.bppv_posterior
    # 6/6 criteria match ⇒ score == 1.0
    assert top.score >= 0.95
    # All 6 posterior-BPPV IDs must appear in rule_ids.
    assert len(top.rule_ids) == 6


def test_vestibular_neuritis_top_candidate() -> None:
    result = evaluate(_avs_periferico_typical())
    assert result.candidates
    assert result.candidates[0].diagnosis == Diagnosis.vestibular_neuritis
    # 5/5 criteria match ⇒ score == 1.0
    assert result.candidates[0].score >= 0.95


def test_meniere_top_candidate() -> None:
    result = evaluate(_meniere_typical())
    assert result.candidates
    assert result.candidates[0].diagnosis == Diagnosis.meniere
    # 5/5 criteria match ⇒ score == 1.0
    assert result.candidates[0].score >= 0.95


def test_central_suspected_in_top() -> None:
    result = evaluate(_central_suspected_typical())
    assert result.candidates
    top_3_dx = [c.diagnosis for c in result.candidates[:3]]
    assert Diagnosis.central_suspected in top_3_dx
    # central_suspected matches 4/6 criteria (no changing_gaze, no ataxia).
    cs = next(c for c in result.candidates if c.diagnosis == Diagnosis.central_suspected)
    assert cs.score > 0.5


# =========================================================================
# Engine properties
# =========================================================================


def test_determinism_two_evaluations_equal() -> None:
    f = _bppv_posterior_typical()
    r1 = evaluate(f)
    r2 = evaluate(f)
    assert r1 == r2


def test_determinism_independent_case_features_instances() -> None:
    """Two equivalent CaseFeatures (same fields) ⇒ equal results."""
    f1 = _meniere_typical()
    f2 = CaseFeatures(
        timing_pattern=TimingPattern.episodic_spontaneous,
        episode_duration=SymptomDuration.hours,
        hearing_loss=HearingLoss.fluctuating,
        tinnitus=True,
        aural_fullness=True,
    )
    r1 = evaluate(f1)
    r2 = evaluate(f2)
    assert r1 == r2


def test_candidates_ordered_desc_no_zero_scores() -> None:
    f = _bppv_posterior_typical()
    result = evaluate(f)
    scores = [c.score for c in result.candidates]
    assert scores == sorted(scores, reverse=True)
    assert scores, "there must be at least one candidate with matching criteria"
    assert all(c.score > 0 for c in result.candidates)


def test_returns_differential_result_type() -> None:
    f = CaseFeatures()
    result = evaluate(f)
    assert isinstance(result, DifferentialResult)


def test_minimal_case_features_only_default_match() -> None:
    """`CaseFeatures()` with all defaults: only matches criteria that depend
    on non-`None` defaults (e.g. `hearing_loss == none` from the
    vestibular_neuritis criterion, given that `hearing_loss` defaults to
    `HearingLoss.none`). BPPV, central, meniere and cardiogenic must NOT
    appear — no spurious evidence.
    """
    result = evaluate(CaseFeatures())
    diagnoses = {c.diagnosis for c in result.candidates}
    assert Diagnosis.bppv_posterior not in diagnoses
    assert Diagnosis.bppv_horizontal not in diagnoses
    assert Diagnosis.meniere not in diagnoses
    assert Diagnosis.vestibular_migraine not in diagnoses
    assert Diagnosis.labyrinthitis not in diagnoses
    assert Diagnosis.central_suspected not in diagnoses
    assert Diagnosis.cardiogenic_suspected not in diagnoses
    # Only expected match: vestibular_neuritis via hearing_loss==none (default).
    assert diagnoses == {Diagnosis.vestibular_neuritis}
    assert result.candidates[0].score < 0.2  # very weak evidence


def test_bppv_posterior_rule_ids_traced() -> None:
    """The top match's rule_ids are the IDs from the CRITERIA table."""
    result = evaluate(_bppv_posterior_typical())
    top = result.candidates[0]
    bppv_ids = {c.id for c in CRITERIA if c.diagnosis == Diagnosis.bppv_posterior}
    assert set(top.rule_ids) == bppv_ids


# =========================================================================
# INV-5: the module does NOT import the forbidden ones.
# Additional check on top of the CI grep: the module's AST contains
# no import of redflag_engine / reasoner / ml_client / orchestrator.
# =========================================================================


def test_no_forbidden_imports() -> None:
    """INV-5: no `import`/`from ... import` statement in the package points
    at the forbidden modules. We check the AST to avoid confusing imports
    with mentions in docstrings.
    """
    import ast

    forbidden = ("redflag_engine", "reasoner", "ml_client", "orchestrator")
    pkg_dir = Path(inspect.getfile(evaluate)).parent  # differential_engine/
    offenders: list[str] = []
    for py in sorted(pkg_dir.glob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for bad in forbidden:
                        if bad in alias.name:
                            offenders.append(
                                f"{py.name}:{node.lineno} import {alias.name}"
                            )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for bad in forbidden:
                    if bad in module or bad in node.module or "":
                        offenders.append(
                            f"{py.name}:{node.lineno} from {module} import ..."
                        )
                        break
                for alias in node.names:
                    for bad in forbidden:
                        if bad in alias.name:
                            offenders.append(
                                f"{py.name}:{node.lineno} from {module} import {alias.name}"
                            )
    assert not offenders, "INV-5 violated: " + "; ".join(offenders)


def test_criteria_are_frozen_dataclasses() -> None:
    """`DiagnosisCriterion` is frozen ⇒ cannot be mutated after construction."""
    from dataclasses import FrozenInstanceError

    c: DiagnosisCriterion = CRITERIA[0]
    try:
        c.weight = 999.0  # type: ignore[misc]
    except FrozenInstanceError:
        return
    raise AssertionError("DiagnosisCriterion should be frozen")


# =========================================================================
# Negative: the engine does NOT touch urgency nor recommend treatment.
# We verify the output is just a pool of candidates, with no urgency,
# forced-action or therapeutic next-steps fields.
# =========================================================================


def test_result_does_not_carry_urgency_or_treatment() -> None:
    """INV-3: the differential is just a pool; urgency/action are sealed by
    RedFlagEngine + rails. We verify at the shape level that
    `DifferentialResult` does not expose those fields.
    """
    from clinibrium.contracts.results import DifferentialResult as DR

    fields = set(DR.model_fields.keys())
    forbidden_field_markers = {
        "urgency",
        "treatment",
        "forced_actions",
        "next_steps",
        "red_flag",
    }
    leaked = forbidden_field_markers & fields
    assert not leaked, f"DifferentialResult exposes forbidden fields: {leaked}"
