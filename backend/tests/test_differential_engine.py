"""Tests del DifferentialEngine — reglas ICVD como datos, scoring determinista.

Cubre los criterios de aceptación de la tarea:
  - BPPV benigno típico ⇒ top == bppv_posterior con score alto.
  - AVS espontáneo (head-impulse abnormal + nistagmo horizontal + sin
    hipoacusia) ⇒ top == vestibular_neuritis.
  - Episódico espontáneo + hipoacusia fluctuante + tinnitus + plenitud
    ⇒ top == meniere.
  - Patrón central (head_impulse normal + nistagmo vertical puro +
    skew) ⇒ central_suspected entre los top.
  - Determinismo: dos evaluaciones iguales.
  - Candidatos ordenados desc; ninguno con score 0.
  - INV-5: el módulo NO importa los prohibidos (redflag_engine,
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
# Helpers de casos clínicos sintéticos
# =========================================================================


def _bppv_posterior_typical() -> CaseFeatures:
    """BPPV canal posterior benigno: positional, <1min, Dix-Hallpike +,
    fatigable, latencia 5s, torsión confirmada por clínico.
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
    """AVS espontáneo: timing agudo, head-impulse abnormal, nistagmo
    horizontal, sin hipoacusia. ⇒ vestibular_neuritis.
    """
    return CaseFeatures(
        timing_pattern=TimingPattern.acute_continuous,
        trigger=Trigger.spontaneous,
        nystagmus_direction=NystagmusDirection.horizontal,
        head_impulse=HeadImpulse.abnormal_corrective_saccade,
        hearing_loss=HearingLoss.none,
    )


def _meniere_typical() -> CaseFeatures:
    """Ménière: episódico espontáneo, episodios de horas, hipoacusia
    fluctuante, tinnitus, plenitud aural.
    """
    return CaseFeatures(
        timing_pattern=TimingPattern.episodic_spontaneous,
        episode_duration=SymptomDuration.hours,
        hearing_loss=HearingLoss.fluctuating,
        tinnitus=True,
        aural_fullness=True,
    )


def _central_suspected_typical() -> CaseFeatures:
    """Patrón central: AVS + head_impulse normal + nistagmo vertical puro
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
# Escenarios clínicos (criterios de aceptación)
# =========================================================================


def test_bppv_posterior_top_candidate() -> None:
    result = evaluate(_bppv_posterior_typical())
    assert result.candidates, "BPPV típico debe generar al menos un candidato"
    top = result.candidates[0]
    assert top.diagnosis == Diagnosis.bppv_posterior
    # 6/6 criterios matchean ⇒ score == 1.0
    assert top.score >= 0.95
    # Los 6 IDs del BPPV posterior deben figurar en rule_ids.
    assert len(top.rule_ids) == 6


def test_vestibular_neuritis_top_candidate() -> None:
    result = evaluate(_avs_periferico_typical())
    assert result.candidates
    assert result.candidates[0].diagnosis == Diagnosis.vestibular_neuritis
    # 5/5 criterios matchean ⇒ score == 1.0
    assert result.candidates[0].score >= 0.95


def test_meniere_top_candidate() -> None:
    result = evaluate(_meniere_typical())
    assert result.candidates
    assert result.candidates[0].diagnosis == Diagnosis.meniere
    # 5/5 criterios matchean ⇒ score == 1.0
    assert result.candidates[0].score >= 0.95


def test_central_suspected_in_top() -> None:
    result = evaluate(_central_suspected_typical())
    assert result.candidates
    top_3_dx = [c.diagnosis for c in result.candidates[:3]]
    assert Diagnosis.central_suspected in top_3_dx
    # central_suspected matchea 4/6 criterios (no changing_gaze, no ataxia).
    cs = next(c for c in result.candidates if c.diagnosis == Diagnosis.central_suspected)
    assert cs.score > 0.5


# =========================================================================
# Propiedades del engine
# =========================================================================


def test_determinism_two_evaluations_equal() -> None:
    f = _bppv_posterior_typical()
    r1 = evaluate(f)
    r2 = evaluate(f)
    assert r1 == r2


def test_determinism_independent_case_features_instances() -> None:
    """Dos CaseFeatures equivalentes (mismos campos) ⇒ resultados iguales."""
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
    assert scores, "debe haber al menos un candidato con criterios matcheando"
    assert all(c.score > 0 for c in result.candidates)


def test_returns_differential_result_type() -> None:
    f = CaseFeatures()
    result = evaluate(f)
    assert isinstance(result, DifferentialResult)


def test_minimal_case_features_only_default_match() -> None:
    """`CaseFeatures()` con todos los defaults: solo matchea criterios que
    dependen de defaults no-`None` (p.ej. `hearing_loss == none` del criterio
    vestibular_neuritis, dado que el default de `hearing_loss` es
    `HearingLoss.none`). NO debe aparecer BPPV, ni central, ni meniere, ni
    cardiogenic — evidencia espuria ausente.
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
    # Único match esperado: vestibular_neuritis por hearing_loss==none (default).
    assert diagnoses == {Diagnosis.vestibular_neuritis}
    assert result.candidates[0].score < 0.2  # evidencia muy débil


def test_bppv_posterior_rule_ids_traced() -> None:
    """Los rule_ids del top match son los IDs de la tabla CRITERIA."""
    result = evaluate(_bppv_posterior_typical())
    top = result.candidates[0]
    bppv_ids = {c.id for c in CRITERIA if c.diagnosis == Diagnosis.bppv_posterior}
    assert set(top.rule_ids) == bppv_ids


# =========================================================================
# INV-5: el módulo NO importa los prohibidos.
# Verificación adicional al grep del CI: el AST del módulo no contiene
# ningún import de redflag_engine / reasoner / ml_client / orchestrator.
# =========================================================================


def test_no_forbidden_imports() -> None:
    """INV-5: ninguna sentencia `import`/`from ... import` del paquete
    apunta a los módulos prohibidos. Verificamos sobre el AST para no
    confundir con menciones en docstrings.
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
    assert not offenders, "INV-5 violado: " + "; ".join(offenders)


def test_criteria_are_frozen_dataclasses() -> None:
    """`DiagnosisCriterion` es frozen ⇒ no se puede mutar post-construcción."""
    from dataclasses import FrozenInstanceError

    c: DiagnosisCriterion = CRITERIA[0]
    try:
        c.weight = 999.0  # type: ignore[misc]
    except FrozenInstanceError:
        return
    raise AssertionError("DiagnosisCriterion debería ser frozen")


# =========================================================================
# Negativo: el engine NO toca urgencia ni recomienda tratamiento.
# Verificamos que el output es solo un pool de candidatos, sin campos
# de urgencia, de acción forzada ni de next-steps terapéuticos.
# =========================================================================


def test_result_does_not_carry_urgency_or_treatment() -> None:
    """INV-3: el diferencial es solo un pool; urgencia/acción las sellan
    RedFlagEngine + rails. Verificamos a nivel de shape que
    `DifferentialResult` no expone esos campos.
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
    assert not leaked, f"DifferentialResult expone campos prohibidos: {leaked}"
