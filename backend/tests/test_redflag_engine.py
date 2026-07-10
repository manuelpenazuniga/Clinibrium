"""Tests del `RedFlagEngine` (T4) — INV-5 + cobertura de la tabla `RULES`.

Cubre los 4 criterios de aceptación de la tarea:
  1. Un test POSITIVO por CADA regla de `RULES` (id, label, severity,
     forced_actions correctos en el hit correspondiente).
  2. Test NEGATIVO base: caso de BPPV benigno típico → `red_flag_activa
     == False` y sin `DERIVAR_URGENTE`.
  3. `red_flag_activa` True/False según haya o no `DERIVAR_URGENTE`.
  4. Determinismo: mismas features → mismo resultado.
  5. INV-5: `redflag_engine` NO importa `differential_engine` /
     `reasoner` / `ml_client` / `orchestrator`. Solo `contracts` (y
     submódulos propios del paquete).
"""
from __future__ import annotations

import ast
import difflib
from pathlib import Path

from clinibrium.contracts import (
    CaseFeatures,
    DixHallpikeResult,
    FocalSign,
    ForcedAction,
    HeadImpulse,
    HearingLoss,
    NystagmusDirection,
    Onset,
    SymptomDuration,
    TimingPattern,
    Trigger,
    VascularRiskFactor,
)
from clinibrium.redflag_engine import (
    AGE_CENTRAL_THRESHOLD,
    RULES,
    RedFlagRule,
    evaluate,
)

# =========================================================================
# Helpers de construcción de casos (minimal features por regla)
# =========================================================================


def _bppv_benign() -> CaseFeatures:
    """Caso negativo base: BPPV de canal posterior, típico, benigno.

    Características:
      - duración < 1 min
      - gatillado por posición de la cabeza
      - patrón temporal episódico-disparado (NO AVS)
      - nistagmo upbeating-torsional (mixed) con latencia y fatigabilidad
      - Dix-Hallpike positivo, torsión confirmada por clínico
      - sin signos focales, sin factores de riesgo vascular
    """
    return CaseFeatures(
        duration=SymptomDuration.under_1min,
        onset=Onset.sudden,
        trigger=Trigger.positional_head,
        timing_pattern=TimingPattern.episodic_triggered,
        nystagmus_direction=NystagmusDirection.mixed,
        nystagmus_latency_s=2.0,
        nystagmus_duration_s=20.0,
        nystagmus_fatigable=True,
        nystagmus_suppressed_by_fixation=True,
        dix_hallpike=DixHallpikeResult.right_positive,
        torsion_confirmed_by_clinician=True,
        episode_count=3,
        episode_duration=SymptomDuration.under_1min,
    )


def _hit(result, rule_id: str):
    matches = [h for h in result.hits if h.id == rule_id]
    assert matches, (
        f"regla {rule_id} no presente en hits={[h.id for h in result.hits]}"
    )
    return matches[0]


# =========================================================================
# Tests POSITIVOS — uno por cada regla de RULES
# =========================================================================


def test_a1_avs_central_hints() -> None:
    f = CaseFeatures(
        timing_pattern=TimingPattern.acute_continuous,
        head_impulse=HeadImpulse.normal,
    )
    r = evaluate(f)
    h = _hit(r, "A1")
    assert h.label == "AVS con HINTS sospechoso de central"
    assert h.severity == "high"
    assert set(h.forced_actions) == {ForcedAction.NO_BENIGNO, ForcedAction.DERIVAR_URGENTE}
    assert r.red_flag_activa is True


def test_a2_pure_vertical_or_torsional_nystagmus() -> None:
    f = CaseFeatures(nystagmus_direction=NystagmusDirection.vertical_pure)
    r = evaluate(f)
    h = _hit(r, "A2")
    assert h.severity == "high"
    assert h.forced_actions == [ForcedAction.DERIVAR_URGENTE]
    assert r.red_flag_activa is True


def test_a2_fires_also_for_torsional_pure() -> None:
    f = CaseFeatures(nystagmus_direction=NystagmusDirection.torsional_pure)
    r = evaluate(f)
    assert any(h.id == "A2" for h in r.hits)
    assert r.red_flag_activa is True


def test_a3_direction_changing_nystagmus() -> None:
    f = CaseFeatures(nystagmus_direction=NystagmusDirection.direction_changing)
    r = evaluate(f)
    h = _hit(r, "A3")
    assert h.severity == "high"
    assert h.forced_actions == [ForcedAction.DERIVAR_URGENTE]
    assert r.red_flag_activa is True


def test_a3_fires_also_for_changing_on_gaze_flag() -> None:
    f = CaseFeatures(
        timing_pattern=TimingPattern.acute_continuous,
        nystagmus_direction_changing_gaze=True,
    )
    r = evaluate(f)
    assert any(h.id == "A3" for h in r.hits)


def test_a4_severe_truncal_ataxia() -> None:
    f = CaseFeatures(truncal_ataxia_severe=True)
    r = evaluate(f)
    h = _hit(r, "A4")
    assert h.severity == "high"
    assert h.forced_actions == [ForcedAction.DERIVAR_URGENTE]
    assert r.red_flag_activa is True


def test_a5_any_focal_sign() -> None:
    f = CaseFeatures(focal_signs={FocalSign.dysarthria})
    r = evaluate(f)
    h = _hit(r, "A5")
    assert h.severity == "high"
    assert h.forced_actions == [ForcedAction.DERIVAR_URGENTE]
    assert r.red_flag_activa is True


def test_a5_fires_for_each_focal_sign() -> None:
    for sign in FocalSign:
        r = evaluate(CaseFeatures(focal_signs={sign}))
        assert any(h.id == "A5" for h in r.hits), f"A5 no disparó con focal_sign={sign}"


def test_a6_sudden_severe_headache_or_neck_pain() -> None:
    f = CaseFeatures(headache_neck_pain_sudden_severe=True)
    r = evaluate(f)
    h = _hit(r, "A6")
    assert h.severity == "high"
    assert h.forced_actions == [ForcedAction.DERIVAR_URGENTE]
    assert r.red_flag_activa is True


def test_a7_avs_age_vascular_risk() -> None:
    """A7 es medium: ESCALAR + NO_BENIGNO, NO activa red_flag por sí sola."""
    f = CaseFeatures(
        timing_pattern=TimingPattern.acute_continuous,
        age_years=AGE_CENTRAL_THRESHOLD,
        vascular_risk_factors={VascularRiskFactor.hypertension},
    )
    r = evaluate(f)
    h = _hit(r, "A7")
    assert h.severity == "medium"
    assert set(h.forced_actions) == {ForcedAction.NO_BENIGNO, ForcedAction.ESCALAR}
    # medium no fuerza derivación → red_flag_activa debe ser False en aislamiento
    assert r.red_flag_activa is False
    assert ForcedAction.ESCALAR in r.forced_actions
    assert ForcedAction.NO_BENIGNO in r.forced_actions


def test_a7_does_not_fire_below_age_threshold() -> None:
    f = CaseFeatures(
        timing_pattern=TimingPattern.acute_continuous,
        age_years=AGE_CENTRAL_THRESHOLD - 1,
        vascular_risk_factors={VascularRiskFactor.hypertension},
    )
    r = evaluate(f)
    assert all(h.id != "A7" for h in r.hits)


def test_a7_does_not_fire_without_risk_factor() -> None:
    f = CaseFeatures(
        timing_pattern=TimingPattern.acute_continuous,
        age_years=AGE_CENTRAL_THRESHOLD + 5,
        vascular_risk_factors=set(),
    )
    r = evaluate(f)
    assert all(h.id != "A7" for h in r.hits)


def test_a8_sudden_unilateral_hearing_loss_with_avs() -> None:
    """A8 es la combinación AICA (súbita + AVS). B1 también dispara aquí por
    diseño (B1 cubre la súbita aislada o con vértigo); verificamos que A8 está
    con su acción específica."""
    f = CaseFeatures(
        timing_pattern=TimingPattern.acute_continuous,
        hearing_loss=HearingLoss.sudden_unilateral,
    )
    r = evaluate(f)
    h = _hit(r, "A8")
    assert h.severity == "high"
    assert h.forced_actions == [ForcedAction.DERIVAR_URGENTE]
    assert r.red_flag_activa is True


def test_b1_sudden_unilateral_hearing_loss_isolated() -> None:
    """B1 cubre la súbita aislada (sin AVS) para no asumir laberintitis."""
    f = CaseFeatures(hearing_loss=HearingLoss.sudden_unilateral)
    r = evaluate(f)
    h = _hit(r, "B1")
    assert h.severity == "high"
    assert h.forced_actions == [ForcedAction.DERIVAR_URGENTE]
    assert r.red_flag_activa is True
    # A8 NO debe disparar sin AVS
    assert all(h.id != "A8" for h in r.hits)


def test_b2_meningismus_or_altered_consciousness() -> None:
    f = CaseFeatures(fever=True, neck_stiffness=True)
    r = evaluate(f)
    h = _hit(r, "B2")
    assert h.severity == "high"
    assert h.forced_actions == [ForcedAction.DERIVAR_URGENTE]
    assert r.red_flag_activa is True


def test_b2_fires_with_altered_consciousness() -> None:
    f = CaseFeatures(fever=True, altered_consciousness=True)
    r = evaluate(f)
    assert any(h.id == "B2" for h in r.hits)
    assert r.red_flag_activa is True


def test_b3_cardiogenic_pattern_escalates_only() -> None:
    """B3 es medium: solo ESCALAR; no debe activar red_flag por sí solo."""
    f = CaseFeatures(presyncope_syncope=True)
    r = evaluate(f)
    h = _hit(r, "B3")
    assert h.severity == "medium"
    assert h.forced_actions == [ForcedAction.ESCALAR]
    assert r.red_flag_activa is False
    assert ForcedAction.ESCALAR in r.forced_actions


def test_b4_otitis_or_mastoiditis() -> None:
    f = CaseFeatures(otitis_mastoiditis=True)
    r = evaluate(f)
    h = _hit(r, "B4")
    assert h.severity == "high"
    assert h.forced_actions == [ForcedAction.DERIVAR_URGENTE]
    assert r.red_flag_activa is True


def test_b5_recent_head_neck_trauma() -> None:
    """B5 medium: PRECAUCION_EXAMEN + ESCALAR; no activa red_flag por sí solo."""
    f = CaseFeatures(recent_head_neck_trauma=True)
    r = evaluate(f)
    h = _hit(r, "B5")
    assert h.severity == "medium"
    assert set(h.forced_actions) == {
        ForcedAction.PRECAUCION_EXAMEN,
        ForcedAction.ESCALAR,
    }
    assert r.red_flag_activa is False


def test_c1_cervical_pathology() -> None:
    f = CaseFeatures(cervical_pathology=True)
    r = evaluate(f)
    h = _hit(r, "C1")
    assert h.severity == "medium"
    assert h.forced_actions == [ForcedAction.PRECAUCION_EXAMEN]
    assert r.red_flag_activa is False


def test_c2_known_carotid_vertebrobasilar_disease() -> None:
    f = CaseFeatures(known_carotid_vertebrobasilar_disease=True)
    r = evaluate(f)
    h = _hit(r, "C2")
    assert h.severity == "medium"
    assert h.forced_actions == [ForcedAction.PRECAUCION_EXAMEN]
    assert r.red_flag_activa is False


def test_c3_cardiovascular_instability() -> None:
    f = CaseFeatures(cardiovascular_instability=True)
    r = evaluate(f)
    h = _hit(r, "C3")
    assert h.severity == "medium"
    assert h.forced_actions == [ForcedAction.PRECAUCION_EXAMEN]
    assert r.red_flag_activa is False


def test_e4_worsening_during_flow() -> None:
    f = CaseFeatures(worsening_during_flow=True)
    r = evaluate(f)
    h = _hit(r, "E4")
    assert h.severity == "high"
    assert h.forced_actions == [ForcedAction.DERIVAR_URGENTE]
    assert r.red_flag_activa is True


# =========================================================================
# Test NEGATIVO base — BPPV benigno típico
# =========================================================================


def test_bppv_benign_no_red_flag() -> None:
    r = evaluate(_bppv_benign())
    assert r.red_flag_activa is False
    assert r.hits == []
    assert r.forced_actions == set()
    assert ForcedAction.DERIVAR_URGENTE not in r.forced_actions


# =========================================================================
# red_flag_activa ↔ DERIVAR_URGENTE
# =========================================================================


def test_red_flag_activa_false_when_only_escalate_or_precaucion() -> None:
    """Combinación de reglas medium (sin DERIVAR_URGENTE) no debe activar
    red_flag_activa aunque las forced_actions del resultado sean no-vacías."""
    f = CaseFeatures(
        cervical_pathology=True,  # C1
        cardiovascular_instability=True,  # C3
        presyncope_syncope=True,  # B3
    )
    r = evaluate(f)
    assert r.hits  # hay hits
    assert r.red_flag_activa is False
    assert ForcedAction.DERIVAR_URGENTE not in r.forced_actions
    assert ForcedAction.PRECAUCION_EXAMEN in r.forced_actions
    assert ForcedAction.ESCALAR in r.forced_actions


def test_red_flag_activa_true_with_any_derivar_urgente() -> None:
    """Una sola regla con DERIVAR_URGENTE ya activa red_flag, aunque haya
    también reglas medium en el mismo caso."""
    f = CaseFeatures(
        cervical_pathology=True,  # C1 — precaution, no red flag por sí sola
        focal_signs={FocalSign.dysarthria},  # A5 — DERIVAR_URGENTE
    )
    r = evaluate(f)
    assert r.red_flag_activa is True
    assert ForcedAction.DERIVAR_URGENTE in r.forced_actions
    assert any(h.id == "A5" for h in r.hits)
    assert any(h.id == "C1" for h in r.hits)


# =========================================================================
# Determinismo
# =========================================================================


def test_evaluate_is_deterministic() -> None:
    f = CaseFeatures(
        timing_pattern=TimingPattern.acute_continuous,
        age_years=70,
        vascular_risk_factors={VascularRiskFactor.hypertension},
        focal_signs={FocalSign.diplopia},
        hearing_loss=HearingLoss.sudden_unilateral,
    )
    r1 = evaluate(f)
    r2 = evaluate(f)
    assert r1.model_dump() == r2.model_dump()


def test_evaluate_does_not_mutate_input() -> None:
    f = _bppv_benign()
    snapshot = f.model_dump()
    _ = evaluate(f)
    assert f.model_dump() == snapshot


# =========================================================================
# INV-5 — `redflag_engine` solo importa `contracts` (y submódulos propios)
# =========================================================================


_FORBIDDEN_IMPORTS = {
    "clinibrium.differential_engine",
    "clinibrium.reasoner",
    "clinibrium.ml_client",
    "clinibrium.orchestrator",
}


def _iter_imports(py_file: Path) -> list[tuple[int, str]]:
    """Devuelve (line_no, module) de CADA import de `clinibrium.*` encontrado."""
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        mod: str | None = None
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                out.append((node.lineno, mod))
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                out.append((node.lineno, node.module))
    return out


def test_redflag_engine_does_not_import_forbidden_modules() -> None:
    """INV-5: ninguna `.py` de `redflag_engine` puede importar
    `differential_engine`, `reasoner`, `ml_client` u `orchestrator`."""
    pkg_root = Path(__file__).resolve().parents[1] / "clinibrium" / "redflag_engine"
    offenders: list[str] = []
    for py in sorted(pkg_root.glob("*.py")):
        for lineno, mod in _iter_imports(py):
            for forbidden in _FORBIDDEN_IMPORTS:
                if mod == forbidden or mod.startswith(forbidden + "."):
                    offenders.append(f"{py.name}:{lineno} → {mod}")
    assert not offenders, (
        "INV-5 violada — imports prohibidos desde redflag_engine:\n  "
        + "\n  ".join(offenders)
    )


def test_redflag_engine_only_imports_from_contracts_and_self() -> None:
    """Refuerzo: el único módulo externo permitido es `clinibrium.contracts`."""
    pkg_root = Path(__file__).resolve().parents[1] / "clinibrium" / "redflag_engine"
    allowed_roots = {"clinibrium.contracts", "clinibrium.redflag_engine"}
    offenders: list[str] = []
    for py in sorted(pkg_root.glob("*.py")):
        for lineno, mod in _iter_imports(py):
            if not mod.startswith("clinibrium."):
                continue
            if not any(mod == a or mod.startswith(a + ".") for a in allowed_roots):
                offenders.append(f"{py.name}:{lineno} → {mod}")
    assert not offenders, (
        "Import cross-module no permitido desde redflag_engine:\n  "
        + "\n  ".join(offenders)
    )


# =========================================================================
# Sanity checks sobre la tabla RULES
# =========================================================================


def test_rules_table_covers_all_documented_ids() -> None:
    """La tabla contiene exactamente los IDs documentados en la spec T4."""
    expected = {"A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8",
                "B1", "B2", "B3", "B4", "B5",
                "C1", "C2", "C3",
                "E4"}
    actual = {r.id for r in RULES}
    assert actual == expected, (
        "diff:\n"
        + "\n".join(
            difflib.unified_diff(
                sorted(expected), sorted(actual), lineterm="", n=1
            )
        )
    )


def test_all_rules_are_red_flag_rule_instances() -> None:
    for r in RULES:
        assert isinstance(r, RedFlagRule)
        assert r.id
        assert r.label
        assert r.severity in ("high", "medium")
        assert len(r.forced_actions) >= 1
        assert all(isinstance(a, ForcedAction) for a in r.forced_actions)
        assert callable(r.predicate)


def test_rule_ids_are_unique() -> None:
    ids = [r.id for r in RULES]
    assert len(ids) == len(set(ids)), f"ids duplicados: {ids}"
