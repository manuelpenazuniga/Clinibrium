"""Tests for the `RedFlagEngine` (T4) — INV-5 + coverage of the `RULES` table.

Covers the task's 4 acceptance criteria:
  1. One POSITIVE test for EACH rule in `RULES` (correct id, label,
     severity, forced_actions on the corresponding hit).
  2. Base NEGATIVE test: typical benign BPPV case → `red_flag_activa
     == False` and no `DERIVAR_URGENTE`.
  3. `red_flag_activa` True/False depending on whether `DERIVAR_URGENTE`
     is present.
  4. Determinism: same features → same result.
  5. INV-5: `redflag_engine` does NOT import `differential_engine` /
     `reasoner` / `ml_client` / `orchestrator`. Only `contracts` (and
     the package's own submodules).
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
# Case-building helpers (minimal features per rule)
# =========================================================================


def _bppv_benign() -> CaseFeatures:
    """Base negative case: typical, benign posterior-canal BPPV.

    Characteristics:
      - duration < 1 min
      - triggered by head position
      - episodic-triggered temporal pattern (NOT AVS)
      - upbeating-torsional (mixed) nystagmus with latency and fatigability
      - positive Dix-Hallpike, torsion confirmed by clinician
      - no focal signs, no vascular risk factors
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
        f"rule {rule_id} not present in hits={[h.id for h in result.hits]}"
    )
    return matches[0]


# =========================================================================
# POSITIVE tests — one per rule in RULES
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


def test_a1_fires_with_skew_deviation_alone() -> None:
    """Audit fix #2: A1 also fires on isolated skew_deviation in AVS."""
    f = CaseFeatures(
        timing_pattern=TimingPattern.acute_continuous,
        skew_deviation=True,
    )
    r = evaluate(f)
    h = _hit(r, "A1")
    assert h.severity == "high"
    assert ForcedAction.NO_BENIGNO in h.forced_actions
    assert ForcedAction.DERIVAR_URGENTE in h.forced_actions
    assert r.red_flag_activa is True


def test_a1_fires_with_nystagmus_direction_changing_enum() -> None:
    """Audit fix #2: A1 also fires when the direction-changing nystagmus comes
    via the enum (not the bool). A3 also fires by design; the test only
    requires A1 to be present with NO_BENIGNO + DERIVAR_URGENTE."""
    f = CaseFeatures(
        timing_pattern=TimingPattern.acute_continuous,
        nystagmus_direction=NystagmusDirection.direction_changing,
    )
    r = evaluate(f)
    h = _hit(r, "A1")
    assert h.severity == "high"
    assert set(h.forced_actions) == {ForcedAction.NO_BENIGNO, ForcedAction.DERIVAR_URGENTE}
    assert r.red_flag_activa is True
    # A3 also fires for the same reason — verify both are present
    assert any(h.id == "A3" for h in r.hits)


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
        assert any(h.id == "A5" for h in r.hits), f"A5 did not fire with focal_sign={sign}"


def test_a6_sudden_severe_headache_or_neck_pain() -> None:
    f = CaseFeatures(headache_neck_pain_sudden_severe=True)
    r = evaluate(f)
    h = _hit(r, "A6")
    assert h.severity == "high"
    assert h.forced_actions == [ForcedAction.DERIVAR_URGENTE]
    assert r.red_flag_activa is True


def test_a7_avs_age_vascular_risk() -> None:
    """A7 is medium: ESCALAR + NO_BENIGNO, does NOT activate red_flag on its own."""
    f = CaseFeatures(
        timing_pattern=TimingPattern.acute_continuous,
        age_years=AGE_CENTRAL_THRESHOLD,
        vascular_risk_factors={VascularRiskFactor.hypertension},
    )
    r = evaluate(f)
    h = _hit(r, "A7")
    assert h.severity == "medium"
    assert set(h.forced_actions) == {ForcedAction.NO_BENIGNO, ForcedAction.ESCALAR}
    # medium does not force referral → red_flag_activa must be False in isolation
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
    """A8 is the AICA combination (sudden + AVS). B1 also fires here by
    design (B1 covers the isolated sudden loss or with vertigo); we verify
    A8 is present with its specific action."""
    f = CaseFeatures(
        timing_pattern=TimingPattern.acute_continuous,
        hearing_loss=HearingLoss.sudden_unilateral,
    )
    r = evaluate(f)
    h = _hit(r, "A8")
    assert h.severity == "high"
    assert set(h.forced_actions) == {ForcedAction.NO_BENIGNO, ForcedAction.DERIVAR_URGENTE}
    assert r.red_flag_activa is True


def test_a8_includes_no_benigno_in_forced_actions() -> None:
    """Audit fix #4: A8 (AICA) is NOT benign — it includes NO_BENIGNO."""
    f = CaseFeatures(
        timing_pattern=TimingPattern.acute_continuous,
        hearing_loss=HearingLoss.sudden_unilateral,
    )
    r = evaluate(f)
    h = _hit(r, "A8")
    assert ForcedAction.NO_BENIGNO in h.forced_actions
    assert ForcedAction.NO_BENIGNO in r.forced_actions


def test_a9_altered_consciousness_afebrile() -> None:
    """Audit fix #1 [CRITICAL]: altered_consciousness without fever fires A9
    (e.g. afebrile meningitis, basilar thrombosis, herniation)."""
    f = CaseFeatures(altered_consciousness=True, fever=False)
    r = evaluate(f)
    h = _hit(r, "A9")
    assert h.label == "Compromiso de conciencia con vértigo agudo"
    assert h.severity == "high"
    assert h.forced_actions == [ForcedAction.DERIVAR_URGENTE]
    assert r.red_flag_activa is True
    # B2 must NOT fire (no fever)
    assert all(h.id != "B2" for h in r.hits)


def test_a9_fires_alongside_b2_when_fever_and_altered_consciousness() -> None:
    """A9 and B2 are independent rules: both fire when there is fever AND
    altered consciousness (redundant coverage on purpose)."""
    f = CaseFeatures(fever=True, altered_consciousness=True)
    r = evaluate(f)
    assert any(h.id == "A9" for h in r.hits)
    assert any(h.id == "B2" for h in r.hits)
    assert r.red_flag_activa is True


def test_a9_does_not_fire_when_consciousness_intact() -> None:
    f = CaseFeatures(altered_consciousness=False)
    r = evaluate(f)
    assert all(h.id != "A9" for h in r.hits)


def test_a10_nystagmus_not_suppressed_in_avs() -> None:
    """Audit fix #3: nystagmus NOT suppressed by fixation in AVS is a central
    sign. Fires only on explicit `False`, not on `None`."""
    f = CaseFeatures(
        timing_pattern=TimingPattern.acute_continuous,
        nystagmus_suppressed_by_fixation=False,
    )
    r = evaluate(f)
    h = _hit(r, "A10")
    assert h.severity == "high"
    assert set(h.forced_actions) == {ForcedAction.NO_BENIGNO, ForcedAction.DERIVAR_URGENTE}
    assert r.red_flag_activa is True


def test_a10_does_not_fire_when_suppressed_by_fixation_is_none() -> None:
    """A10 does NOT fire on an unknown value (None): only explicit False."""
    f = CaseFeatures(
        timing_pattern=TimingPattern.acute_continuous,
        nystagmus_suppressed_by_fixation=None,
    )
    r = evaluate(f)
    assert all(h.id != "A10" for h in r.hits)


def test_a10_does_not_fire_when_suppressed_is_true() -> None:
    """Nystagmus suppressed by fixation = peripheral sign. A10 does NOT fire."""
    f = CaseFeatures(
        timing_pattern=TimingPattern.acute_continuous,
        nystagmus_suppressed_by_fixation=True,
    )
    r = evaluate(f)
    assert all(h.id != "A10" for h in r.hits)


def test_a10_does_not_fire_outside_avs() -> None:
    """A10 requires AVS — outside AVS the field does not imply centrality."""
    f = CaseFeatures(
        timing_pattern=TimingPattern.episodic_triggered,
        nystagmus_suppressed_by_fixation=False,
    )
    r = evaluate(f)
    assert all(h.id != "A10" for h in r.hits)


def test_b1_sudden_unilateral_hearing_loss_isolated() -> None:
    """T-CLIN r1: ISOLATED sudden hearing loss (no AVS) = PRIORITY ENT (48h),
    NOT an emergency. B1 contributes ESCALAR (priority), not DERIVAR_URGENTE,
    and does NOT activate red_flag_activa (which is only for immediate)."""
    f = CaseFeatures(hearing_loss=HearingLoss.sudden_unilateral)
    r = evaluate(f)
    h = _hit(r, "B1")
    assert h.severity == "medium"
    assert h.forced_actions == [ForcedAction.ESCALAR]
    assert r.red_flag_activa is False  # isolated is NOT an emergency
    # A8 must NOT fire without AVS
    assert all(hit.id != "A8" for hit in r.hits)


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
    """B3 is medium: ESCALAR only; must not activate red_flag on its own."""
    f = CaseFeatures(presyncope_syncope=True)
    r = evaluate(f)
    h = _hit(r, "B3")
    assert h.severity == "medium"
    assert h.forced_actions == [ForcedAction.ESCALAR]
    assert r.red_flag_activa is False
    assert ForcedAction.ESCALAR in r.forced_actions


def test_b3_fires_with_chest_pain_alone() -> None:
    """Audit fix #5: B3 fires with isolated chest_pain."""
    f = CaseFeatures(chest_pain=True)
    r = evaluate(f)
    h = _hit(r, "B3")
    assert h.severity == "medium"
    assert h.forced_actions == [ForcedAction.ESCALAR]
    assert r.red_flag_activa is False


def test_b3_fires_with_palpitations_alone() -> None:
    """Audit fix #5: B3 fires with isolated palpitations."""
    f = CaseFeatures(palpitations=True)
    r = evaluate(f)
    h = _hit(r, "B3")
    assert h.severity == "medium"
    assert h.forced_actions == [ForcedAction.ESCALAR]
    assert r.red_flag_activa is False


def test_b4_otitis_or_mastoiditis() -> None:
    f = CaseFeatures(otitis_mastoiditis=True)
    r = evaluate(f)
    h = _hit(r, "B4")
    assert h.severity == "high"
    assert h.forced_actions == [ForcedAction.DERIVAR_URGENTE]
    assert r.red_flag_activa is True


def test_b5_recent_head_neck_trauma() -> None:
    """B5 medium: PRECAUCION_EXAMEN + ESCALAR; does not activate red_flag on its own."""
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
# Base NEGATIVE test — typical benign BPPV
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
    """A combination of medium rules (no DERIVAR_URGENTE) must not activate
    red_flag_activa even when the result's forced_actions are non-empty."""
    f = CaseFeatures(
        cervical_pathology=True,  # C1
        cardiovascular_instability=True,  # C3
        presyncope_syncope=True,  # B3
    )
    r = evaluate(f)
    assert r.hits  # there are hits
    assert r.red_flag_activa is False
    assert ForcedAction.DERIVAR_URGENTE not in r.forced_actions
    assert ForcedAction.PRECAUCION_EXAMEN in r.forced_actions
    assert ForcedAction.ESCALAR in r.forced_actions


def test_red_flag_activa_true_with_any_derivar_urgente() -> None:
    """A single rule with DERIVAR_URGENTE already activates red_flag, even
    when medium rules are also present in the same case."""
    f = CaseFeatures(
        cervical_pathology=True,  # C1 — precaution, no red flag on its own
        focal_signs={FocalSign.dysarthria},  # A5 — DERIVAR_URGENTE
    )
    r = evaluate(f)
    assert r.red_flag_activa is True
    assert ForcedAction.DERIVAR_URGENTE in r.forced_actions
    assert any(h.id == "A5" for h in r.hits)
    assert any(h.id == "C1" for h in r.hits)


# =========================================================================
# Determinism
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
# INV-5 — `redflag_engine` only imports `contracts` (and its own submodules)
# =========================================================================


_FORBIDDEN_IMPORTS = {
    "clinibrium.differential_engine",
    "clinibrium.reasoner",
    "clinibrium.ml_client",
    "clinibrium.orchestrator",
}


def _iter_imports(py_file: Path) -> list[tuple[int, str]]:
    """Returns (line_no, module) for EVERY `clinibrium.*` import found."""
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
    """INV-5: no `.py` in `redflag_engine` may import
    `differential_engine`, `reasoner`, `ml_client` or `orchestrator`."""
    pkg_root = Path(__file__).resolve().parents[1] / "clinibrium" / "redflag_engine"
    offenders: list[str] = []
    for py in sorted(pkg_root.glob("*.py")):
        for lineno, mod in _iter_imports(py):
            for forbidden in _FORBIDDEN_IMPORTS:
                if mod == forbidden or mod.startswith(forbidden + "."):
                    offenders.append(f"{py.name}:{lineno} → {mod}")
    assert not offenders, (
        "INV-5 violated — forbidden imports from redflag_engine:\n  "
        + "\n  ".join(offenders)
    )


def test_redflag_engine_only_imports_from_contracts_and_self() -> None:
    """Reinforcement: the only allowed external module is `clinibrium.contracts`."""
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
        "Disallowed cross-module import from redflag_engine:\n  "
        + "\n  ".join(offenders)
    )


# =========================================================================
# Sanity checks over the RULES table
# =========================================================================


def test_rules_table_covers_all_documented_ids() -> None:
    """The table contains exactly the IDs documented in the T4 spec
    and in the correctness audit (A9, A10)."""
    expected = {"A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10",
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
    assert len(ids) == len(set(ids)), f"duplicate ids: {ids}"
