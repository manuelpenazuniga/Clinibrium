"""ICVD criteria table for the DifferentialEngine.

Rules live as DATA (`DiagnosisCriterion`), not as scattered logic:
editing weights or predicates is all it takes to recalibrate the
prioritized diagnosis pool. This table is the engine's **provisional
source of truth**; weights and thresholds are marked with
`# TODO(clinical)` and will be calibrated with clinical validation
data (see roadmap v7.3 §4.4 and §11).

Leaf of the `clinibrium.*` graph (INV-5): this module ONLY imports from
`clinibrium.contracts`. It does NOT import `redflag_engine`, `reasoner`,
`ml_client` or `orchestrator`. No I/O, LLM or randomness either.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from clinibrium.contracts.enums import (
    Diagnosis,
    DixHallpikeResult,
    HeadImpulse,
    HearingLoss,
    NystagmusDirection,
    SymptomDuration,
    TimingPattern,
    Trigger,
)
from clinibrium.contracts.features import CaseFeatures


@dataclass(frozen=True)
class DiagnosisCriterion:
    """Individual criterion: if `predicate(features)` is True, it adds
    `weight` to the raw score of `diagnosis`.

    IDs are traceable to the ICVD rules (format `DX-<KEY>-<n>`).
    """

    id: str
    diagnosis: Diagnosis
    weight: float
    predicate: Callable[[CaseFeatures], bool]


# ---------------------------------------------------------------------------
# Pure predicates over CaseFeatures (EXACT names from contracts.features).
# Each one is a boolean function with no side effects, no I/O, no LLM.
# ---------------------------------------------------------------------------


def _trigger_is_positional(f: CaseFeatures) -> bool:
    return f.trigger == Trigger.positional_head


def _duration_short(f: CaseFeatures) -> bool:
    return f.duration in {SymptomDuration.seconds, SymptomDuration.under_1min}


def _dix_hallpike_positive(f: CaseFeatures) -> bool:
    return f.dix_hallpike in {
        DixHallpikeResult.right_positive,
        DixHallpikeResult.left_positive,
    }


def _nystagmus_fatigable(f: CaseFeatures) -> bool:
    return f.nystagmus_fatigable is True


def _nystagmus_latency_short(f: CaseFeatures) -> bool:
    # TODO(clinical): the 20 s threshold is provisional; typical BPPV nystagmus is 1-5 s.
    return f.nystagmus_latency_s is not None and f.nystagmus_latency_s <= 20


def _torsion_confirmed(f: CaseFeatures) -> bool:
    return f.torsion_confirmed_by_clinician is True


def _nystagmus_horizontal(f: CaseFeatures) -> bool:
    return f.nystagmus_direction == NystagmusDirection.horizontal


def _timing_episodic_spontaneous(f: CaseFeatures) -> bool:
    return f.timing_pattern == TimingPattern.episodic_spontaneous


def _timing_episodic_any(f: CaseFeatures) -> bool:
    return f.timing_pattern in {
        TimingPattern.episodic_spontaneous,
        TimingPattern.episodic_triggered,
    }


def _timing_acute_continuous(f: CaseFeatures) -> bool:
    return f.timing_pattern == TimingPattern.acute_continuous


def _episode_duration_medium(f: CaseFeatures) -> bool:
    return f.episode_duration in {SymptomDuration.minutes, SymptomDuration.hours}


def _hearing_loss_fluctuating(f: CaseFeatures) -> bool:
    return f.hearing_loss == HearingLoss.fluctuating


def _hearing_loss_none(f: CaseFeatures) -> bool:
    return f.hearing_loss == HearingLoss.none


def _hearing_loss_unilateral_or_fluctuating(f: CaseFeatures) -> bool:
    return f.hearing_loss in {HearingLoss.sudden_unilateral, HearingLoss.fluctuating}


def _tinnitus(f: CaseFeatures) -> bool:
    return f.tinnitus


def _aural_fullness(f: CaseFeatures) -> bool:
    return f.aural_fullness


def _migrainous_features(f: CaseFeatures) -> bool:
    return f.migrainous_features


def _trigger_spontaneous(f: CaseFeatures) -> bool:
    return f.trigger == Trigger.spontaneous


def _head_impulse_abnormal(f: CaseFeatures) -> bool:
    return f.head_impulse == HeadImpulse.abnormal_corrective_saccade


def _acute_continuous_and_head_impulse_normal(f: CaseFeatures) -> bool:
    return (
        f.timing_pattern == TimingPattern.acute_continuous
        and f.head_impulse == HeadImpulse.normal
    )


def _nystagmus_central_direction(f: CaseFeatures) -> bool:
    return f.nystagmus_direction in {
        NystagmusDirection.vertical_pure,
        NystagmusDirection.torsional_pure,
        NystagmusDirection.direction_changing,
    }


def _nystagmus_direction_changing_gaze(f: CaseFeatures) -> bool:
    return f.nystagmus_direction_changing_gaze


def _skew_deviation(f: CaseFeatures) -> bool:
    return f.skew_deviation


def _has_focal_signs(f: CaseFeatures) -> bool:
    return len(f.focal_signs) > 0


def _truncal_ataxia_severe(f: CaseFeatures) -> bool:
    return f.truncal_ataxia_severe


def _presyncope_syncope(f: CaseFeatures) -> bool:
    return f.presyncope_syncope


def _palpitations(f: CaseFeatures) -> bool:
    return f.palpitations


def _chest_pain(f: CaseFeatures) -> bool:
    return f.chest_pain


# ---------------------------------------------------------------------------
# CRITERIA — provisional source of truth for scoring. Editing weights/predicates
# is ALL that is needed to recalibrate. The `# TODO(clinical)` markers and the
# `# NOTE` comments document the provisional regime and the points where the
# rails / RedFlagEngine (separate) impose the final decision.
# ---------------------------------------------------------------------------

CRITERIA: list[DiagnosisCriterion] = [
    # --- bppv_posterior ---
    # TODO(clinical): calibrate weights and thresholds with validation data.
    DiagnosisCriterion(
        id="DX-BPPV-P-1",
        diagnosis=Diagnosis.bppv_posterior,
        weight=0.30,
        predicate=_trigger_is_positional,
    ),
    DiagnosisCriterion(
        id="DX-BPPV-P-2",
        diagnosis=Diagnosis.bppv_posterior,
        weight=0.25,
        predicate=_duration_short,
    ),
    DiagnosisCriterion(
        id="DX-BPPV-P-3",
        diagnosis=Diagnosis.bppv_posterior,
        weight=0.30,
        predicate=_dix_hallpike_positive,
    ),
    DiagnosisCriterion(
        id="DX-BPPV-P-4",
        diagnosis=Diagnosis.bppv_posterior,
        weight=0.15,
        predicate=_nystagmus_fatigable,
    ),
    DiagnosisCriterion(
        id="DX-BPPV-P-5",
        diagnosis=Diagnosis.bppv_posterior,
        weight=0.10,
        predicate=_nystagmus_latency_short,
    ),
    DiagnosisCriterion(
        id="DX-BPPV-P-6",
        diagnosis=Diagnosis.bppv_posterior,
        weight=0.20,
        predicate=_torsion_confirmed,
    ),
    # --- bppv_horizontal ---
    # TODO(clinical): requires supine roll test, not modeled yet;
    # low-confidence candidate.
    DiagnosisCriterion(
        id="DX-BPPV-H-1",
        diagnosis=Diagnosis.bppv_horizontal,
        weight=0.30,
        predicate=_trigger_is_positional,
    ),
    DiagnosisCriterion(
        id="DX-BPPV-H-2",
        diagnosis=Diagnosis.bppv_horizontal,
        weight=0.20,
        predicate=_nystagmus_horizontal,
    ),
    DiagnosisCriterion(
        id="DX-BPPV-H-3",
        diagnosis=Diagnosis.bppv_horizontal,
        weight=0.15,
        predicate=_duration_short,
    ),
    # --- meniere ---
    DiagnosisCriterion(
        id="DX-MENIERE-1",
        diagnosis=Diagnosis.meniere,
        weight=0.30,
        predicate=_timing_episodic_spontaneous,
    ),
    DiagnosisCriterion(
        id="DX-MENIERE-2",
        diagnosis=Diagnosis.meniere,
        weight=0.25,
        predicate=_episode_duration_medium,
    ),
    DiagnosisCriterion(
        id="DX-MENIERE-3",
        diagnosis=Diagnosis.meniere,
        weight=0.30,
        predicate=_hearing_loss_fluctuating,
    ),
    DiagnosisCriterion(
        id="DX-MENIERE-4",
        diagnosis=Diagnosis.meniere,
        weight=0.15,
        predicate=_tinnitus,
    ),
    DiagnosisCriterion(
        id="DX-MENIERE-5",
        diagnosis=Diagnosis.meniere,
        weight=0.15,
        predicate=_aural_fullness,
    ),
    # --- vestibular_migraine ---
    DiagnosisCriterion(
        id="DX-VM-1",
        diagnosis=Diagnosis.vestibular_migraine,
        weight=0.40,
        predicate=_migrainous_features,
    ),
    DiagnosisCriterion(
        id="DX-VM-2",
        diagnosis=Diagnosis.vestibular_migraine,
        weight=0.20,
        predicate=_timing_episodic_any,
    ),
    DiagnosisCriterion(
        id="DX-VM-3",
        diagnosis=Diagnosis.vestibular_migraine,
        weight=0.15,
        predicate=_episode_duration_medium,
    ),
    # --- vestibular_neuritis (peripheral sign: abnormal head-impulse) ---
    DiagnosisCriterion(
        id="DX-VN-1",
        diagnosis=Diagnosis.vestibular_neuritis,
        weight=0.30,
        predicate=_timing_acute_continuous,
    ),
    DiagnosisCriterion(
        id="DX-VN-2",
        diagnosis=Diagnosis.vestibular_neuritis,
        weight=0.20,
        predicate=_trigger_spontaneous,
    ),
    DiagnosisCriterion(
        id="DX-VN-3",
        diagnosis=Diagnosis.vestibular_neuritis,
        weight=0.20,
        predicate=_nystagmus_horizontal,
    ),
    DiagnosisCriterion(
        id="DX-VN-4",
        diagnosis=Diagnosis.vestibular_neuritis,
        weight=0.25,
        predicate=_head_impulse_abnormal,
    ),
    DiagnosisCriterion(
        id="DX-VN-5",
        diagnosis=Diagnosis.vestibular_neuritis,
        weight=0.10,
        predicate=_hearing_loss_none,
    ),
    # --- labyrinthitis ---
    # NOTE: sudden hearing loss also triggers a red flag (A8/B1); the
    # differential lists labyrinthitis but the rails force urgent — AICA
    # cannot be excluded. v7.3 §4.2.
    DiagnosisCriterion(
        id="DX-LAB-1",
        diagnosis=Diagnosis.labyrinthitis,
        weight=0.25,
        predicate=_timing_acute_continuous,
    ),
    DiagnosisCriterion(
        id="DX-LAB-2",
        diagnosis=Diagnosis.labyrinthitis,
        weight=0.15,
        predicate=_trigger_spontaneous,
    ),
    DiagnosisCriterion(
        id="DX-LAB-3",
        diagnosis=Diagnosis.labyrinthitis,
        weight=0.30,
        predicate=_hearing_loss_unilateral_or_fluctuating,
    ),
    DiagnosisCriterion(
        id="DX-LAB-4",
        diagnosis=Diagnosis.labyrinthitis,
        weight=0.10,
        predicate=_tinnitus,
    ),
    # --- central_suspected (HINTS: normal head-impulse in AVS is SUSPICIOUS) ---
    DiagnosisCriterion(
        id="DX-CENT-1",
        diagnosis=Diagnosis.central_suspected,
        weight=0.30,
        predicate=_acute_continuous_and_head_impulse_normal,
    ),
    DiagnosisCriterion(
        id="DX-CENT-2",
        diagnosis=Diagnosis.central_suspected,
        weight=0.30,
        predicate=_nystagmus_central_direction,
    ),
    DiagnosisCriterion(
        id="DX-CENT-3",
        diagnosis=Diagnosis.central_suspected,
        weight=0.20,
        predicate=_nystagmus_direction_changing_gaze,
    ),
    DiagnosisCriterion(
        id="DX-CENT-4",
        diagnosis=Diagnosis.central_suspected,
        weight=0.20,
        predicate=_skew_deviation,
    ),
    DiagnosisCriterion(
        id="DX-CENT-5",
        diagnosis=Diagnosis.central_suspected,
        weight=0.30,
        predicate=_has_focal_signs,
    ),
    DiagnosisCriterion(
        id="DX-CENT-6",
        diagnosis=Diagnosis.central_suspected,
        weight=0.20,
        predicate=_truncal_ataxia_severe,
    ),
    # --- cardiogenic_suspected ---
    DiagnosisCriterion(
        id="DX-CARDIO-1",
        diagnosis=Diagnosis.cardiogenic_suspected,
        weight=0.40,
        predicate=_presyncope_syncope,
    ),
    DiagnosisCriterion(
        id="DX-CARDIO-2",
        diagnosis=Diagnosis.cardiogenic_suspected,
        weight=0.20,
        predicate=_palpitations,
    ),
    DiagnosisCriterion(
        id="DX-CARDIO-3",
        diagnosis=Diagnosis.cardiogenic_suspected,
        weight=0.20,
        predicate=_chest_pain,
    ),
]
