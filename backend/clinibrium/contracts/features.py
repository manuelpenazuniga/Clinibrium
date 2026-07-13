"""`CaseFeatures`: structured ALLOWLIST of what crosses the network (INV-2).

Leaf of the `clinibrium.*` graph: this model contains NO PII or free text and
imports NOTHING from other `clinibrium` modules outside the
`clinibrium.contracts` package itself.

`NETWORK_SAFE_FIELDS` is the source of truth for the allowlist consumed by
the reasoner validator (INV-2): it rejects any payload that includes fields
outside this set (PII, free text, video, etc.).
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from clinibrium.contracts.enums import (
    DixHallpikeResult,
    FocalSign,
    HeadImpulse,
    HearingLoss,
    NystagmusDirection,
    Onset,
    SymptomDuration,
    TimingPattern,
    Trigger,
    VascularRiskFactor,
)


class CaseFeatures(BaseModel):
    """De-identified structured features that cross the network.

    All fields are optional unless noted; the defaults allow instantiating
    `CaseFeatures()` and filling it in progressively.

    FORBIDDEN: name, RUT/DNI, date of birth, address, free-form notes,
    free text, video frames, patient identifiers. The reasoner validator
    must reject any payload carrying fields outside `NETWORK_SAFE_FIELDS`.
    """

    model_config = ConfigDict(extra="forbid")

    # --- Temporal / trigger ---
    duration: SymptomDuration | None = None
    onset: Onset | None = None
    trigger: Trigger | None = None
    timing_pattern: TimingPattern | None = None

    # --- Nystagmus (bedside + on-device) ---
    nystagmus_direction: NystagmusDirection = NystagmusDirection.none
    nystagmus_direction_changing_gaze: bool = False
    nystagmus_latency_s: float | None = None
    nystagmus_duration_s: float | None = None
    nystagmus_fatigable: bool | None = None
    nystagmus_suppressed_by_fixation: bool | None = None

    # --- HINTS ---
    head_impulse: HeadImpulse = HeadImpulse.not_done
    skew_deviation: bool = False

    # --- Hearing ---
    hearing_loss: HearingLoss = HearingLoss.none
    tinnitus: bool = False
    aural_fullness: bool = False

    # --- Central / neuro ---
    focal_signs: set[FocalSign] = set()
    truncal_ataxia_severe: bool = False
    headache_neck_pain_sudden_severe: bool = False
    migrainous_features: bool = False

    # --- Vascular risk ---
    age_years: int | None = None
    vascular_risk_factors: set[VascularRiskFactor] = set()

    # --- Other emergencies (Block B) ---
    fever: bool = False
    neck_stiffness: bool = False
    altered_consciousness: bool = False
    presyncope_syncope: bool = False
    palpitations: bool = False
    chest_pain: bool = False
    otitis_mastoiditis: bool = False
    recent_head_neck_trauma: bool = False

    # --- Physical-exam contraindications (Block C) ---
    cervical_pathology: bool = False
    known_carotid_vertebrobasilar_disease: bool = False
    cardiovascular_instability: bool = False

    # --- Positional / torsion (Dix-Hallpike) ---
    dix_hallpike: DixHallpikeResult = DixHallpikeResult.not_done
    torsion_confirmed_by_clinician: bool | None = None

    # --- Episodic (Ménière / VM) ---
    episode_count: int | None = None
    episode_duration: SymptomDuration | None = None

    # --- Meta ---
    worsening_during_flow: bool = False  # E4


NETWORK_SAFE_FIELDS: frozenset[str] = frozenset(CaseFeatures.model_fields.keys())
