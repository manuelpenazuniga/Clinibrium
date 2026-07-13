"""Shared enums for the VertigoDx domain (structured vocabulary).

Leaf: this module imports NOTHING from `clinibrium.*` except within the
`clinibrium.contracts` package itself.
"""
from __future__ import annotations

from enum import Enum


class SymptomDuration(str, Enum):
    seconds = "seconds"
    under_1min = "under_1min"
    minutes = "minutes"
    hours = "hours"
    over_24h_continuous = "over_24h_continuous"
    days = "days"
    recurrent_episodic = "recurrent_episodic"


class Onset(str, Enum):
    sudden = "sudden"
    gradual = "gradual"
    unknown = "unknown"


class Trigger(str, Enum):
    positional_head = "positional_head"
    spontaneous = "spontaneous"
    orthostatic = "orthostatic"
    valsalva = "valsalva"
    sound_pressure = "sound_pressure"
    none = "none"


class TimingPattern(str, Enum):
    """TiTrATE-style temporal pattern."""

    acute_continuous = "acute_continuous"
    episodic_triggered = "episodic_triggered"
    episodic_spontaneous = "episodic_spontaneous"
    chronic = "chronic"


class NystagmusDirection(str, Enum):
    none = "none"
    horizontal = "horizontal"
    vertical_pure = "vertical_pure"
    torsional_pure = "torsional_pure"
    mixed = "mixed"
    direction_changing = "direction_changing"


class HeadImpulse(str, Enum):
    """Result of the HINTS head-impulse test.

    CAUTION: in acute vestibular syndrome (AVS), `normal` is SUSPICIOUS for a
    central cause (the HINTS test is only interpreted with the full triad).
    """

    normal = "normal"
    abnormal_corrective_saccade = "abnormal_corrective_saccade"
    not_done = "not_done"


class HearingLoss(str, Enum):
    none = "none"
    sudden_unilateral = "sudden_unilateral"
    fluctuating = "fluctuating"
    chronic = "chronic"


class FocalSign(str, Enum):
    dysarthria = "dysarthria"
    dysphagia = "dysphagia"
    diplopia = "diplopia"
    limb_weakness = "limb_weakness"
    facial_droop = "facial_droop"
    numbness = "numbness"
    hiccups = "hiccups"
    horner = "horner"


class VascularRiskFactor(str, Enum):
    hypertension = "hypertension"
    diabetes = "diabetes"
    atrial_fibrillation = "atrial_fibrillation"
    smoking = "smoking"
    prior_stroke_tia = "prior_stroke_tia"


class DixHallpikeResult(str, Enum):
    right_positive = "right_positive"
    left_positive = "left_positive"
    bilateral_positive = "bilateral_positive"
    negative = "negative"
    not_done = "not_done"


class Diagnosis(str, Enum):
    bppv_posterior = "bppv_posterior"
    bppv_horizontal = "bppv_horizontal"
    meniere = "meniere"
    vestibular_migraine = "vestibular_migraine"
    vestibular_neuritis = "vestibular_neuritis"
    labyrinthitis = "labyrinthitis"
    central_suspected = "central_suspected"
    cardiogenic_suspected = "cardiogenic_suspected"
    undetermined = "undetermined"


class Urgency(str, Enum):
    """Urgency level.

    `inmediata` is ONLY set by `RedFlagEngine` or the rails (INV-1); the LLM
    must NEVER be able to assign it on its own.
    """

    inmediata = "inmediata"
    prioritaria = "prioritaria"
    ambulatoria = "ambulatoria"


class ForcedAction(str, Enum):
    """Actions forced by red flags / rails (non-negotiable)."""

    DERIVAR_URGENTE = "DERIVAR_URGENTE"
    NO_BENIGNO = "NO_BENIGNO"
    BLOQUEAR_EPLEY = "BLOQUEAR_EPLEY"
    PRECAUCION_EXAMEN = "PRECAUCION_EXAMEN"
    RED_SEGURIDAD = "RED_SEGURIDAD"
    ESCALAR = "ESCALAR"


class ActorType(str, Enum):
    system = "system"
    clinician = "clinician"
