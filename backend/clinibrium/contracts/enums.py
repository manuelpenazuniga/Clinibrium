"""Enums compartidos del dominio VertigoDx (vocabulario estructurado).

Hoja: este módulo NO importa nada de `clinibrium.*` salvo dentro del propio
paquete `clinibrium.contracts`.
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
    """TiTrATE-style patrón temporal."""

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
    """Resultado del HINTS head-impulse test.

    OJO: en síndrome vestibular agudo (AVS), `normal` es SOSPECHOSO de causa
    central (test de HINTS solo se interpreta con el trío completo).
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
    """Nivel de urgencia.

    `inmediata` SOLO la fija `RedFlagEngine` o los rails (INV-1); el LLM
    NUNCA debe poder asignarla por sí solo.
    """

    inmediata = "inmediata"
    prioritaria = "prioritaria"
    ambulatoria = "ambulatoria"


class ForcedAction(str, Enum):
    """Acciones forzadas por red flags / rails (no negociables)."""

    DERIVAR_URGENTE = "DERIVAR_URGENTE"
    NO_BENIGNO = "NO_BENIGNO"
    BLOQUEAR_EPLEY = "BLOQUEAR_EPLEY"
    PRECAUCION_EXAMEN = "PRECAUCION_EXAMEN"
    RED_SEGURIDAD = "RED_SEGURIDAD"
    ESCALAR = "ESCALAR"


class ActorType(str, Enum):
    system = "system"
    clinician = "clinician"
