"""`CaseFeatures`: ALLOWLIST estructurado de lo que cruza la red (INV-2).

Hoja del grafo `clinibrium.*`: este modelo NO contiene PII ni texto libre y
NO importa nada de otros módulos de `clinibrium` fuera del propio paquete
`clinibrium.contracts`.

`NETWORK_SAFE_FIELDS` es la fuente de verdad del allowlist que consume el
validador del reasoner (INV-2): rechaza cualquier payload que incluya campos
fuera de este set (PII, texto libre, video, etc.).
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
    """Features estructuradas desidentificadas que cruzan la red.

    Todos los campos son opcionales salvo indicación; los defaults permiten
    instanciar `CaseFeatures()` y rellenar progresivamente.

    PROHIBIDO: nombre, RUT/DNI, fecha de nacimiento, dirección, notas libres,
    texto libre, frames de video, identificadores de paciente. El validador
    del reasoner debe rechazar cualquier payload que traiga campos fuera de
    `NETWORK_SAFE_FIELDS`.
    """

    model_config = ConfigDict(extra="forbid")

    # --- Temporal / gatillo ---
    duration: SymptomDuration | None = None
    onset: Onset | None = None
    trigger: Trigger | None = None
    timing_pattern: TimingPattern | None = None

    # --- Nistagmo (bedside + on-device) ---
    nystagmus_direction: NystagmusDirection = NystagmusDirection.none
    nystagmus_direction_changing_gaze: bool = False
    nystagmus_latency_s: float | None = None
    nystagmus_duration_s: float | None = None
    nystagmus_fatigable: bool | None = None
    nystagmus_suppressed_by_fixation: bool | None = None

    # --- HINTS ---
    head_impulse: HeadImpulse = HeadImpulse.not_done
    skew_deviation: bool = False

    # --- Audición ---
    hearing_loss: HearingLoss = HearingLoss.none
    tinnitus: bool = False
    aural_fullness: bool = False

    # --- Central / neuro ---
    focal_signs: set[FocalSign] = set()
    truncal_ataxia_severe: bool = False
    headache_neck_pain_sudden_severe: bool = False
    migrainous_features: bool = False

    # --- Riesgo vascular ---
    age_years: int | None = None
    vascular_risk_factors: set[VascularRiskFactor] = set()

    # --- Otras urgencias (Bloque B) ---
    fever: bool = False
    neck_stiffness: bool = False
    altered_consciousness: bool = False
    presyncope_syncope: bool = False
    palpitations: bool = False
    chest_pain: bool = False
    otitis_mastoiditis: bool = False
    recent_head_neck_trauma: bool = False

    # --- Contraindicaciones examen (Bloque C) ---
    cervical_pathology: bool = False
    known_carotid_vertebrobasilar_disease: bool = False
    cardiovascular_instability: bool = False

    # --- Posicional / torsión (Dix-Hallpike) ---
    dix_hallpike: DixHallpikeResult = DixHallpikeResult.not_done
    torsion_confirmed_by_clinician: bool | None = None

    # --- Episódico (Ménière / MV) ---
    episode_count: int | None = None
    episode_duration: SymptomDuration | None = None

    # --- Meta ---
    worsening_during_flow: bool = False  # E4


NETWORK_SAFE_FIELDS: frozenset[str] = frozenset(CaseFeatures.model_fields.keys())
