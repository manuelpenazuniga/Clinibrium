"""Tabla de red flags — fuente de verdad provisional de las reglas clínicas.

INV-5: este módulo SOLO importa `contracts` (hoja). NO importa
`differential_engine`, `reasoner`, `ml_client` ni `orchestrator`.

Las reglas se modelan como **datos** (lista `RULES`) para que la validación
clínica por el superespecialista sea una edición de tabla, no una refactorización
de lógica dispersa. Cada umbral pendiente está marcado con `# TODO(clinical)`.

Reglas NO modeladas aquí como predicado de features (se manejan en rails / T8):
  - C4: flags centrales (p.ej. HINTS positivo) → derivar ANTES de la maniobra
        posicional. Es una restricción de *flujo*, no de feature.
  - E1: nunca descartar stroke por RM normal → invariante rails.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from clinibrium.contracts.enums import (
    ForcedAction,
    HeadImpulse,
    HearingLoss,
    NystagmusDirection,
    TimingPattern,
)
from clinibrium.contracts.features import CaseFeatures

# Umbral provisional de edad para A7 (vértigo agudo + edad + riesgo vascular).
# TODO(clinical): pregunta abierta A7 — confirmar con superespecialista.
AGE_CENTRAL_THRESHOLD = 60


def _avs(f: CaseFeatures) -> bool:
    """AVS = acute vestibular syndrome = timing continuo agudo."""
    return f.timing_pattern == TimingPattern.acute_continuous


Severity = Literal["high", "medium"]


@dataclass(frozen=True)
class RedFlagRule:
    """Una regla de red flag como dato puro.

    `predicate` es una **función pura** sobre `CaseFeatures` — sin I/O, sin
    acceso a red, sin random. Validar una regla = editar esta tabla.
    """

    id: str
    label: str
    severity: Severity
    forced_actions: tuple[ForcedAction, ...]
    predicate: Callable[[CaseFeatures], bool]


# -----------------------------------------------------------------------------
# Predicados de las reglas. Cada función es pura sobre `CaseFeatures`.
# -----------------------------------------------------------------------------

# TODO(clinical): gating exacto de HINTS (head-impulse normal solo es central
# en AVS con nistagmo espontáneo). Pendiente de criterio fino del especialista.
def _a1_avs_central_hints(f: CaseFeatures) -> bool:
    return _avs(f) and (
        f.head_impulse == HeadImpulse.normal
        or f.nystagmus_direction_changing_gaze
        or f.nystagmus_direction == NystagmusDirection.direction_changing
        or f.skew_deviation
    )


def _a2_pure_vertical_or_torsional_nystagmus(f: CaseFeatures) -> bool:
    return f.nystagmus_direction in {
        NystagmusDirection.vertical_pure,
        NystagmusDirection.torsional_pure,
    }


def _a3_direction_changing_nystagmus(f: CaseFeatures) -> bool:
    return (
        f.nystagmus_direction_changing_gaze
        or f.nystagmus_direction == NystagmusDirection.direction_changing
    )


def _a4_severe_truncal_ataxia(f: CaseFeatures) -> bool:
    return f.truncal_ataxia_severe


def _a5_any_focal_sign(f: CaseFeatures) -> bool:
    return len(f.focal_signs) > 0


def _a6_sudden_severe_headache_or_neck_pain(f: CaseFeatures) -> bool:
    return f.headache_neck_pain_sudden_severe


def _a7_avs_age_vascular_risk(f: CaseFeatures) -> bool:
    return (
        _avs(f)
        and (f.age_years or 0) >= AGE_CENTRAL_THRESHOLD
        and len(f.vascular_risk_factors) >= 1
    )


def _a8_sudden_unilateral_hearing_loss_with_avs(f: CaseFeatures) -> bool:
    return f.hearing_loss == HearingLoss.sudden_unilateral and _avs(f)


# TODO(clinical): regla defensiva agregada tras auditoría; confirmar con
# especialista. Cubre meningitis afébril, trombosis basilar, hemorragia
# cerebelosa y herniación — un falso negativo aquí no es tolerable.
def _a9_altered_consciousness(f: CaseFeatures) -> bool:
    return f.altered_consciousness


# TODO(clinical): confirmar con especialista. `nystagmus_suppressed_by_fixation`
# es bool|None; usamos `is False` para NO disparar con valor desconocido (None).
def _a10_nystagmus_not_suppressed_in_avs(f: CaseFeatures) -> bool:
    return (
        _avs(f)
        and f.nystagmus_suppressed_by_fixation is False
    )


def _b1_sudden_unilateral_hearing_loss(f: CaseFeatures) -> bool:
    return f.hearing_loss == HearingLoss.sudden_unilateral


def _b2_meningismus_or_altered_consciousness(f: CaseFeatures) -> bool:
    return f.fever and (f.neck_stiffness or f.altered_consciousness)


def _b3_cardiogenic_pattern(f: CaseFeatures) -> bool:
    return f.presyncope_syncope or f.palpitations or f.chest_pain


def _b4_otitis_or_mastoiditis(f: CaseFeatures) -> bool:
    return f.otitis_mastoiditis


def _b5_recent_head_neck_trauma(f: CaseFeatures) -> bool:
    return f.recent_head_neck_trauma


def _c1_cervical_pathology(f: CaseFeatures) -> bool:
    return f.cervical_pathology


def _c2_known_carotid_vertebrobasilar_disease(f: CaseFeatures) -> bool:
    return f.known_carotid_vertebrobasilar_disease


def _c3_cardiovascular_instability(f: CaseFeatures) -> bool:
    return f.cardiovascular_instability


def _e4_worsening_during_flow(f: CaseFeatures) -> bool:
    return f.worsening_during_flow


# -----------------------------------------------------------------------------
# Tabla provisional de reglas. Fuente de verdad editable.
# Orden estable: el orden de `RULES` define el orden de `hits` en el resultado.
# -----------------------------------------------------------------------------

RULES: list[RedFlagRule] = [
    # --- Bloque A: stroke / causa central en AVS ---
    RedFlagRule(
        id="A1",
        label="AVS con HINTS sospechoso de central",
        severity="high",
        forced_actions=(ForcedAction.NO_BENIGNO, ForcedAction.DERIVAR_URGENTE),
        predicate=_a1_avs_central_hints,
    ),
    RedFlagRule(
        id="A2",
        label="Nistagmo vertical o torsional puro",
        severity="high",
        forced_actions=(ForcedAction.DERIVAR_URGENTE,),
        predicate=_a2_pure_vertical_or_torsional_nystagmus,
    ),
    RedFlagRule(
        id="A3",
        label="Nistagmo que cambia de dirección",
        severity="high",
        forced_actions=(ForcedAction.DERIVAR_URGENTE,),
        predicate=_a3_direction_changing_nystagmus,
    ),
    RedFlagRule(
        id="A4",
        label="Ataxia truncal severa",
        severity="high",
        forced_actions=(ForcedAction.DERIVAR_URGENTE,),
        predicate=_a4_severe_truncal_ataxia,
    ),
    RedFlagRule(
        id="A5",
        label="Signos neurológicos focales",
        severity="high",
        forced_actions=(ForcedAction.DERIVAR_URGENTE,),
        predicate=_a5_any_focal_sign,
    ),
    RedFlagRule(
        id="A6",
        label="Cefalea o cervicalgia súbita severa",
        severity="high",
        forced_actions=(ForcedAction.DERIVAR_URGENTE,),
        predicate=_a6_sudden_severe_headache_or_neck_pain,
    ),
    RedFlagRule(
        id="A7",
        label="AVS + edad + riesgo vascular",
        severity="medium",
        forced_actions=(ForcedAction.NO_BENIGNO, ForcedAction.ESCALAR),
        predicate=_a7_avs_age_vascular_risk,
    ),
    RedFlagRule(
        id="A8",
        label="Hipoacusia súbita unilateral + vértigo agudo (AICA)",
        severity="high",
        forced_actions=(ForcedAction.NO_BENIGNO, ForcedAction.DERIVAR_URGENTE),
        predicate=_a8_sudden_unilateral_hearing_loss_with_avs,
    ),
    RedFlagRule(
        id="A9",
        label="Compromiso de conciencia con vértigo agudo",
        severity="high",
        forced_actions=(ForcedAction.DERIVAR_URGENTE,),
        predicate=_a9_altered_consciousness,
    ),
    RedFlagRule(
        id="A10",
        label="Nistagmo no suprimido por fijación en AVS",
        severity="high",
        forced_actions=(ForcedAction.NO_BENIGNO, ForcedAction.DERIVAR_URGENTE),
        predicate=_a10_nystagmus_not_suppressed_in_avs,
    ),
    # --- Bloque B: otras urgencias ---
    RedFlagRule(
        id="B1",
        label="Hipoacusia neurosensorial súbita",
        severity="high",
        forced_actions=(ForcedAction.DERIVAR_URGENTE,),
        predicate=_b1_sudden_unilateral_hearing_loss,
    ),
    RedFlagRule(
        id="B2",
        label="Fiebre con rigidez de nuca o compromiso de conciencia",
        severity="high",
        forced_actions=(ForcedAction.DERIVAR_URGENTE,),
        predicate=_b2_meningismus_or_altered_consciousness,
    ),
    RedFlagRule(
        id="B3",
        label="Patrón cardiogénico (síncope, palpitaciones, dolor torácico)",
        severity="medium",
        forced_actions=(ForcedAction.ESCALAR,),
        predicate=_b3_cardiogenic_pattern,
    ),
    RedFlagRule(
        id="B4",
        label="Otitis o mastoiditis con vértigo",
        severity="high",
        forced_actions=(ForcedAction.DERIVAR_URGENTE,),
        predicate=_b4_otitis_or_mastoiditis,
    ),
    RedFlagRule(
        id="B5",
        label="Trauma craneal o cervical reciente",
        severity="medium",
        forced_actions=(
            ForcedAction.PRECAUCION_EXAMEN,
            ForcedAction.ESCALAR,
        ),
        predicate=_b5_recent_head_neck_trauma,
    ),
    # --- Bloque C: contraindicaciones de examen físico ---
    RedFlagRule(
        id="C1",
        label="Patología cervical significativa",
        severity="medium",
        forced_actions=(ForcedAction.PRECAUCION_EXAMEN,),
        predicate=_c1_cervical_pathology,
    ),
    RedFlagRule(
        id="C2",
        label="Enfermedad carotídea o vertebrobasilar conocida",
        severity="medium",
        forced_actions=(ForcedAction.PRECAUCION_EXAMEN,),
        predicate=_c2_known_carotid_vertebrobasilar_disease,
    ),
    RedFlagRule(
        id="C3",
        label="Inestabilidad cardiovascular",
        severity="medium",
        forced_actions=(ForcedAction.PRECAUCION_EXAMEN,),
        predicate=_c3_cardiovascular_instability,
    ),
    # --- Bloque E: seguridad del flujo ---
    RedFlagRule(
        id="E4",
        label="Empeoramiento durante el flujo",
        severity="high",
        forced_actions=(ForcedAction.DERIVAR_URGENTE,),
        predicate=_e4_worsening_during_flow,
    ),
]


__all__ = ["RULES", "AGE_CENTRAL_THRESHOLD", "RedFlagRule"]
