"""Dominio VÉRTIGO (instancia #1) — config, no código.

Define el ``Domain`` de vértigo: features (subconjunto del allowlist de
``CaseFeatures``), transformadores derivados PUROS (definidos AQUÍ, no en el
core — así el core es agnóstico), la jerarquía con gate binario de peligro, y
los priors sintéticos (provisionales — el especialista los refina, T-CLIN).

NO importa ``clinibrium``: el vocabulario (strings de enums) es config de este
dominio. Las categorías coinciden con los ``.value`` de los enums de A para
que el modelo entrenado sobre sintético consuma los mismos strings al servir.
"""
from __future__ import annotations

import math

from ml_engine.core.spec import (
    DerivedFeature,
    Domain,
    FeatureSpec,
    LabelHierarchy,
    LabelProfile,
    Node,
    NumericDist,
    RawFeature,
    Row,
    SyntheticSpec,
)

SEED = 20260711

# --------------------------------------------------------------------------
# Transformadores derivados PUROS (NaN-safe). Leen la fila cruda de input.
# Robustos a la forma "serving" (focal_signs = lista) y "synth" (= conteo num).
# --------------------------------------------------------------------------


def _is_nan(v: object) -> bool:
    return isinstance(v, float) and math.isnan(v)


def _b(v: object) -> float:
    # NaN-safe: en Python bool(nan) es True → hay que guardarlo explícito.
    if v is None or _is_nan(v):
        return 0.0
    return 1.0 if v else 0.0


def _count(v: object) -> float:
    if v is None or _is_nan(v):
        return 0.0
    if isinstance(v, (list, tuple, set, frozenset)):
        return float(len(v))
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    return 0.0


def danger_sign_count(row: Row) -> float:
    """Signos neurológicos de alarma (focal + ataxia troncal + cefalea súbita)."""
    return (
        _count(row.get("focal_signs"))
        + _b(row.get("truncal_ataxia_severe"))
        + _b(row.get("headache_neck_pain_sudden_severe"))
    )


def hints_central_pattern(row: Row) -> float:
    """Patrón HINTS 'central': head-impulse NORMAL en AVS (acute_continuous)."""
    return (
        1.0
        if row.get("head_impulse") == "normal"
        and row.get("timing_pattern") == "acute_continuous"
        else 0.0
    )


def vascular_risk_count(row: Row) -> float:
    """Carga de riesgo vascular: |factores| + (edad ≥ 60)."""
    age = row.get("age_years")
    age_pt = 1.0 if isinstance(age, (int, float)) and not isinstance(age, bool) and age >= 60 else 0.0
    return _count(row.get("vascular_risk_factors")) + age_pt


def cardiogenic_cluster(row: Row) -> float:
    """Cluster cardiogénico: presíncope + palpitaciones + dolor torácico + ortostático."""
    return (
        _b(row.get("presyncope_syncope"))
        + _b(row.get("palpitations"))
        + _b(row.get("chest_pain"))
        + (1.0 if row.get("trigger") == "orthostatic" else 0.0)
    )


def central_nystagmus_pattern(row: Row) -> float:
    """Patrón de nistagmo CENTRAL: puro torsional/vertical o cambiante (A2/A3).

    Reconcilia A↔B: el motor A trata estas direcciones como red flag central;
    esta derivada las lleva al gate de peligro (feature de riesgo monótona) para
    que B coincida. (El torsional POSICIONAL del VPPB va por dix_hallpike +
    nystagmus_fatigable, no por nystagmus_direction.)
    """
    return (
        1.0
        if row.get("nystagmus_direction") in {"torsional_pure", "vertical_pure", "direction_changing"}
        else 0.0
    )


# --------------------------------------------------------------------------
# FeatureSpec — features que ve el modelo + derivadas + risk features
# --------------------------------------------------------------------------

_RAW: tuple[RawFeature, ...] = (
    # Categóricas (categorías = .value exactos de los enums de A)
    RawFeature("duration", "categorical",
               ("seconds", "under_1min", "minutes", "hours", "over_24h_continuous", "days", "recurrent_episodic")),
    RawFeature("onset", "categorical", ("sudden", "gradual", "unknown")),
    RawFeature("trigger", "categorical",
               ("positional_head", "spontaneous", "orthostatic", "valsalva", "sound_pressure", "none")),
    RawFeature("timing_pattern", "categorical",
               ("acute_continuous", "episodic_triggered", "episodic_spontaneous", "chronic")),
    RawFeature("nystagmus_direction", "categorical",
               ("none", "horizontal", "vertical_pure", "torsional_pure", "mixed", "direction_changing")),
    RawFeature("head_impulse", "categorical", ("normal", "abnormal_corrective_saccade", "not_done")),
    RawFeature("hearing_loss", "categorical", ("none", "sudden_unilateral", "fluctuating", "chronic")),
    RawFeature("dix_hallpike", "categorical",
               ("right_positive", "left_positive", "bilateral_positive", "negative", "not_done")),
    # Booleanas (encode → 0/1)
    RawFeature("skew_deviation", "boolean"),
    RawFeature("nystagmus_direction_changing_gaze", "boolean"),
    RawFeature("tinnitus", "boolean"),
    RawFeature("aural_fullness", "boolean"),
    RawFeature("truncal_ataxia_severe", "boolean"),
    RawFeature("headache_neck_pain_sudden_severe", "boolean"),
    RawFeature("migrainous_features", "boolean"),
    RawFeature("presyncope_syncope", "boolean"),
    RawFeature("palpitations", "boolean"),
    RawFeature("chest_pain", "boolean"),
    # Fatigabilidad del nistagmo: signo BENIGNO (posicional/VPPB). Reconcilia
    # A↔B: el torsional/vertical ESPONTÁNEO es central (A2); el de VPPB es
    # posicional + fatigable (dix_hallpike + este flag), NO nystagmus_direction.
    RawFeature("nystagmus_fatigable", "boolean"),
    # Numéricas
    RawFeature("nystagmus_latency_s", "numeric"),
    RawFeature("nystagmus_duration_s", "numeric"),
    RawFeature("age_years", "numeric"),
    RawFeature("episode_count", "numeric"),
)

_DERIVED: tuple[DerivedFeature, ...] = (
    DerivedFeature("danger_sign_count", danger_sign_count),
    DerivedFeature("hints_central_pattern", hints_central_pattern),
    DerivedFeature("vascular_risk_count", vascular_risk_count),
    DerivedFeature("cardiogenic_cluster", cardiogenic_cluster),
    DerivedFeature("central_nystagmus_pattern", central_nystagmus_pattern),
)

# Features de riesgo (monótonas +1 hacia peligro en el gate N0a). Todas numéricas.
_RISK: tuple[str, ...] = (
    "danger_sign_count",
    "hints_central_pattern",
    "vascular_risk_count",
    "cardiogenic_cluster",
    "central_nystagmus_pattern",
    "skew_deviation",
    "nystagmus_direction_changing_gaze",
)

# Allowlist de input del servicio (extra="forbid"): el shape completo de
# CaseFeatures (frontera de privacidad). Duplicado a propósito como config del
# dominio; el servicio rechaza cualquier clave fuera de acá.
_ALLOWLIST: frozenset[str] = frozenset({
    "duration", "onset", "trigger", "timing_pattern",
    "nystagmus_direction", "nystagmus_direction_changing_gaze", "nystagmus_latency_s",
    "nystagmus_duration_s", "nystagmus_fatigable", "nystagmus_suppressed_by_fixation",
    "head_impulse", "skew_deviation", "hearing_loss", "tinnitus", "aural_fullness",
    "focal_signs", "truncal_ataxia_severe", "headache_neck_pain_sudden_severe",
    "migrainous_features", "age_years", "vascular_risk_factors", "fever", "neck_stiffness",
    "altered_consciousness", "presyncope_syncope", "palpitations", "chest_pain",
    "otitis_mastoiditis", "recent_head_neck_trauma", "cervical_pathology",
    "known_carotid_vertebrobasilar_disease", "cardiovascular_instability",
    "dix_hallpike", "torsion_confirmed_by_clinician", "episode_count",
    "episode_duration", "worsening_during_flow",
})

FEATURES = FeatureSpec(raw=_RAW, derived=_DERIVED, risk_features=_RISK, input_allowlist=_ALLOWLIST)

# --------------------------------------------------------------------------
# LabelHierarchy — gate binario de peligro (INV-9)
# --------------------------------------------------------------------------

HIERARCHY = LabelHierarchy(
    root="gate_danger",
    nodes=(
        Node("gate_danger", ("branch_danger", "branch_peripheral")),
        Node("branch_danger", ("central_suspected", "cardiogenic_suspected")),
        Node("branch_peripheral",
             ("node_bppv", "meniere", "vestibular_migraine", "vestibular_neuritis", "labyrinthitis")),
        Node("node_bppv", ("bppv_posterior", "bppv_horizontal")),
    ),
    leaves=(
        "bppv_posterior", "bppv_horizontal", "meniere", "vestibular_migraine",
        "vestibular_neuritis", "labyrinthitis", "central_suspected", "cardiogenic_suspected",
    ),
    danger_child="branch_danger",
    abstain_label="undetermined",
)

# --------------------------------------------------------------------------
# SyntheticSpec — priors PROVISIONALES (T-CLIN los firma). Documentado en cards.
# Solo se listan las distribuciones DISCRIMINANTES por label; las no listadas
# usan defaults neutrales del generador (TB1.2).
# --------------------------------------------------------------------------


def _p(**kw: float) -> dict[str, float]:
    return dict(kw)


_PROFILES: tuple[LabelProfile, ...] = (
    LabelProfile(
        "bppv_posterior", prevalence=0.28,
        categorical={
            "duration": _p(under_1min=0.65, seconds=0.30, minutes=0.05),
            "trigger": _p(positional_head=0.90, spontaneous=0.10),
            "timing_pattern": _p(episodic_triggered=0.85, episodic_spontaneous=0.15),
            "onset": _p(sudden=0.70, gradual=0.30),
            "dix_hallpike": _p(right_positive=0.42, left_positive=0.42, negative=0.08, not_done=0.08),
            # VPPB NO tiene nistagmo espontáneo puro torsional/vertical (eso es
            # CENTRAL, A2). Su nistagmo es posicional (dix_hallpike) + fatigable.
            "nystagmus_direction": _p(none=0.60, mixed=0.25, horizontal=0.15),
        },
        boolean={"nystagmus_fatigable": 0.85},  # signo benigno clave de VPPB
        numeric={"nystagmus_latency_s": NumericDist(5, 3, 1, 20)},
    ),
    LabelProfile(
        "bppv_horizontal", prevalence=0.07,
        categorical={
            "duration": _p(under_1min=0.6, seconds=0.35, minutes=0.05),
            "trigger": _p(positional_head=0.88, spontaneous=0.12),
            "timing_pattern": _p(episodic_triggered=0.85, episodic_spontaneous=0.15),
            "nystagmus_direction": _p(horizontal=0.7, mixed=0.2, none=0.1),
            "dix_hallpike": _p(negative=0.4, right_positive=0.25, left_positive=0.25, not_done=0.1),
        },
        boolean={"nystagmus_fatigable": 0.7},  # posicional/benigno
        numeric={"nystagmus_latency_s": NumericDist(3, 2, 0, 12)},
    ),
    LabelProfile(
        "meniere", prevalence=0.10,
        categorical={
            "duration": _p(hours=0.70, minutes=0.25, days=0.05),
            "timing_pattern": _p(episodic_spontaneous=0.80, episodic_triggered=0.20),
            "trigger": _p(spontaneous=0.85, sound_pressure=0.15),
            "hearing_loss": _p(fluctuating=0.70, sudden_unilateral=0.15, none=0.15),
        },
        boolean={"tinnitus": 0.80, "aural_fullness": 0.70},
        numeric={"episode_count": NumericDist(5, 3, 1, 20)},
    ),
    LabelProfile(
        "vestibular_migraine", prevalence=0.14,
        categorical={
            "duration": _p(hours=0.45, minutes=0.35, days=0.20),
            "timing_pattern": _p(episodic_spontaneous=0.6, episodic_triggered=0.25, chronic=0.15),
            "hearing_loss": _p(none=0.85, fluctuating=0.15),
        },
        boolean={"migrainous_features": 0.85},
    ),
    LabelProfile(
        "vestibular_neuritis", prevalence=0.12,
        categorical={
            "duration": _p(over_24h_continuous=0.70, days=0.25, hours=0.05),
            "timing_pattern": _p(acute_continuous=0.88, episodic_spontaneous=0.12),
            "onset": _p(sudden=0.75, gradual=0.25),
            "head_impulse": _p(abnormal_corrective_saccade=0.78, normal=0.10, not_done=0.12),
            "hearing_loss": _p(none=0.88, sudden_unilateral=0.12),
            "nystagmus_direction": _p(horizontal=0.65, mixed=0.20, none=0.15),
        },
    ),
    LabelProfile(
        "labyrinthitis", prevalence=0.06,
        categorical={
            "duration": _p(over_24h_continuous=0.65, days=0.30, hours=0.05),
            "timing_pattern": _p(acute_continuous=0.85, episodic_spontaneous=0.15),
            "head_impulse": _p(abnormal_corrective_saccade=0.70, normal=0.12, not_done=0.18),
            "hearing_loss": _p(sudden_unilateral=0.60, fluctuating=0.20, none=0.20),
        },
        boolean={"tinnitus": 0.40},
    ),
    LabelProfile(
        "central_suspected", prevalence=0.13,
        categorical={
            "duration": _p(over_24h_continuous=0.55, hours=0.30, days=0.15),
            "timing_pattern": _p(acute_continuous=0.72, episodic_spontaneous=0.28),
            "head_impulse": _p(normal=0.70, abnormal_corrective_saccade=0.15, not_done=0.15),
            # Nistagmo espontáneo puro torsional/vertical o cambiante = CENTRAL (A2/A3).
            "nystagmus_direction": _p(direction_changing=0.40, vertical_pure=0.20,
                                      torsional_pure=0.20, horizontal=0.10, mixed=0.10),
        },
        boolean={
            "skew_deviation": 0.50, "nystagmus_direction_changing_gaze": 0.50,
            "truncal_ataxia_severe": 0.45, "headache_neck_pain_sudden_severe": 0.35,
            "nystagmus_fatigable": 0.02,  # central NO es fatigable (persistente)
        },
        numeric={
            "age_years": NumericDist(66, 10, 30, 90),
            "focal_signs": NumericDist(1.2, 1.0, 0, 4),          # conteo (transform-input)
            "vascular_risk_factors": NumericDist(1.6, 1.1, 0, 5),  # conteo (transform-input)
        },
    ),
    LabelProfile(
        "cardiogenic_suspected", prevalence=0.10,
        categorical={
            "duration": _p(seconds=0.45, under_1min=0.35, minutes=0.20),
            "trigger": _p(orthostatic=0.55, spontaneous=0.35, valsalva=0.10),
            "timing_pattern": _p(episodic_spontaneous=0.6, episodic_triggered=0.4),
        },
        boolean={"presyncope_syncope": 0.75, "palpitations": 0.55, "chest_pain": 0.30},
        numeric={
            "age_years": NumericDist(63, 12, 30, 90),
            "vascular_risk_factors": NumericDist(1.3, 1.0, 0, 5),
        },
    ),
)

SYNTHETIC = SyntheticSpec(profiles=_PROFILES, n_samples=8000, seed=SEED, missing_rate=0.15)

# --------------------------------------------------------------------------
# El Domain (bundle que instancia la plataforma)
# --------------------------------------------------------------------------

VERTIGO = Domain(name="vertigo", features=FEATURES, hierarchy=HIERARCHY, synthetic=SYNTHETIC)
