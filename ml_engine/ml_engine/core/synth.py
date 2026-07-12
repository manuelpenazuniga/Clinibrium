"""Generador de datos SINTÉTICOS (agnóstico de dominio, seeded, reproducible).

Consume un ``SyntheticSpec`` (priors por label) + ``FeatureSpec`` (qué features
existen) y produce un ``DataFrame`` con una columna ``label`` + las features
crudas + los campos-input de las derivadas (p.ej. conteos).

HONESTIDAD (AD-17): los datos NO son de pacientes reales; miden recuperación
del proceso generativo, no validez clínica. Reproducibilidad NUMÉRICA bajo
entorno bloqueado + tolerancia (no "byte-idéntico" entre plataformas).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml_engine.core.spec import FeatureSpec, LabelProfile, NumericDist, RawFeature, SyntheticSpec

# Probabilidad de la categoría "neutral" cuando el profile no especifica una
# feature categórica (p.ej. hearing_loss=none si el label no habla de audición).
_NEUTRAL_CAT_P = 0.85
# P(True) por defecto para una booleana no especificada por el profile.
_DEFAULT_BOOL_P = 0.03


def _integer_counts(prev: np.ndarray, n: int) -> list[int]:
    """Conteos enteros por label que suman EXACTO n (mayor-resto, determinista)."""
    exact = prev * n
    base = np.floor(exact).astype(int)
    remainder = n - int(base.sum())
    # reparte el resto a los de mayor parte fraccionaria (orden estable)
    frac = exact - base
    order = np.argsort(-frac, kind="stable")
    for i in range(remainder):
        base[order[i]] += 1
    return base.tolist()


def _neutral_category(cats: tuple[str, ...]) -> str:
    for pref in ("none", "not_done", "unknown"):
        if pref in cats:
            return pref
    return cats[0]


def _sample_categorical(
    rf: RawFeature, profile: LabelProfile, rng: np.random.RandomState
) -> str:
    dist = profile.categorical.get(rf.name)
    cats = rf.categories
    if dist:
        probs = np.array([dist.get(c, 0.0) for c in cats], dtype=float)
        total = probs.sum()
        if total <= 0:
            probs = np.ones(len(cats)) / len(cats)
        else:
            probs = probs / total
        return str(rng.choice(cats, p=probs))
    # neutral: masa en none/not_done/unknown, resto uniforme
    neutral = _neutral_category(cats)
    probs = np.full(len(cats), (1.0 - _NEUTRAL_CAT_P) / max(len(cats) - 1, 1))
    probs[cats.index(neutral)] = _NEUTRAL_CAT_P
    probs = probs / probs.sum()
    return str(rng.choice(cats, p=probs))


def _sample_numeric(dist: NumericDist, rng: np.random.RandomState) -> float:
    val = rng.normal(dist.mean, dist.std)
    return float(np.clip(val, dist.lo, dist.hi))


def _sample_raw(
    rf: RawFeature, profile: LabelProfile, rng: np.random.RandomState
) -> object:
    if rf.kind == "categorical":
        return _sample_categorical(rf, profile, rng)
    if rf.kind == "boolean":
        p = profile.boolean.get(rf.name, _DEFAULT_BOOL_P)
        return bool(rng.random_sample() < p)
    # numeric: si el profile lo especifica, muestrear; si no, faltante (NaN)
    dist = profile.numeric.get(rf.name)
    if dist is not None:
        return _sample_numeric(dist, rng)
    return float("nan")


def generate(
    spec: SyntheticSpec, features: FeatureSpec, *, seed: int | None = None
) -> pd.DataFrame:
    """Genera el dataset sintético. Mismo seed ⇒ mismo DataFrame (reproducible).

    Columnas: ``label`` + features crudas + campos-input de derivadas (los que
    algún profile declara en ``numeric`` sin ser raw feature, p.ej. conteos de
    ``focal_signs`` / ``vascular_risk_factors``).
    """
    rng = np.random.RandomState(spec.seed if seed is None else seed)
    prev = np.array([p.prevalence for p in spec.profiles], dtype=float)
    prev = prev / prev.sum()
    counts = _integer_counts(prev, spec.n_samples)

    raw_names = {rf.name for rf in features.raw}
    rows: list[dict[str, object]] = []
    for profile, n_i in zip(spec.profiles, counts, strict=True):
        # campos-input de derivadas que este profile declara y NO son raw
        extra_numeric = {f: d for f, d in profile.numeric.items() if f not in raw_names}
        for _ in range(n_i):
            row: dict[str, object] = {"label": profile.label}
            for rf in features.raw:
                row[rf.name] = _sample_raw(rf, profile, rng)
            for feat, dist in extra_numeric.items():
                row[feat] = _sample_numeric(dist, rng)
            # missingness: dropear features (→ None) para simular inputs esparsos
            if spec.missing_rate > 0.0:
                for key in list(row):
                    if key != "label" and rng.random_sample() < spec.missing_rate:
                        row[key] = None
            rows.append(row)

    df = pd.DataFrame(rows)
    # shuffle determinista (con el mismo rng)
    order = rng.permutation(len(df))
    return df.iloc[order].reset_index(drop=True)
