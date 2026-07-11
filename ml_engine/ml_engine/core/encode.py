"""Ejecutor CIEGO de features (agnóstico de dominio).

Toma filas crudas (dicts o DataFrame) + un ``FeatureSpec`` y produce la matriz
de features para CatBoost: categóricas como string, booleanas/numéricas como
float, y las derivadas aplicando los transformadores PUROS que el dominio
declaró (el core no conoce ninguna lógica de dominio — fix Gemini #3).

Convención de faltantes: categórica ausente → ``"__nan__"`` (categoría propia,
CatBoost la maneja); numérica ausente → ``NaN`` (CatBoost la maneja nativo).
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from ml_engine.core.spec import FeatureSpec

_CAT_MISSING = "__nan__"


def _is_missing(v: object) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _to_cat(v: object) -> str:
    return _CAT_MISSING if _is_missing(v) else str(v)


def _to_float01(v: object) -> float:
    # booleana → {0,1}; ausente → 0 (semántica: feature no presente)
    if _is_missing(v):
        return 0.0
    return 1.0 if v else 0.0


def _to_numeric(v: object) -> float:
    if _is_missing(v):
        return float("nan")
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _rows(data: pd.DataFrame | list[Mapping[str, object]]) -> list[dict[str, object]]:
    if isinstance(data, pd.DataFrame):
        return data.to_dict("records")  # type: ignore[return-value]
    return [dict(r) for r in data]


def encode(
    data: pd.DataFrame | list[Mapping[str, object]], features: FeatureSpec
) -> tuple[pd.DataFrame, list[str]]:
    """Codifica filas crudas → (X, categorical_feature_names).

    ``X`` tiene columnas en el orden ``features.feature_names`` (raw + derivadas).
    Las derivadas se computan aplicando ``DerivedFeature.fn`` a la fila cruda.
    """
    records = _rows(data)
    cols: dict[str, list[object]] = {}

    for rf in features.raw:
        raw_vals = [r.get(rf.name) for r in records]
        if rf.kind == "categorical":
            cols[rf.name] = [_to_cat(v) for v in raw_vals]
        elif rf.kind == "boolean":
            cols[rf.name] = [_to_float01(v) for v in raw_vals]
        else:
            cols[rf.name] = [_to_numeric(v) for v in raw_vals]

    for d in features.derived:
        cols[d.name] = [float(d.fn(r)) for r in records]

    x = pd.DataFrame(cols)
    # dtypes explícitos: categóricas como str, resto float
    for name in features.categorical_names:
        x[name] = x[name].astype(str)
    for name in features.numeric_feature_names:
        x[name] = pd.to_numeric(x[name], errors="coerce").astype(float)

    x = x[list(features.feature_names)]
    x = x.replace({np.nan: np.nan})  # normaliza cualquier None residual numérico
    return x, list(features.categorical_names)
