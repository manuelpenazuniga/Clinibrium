"""BLIND feature executor (domain-agnostic).

Takes raw rows (dicts or DataFrame) + a ``FeatureSpec`` and produces the
feature matrix for CatBoost: categoricals as string, booleans/numerics as
float, and the derived features by applying the PURE transformers the domain
declared (the core knows no domain logic — Gemini fix #3).

Missing-value convention: missing categorical → ``"__nan__"`` (its own
category, CatBoost handles it); missing numeric → ``NaN`` (CatBoost handles it
natively).
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
    # boolean → {0,1}; missing → 0 (semantics: feature not present)
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
    """Encodes raw rows → (X, categorical_feature_names).

    ``X`` has columns in ``features.feature_names`` order (raw + derived).
    Derived features are computed by applying ``DerivedFeature.fn`` to the raw row.
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
    # explicit dtypes: categoricals as str, rest float
    for name in features.categorical_names:
        x[name] = x[name].astype(str)
    for name in features.numeric_feature_names:
        x[name] = pd.to_numeric(x[name], errors="coerce").astype(float)

    x = x[list(features.feature_names)]
    x = x.replace({np.nan: np.nan})  # normalizes any residual numeric None
    return x, list(features.categorical_names)
