"""SYNTHETIC data generator (domain-agnostic, seeded, reproducible).

Consumes a ``SyntheticSpec`` (per-label priors) + ``FeatureSpec`` (which
features exist) and produces a ``DataFrame`` with a ``label`` column + the raw
features + the input fields of the derived features (e.g. counts).

HONESTY (AD-17): the data does NOT come from real patients; it measures
recovery of the generative process, not clinical validity. NUMERIC
reproducibility under a locked environment + tolerance (not "byte-identical"
across platforms).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml_engine.core.spec import FeatureSpec, LabelProfile, NumericDist, RawFeature, SyntheticSpec

# Probability of the "neutral" category when the profile does not specify a
# categorical feature (e.g. hearing_loss=none if the label says nothing about hearing).
_NEUTRAL_CAT_P = 0.85
# Default P(True) for a boolean not specified by the profile.
_DEFAULT_BOOL_P = 0.03


def _integer_counts(prev: np.ndarray, n: int) -> list[int]:
    """Integer counts per label that sum to EXACTLY n (largest-remainder, deterministic)."""
    exact = prev * n
    base = np.floor(exact).astype(int)
    remainder = n - int(base.sum())
    # distribute the remainder to the largest fractional parts (stable order)
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
    # neutral: mass on none/not_done/unknown, rest uniform
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
    # numeric: if the profile specifies it, sample; otherwise missing (NaN)
    dist = profile.numeric.get(rf.name)
    if dist is not None:
        return _sample_numeric(dist, rng)
    return float("nan")


def generate(
    spec: SyntheticSpec, features: FeatureSpec, *, seed: int | None = None
) -> pd.DataFrame:
    """Generates the synthetic dataset. Same seed ⇒ same DataFrame (reproducible).

    Columns: ``label`` + raw features + input fields of derived features (those
    a profile declares in ``numeric`` without being a raw feature, e.g. counts of
    ``focal_signs`` / ``vascular_risk_factors``).
    """
    rng = np.random.RandomState(spec.seed if seed is None else seed)
    prev = np.array([p.prevalence for p in spec.profiles], dtype=float)
    prev = prev / prev.sum()
    counts = _integer_counts(prev, spec.n_samples)

    raw_names = {rf.name for rf in features.raw}
    rows: list[dict[str, object]] = []
    for profile, n_i in zip(spec.profiles, counts, strict=True):
        # input fields of derived features that this profile declares and are NOT raw
        extra_numeric = {f: d for f, d in profile.numeric.items() if f not in raw_names}
        for _ in range(n_i):
            row: dict[str, object] = {"label": profile.label}
            for rf in features.raw:
                row[rf.name] = _sample_raw(rf, profile, rng)
            for feat, dist in extra_numeric.items():
                row[feat] = _sample_numeric(dist, rng)
            # missingness: drop features (→ None) to simulate sparse inputs
            if spec.missing_rate > 0.0:
                for key in list(row):
                    if key != "label" and rng.random_sample() < spec.missing_rate:
                        row[key] = None
            rows.append(row)

    df = pd.DataFrame(rows)
    # deterministic shuffle (with the same rng)
    order = rng.permutation(len(df))
    return df.iloc[order].reset_index(drop=True)
