"""TB1.2 — synthetic generator + blind encode."""
import dataclasses

import numpy as np

from ml_engine.core.encode import encode
from ml_engine.core.synth import generate
from ml_engine.domains import vertigo

SPEC = vertigo.SYNTHETIC
FEATURES = vertigo.FEATURES


def test_synthetic_reproducible() -> None:
    df1 = generate(SPEC, FEATURES, seed=20260711)
    df2 = generate(SPEC, FEATURES, seed=20260711)
    assert df1.equals(df2)


def test_different_seed_differs() -> None:
    df1 = generate(SPEC, FEATURES, seed=1)
    df2 = generate(SPEC, FEATURES, seed=2)
    assert not df1.equals(df2)


def test_row_count_matches_n_samples() -> None:
    df = generate(SPEC, FEATURES, seed=20260711)
    assert len(df) == SPEC.n_samples
    assert set(df["label"].unique()) == set(SPEC.labels)


def test_bppv_prior_is_positional() -> None:
    df = generate(SPEC, FEATURES, seed=20260711)
    bppv = df[df["label"] == "bppv_posterior"]
    # robust to missingness: measure over the present values
    trig = bppv["trigger"].dropna()
    assert (trig == "positional_head").mean() > 0.7


def test_missingness_drops_some_features() -> None:
    """SPEC has missing_rate>0 → some features end up missing (sparse)."""
    df = generate(SPEC, FEATURES, seed=20260711)
    assert df["trigger"].isna().mean() > 0.05
    # without missingness: categoricals always present
    df0 = generate(dataclasses.replace(SPEC, missing_rate=0.0), FEATURES, seed=1)
    assert df0["trigger"].isna().sum() == 0


def test_bppv_has_no_spontaneous_pure_torsional() -> None:
    """P0.5 fix: the BPPV profile must NOT emit pure torsional/vertical (that is central)."""
    df = generate(dataclasses.replace(SPEC, missing_rate=0.0), FEATURES, seed=1)
    bppv = df[df["label"] == "bppv_posterior"]["nystagmus_direction"]
    assert (bppv == "torsional_pure").sum() == 0
    assert (bppv == "vertical_pure").sum() == 0


def test_central_prior_has_danger_signal() -> None:
    df = generate(SPEC, FEATURES, seed=20260711)
    x, _ = encode(df, FEATURES)
    x["label"] = df["label"].values
    central = x[x["label"] == "central_suspected"]
    bppv = x[x["label"] == "bppv_posterior"]
    # the danger gate must see more risk signal in central than in BPPV
    assert central["danger_sign_count"].mean() > bppv["danger_sign_count"].mean()
    assert central["hints_central_pattern"].mean() > bppv["hints_central_pattern"].mean()


def test_encode_shape_and_categoricals() -> None:
    df = generate(SPEC, FEATURES, seed=20260711)
    x, cat_cols = encode(df, FEATURES)
    assert list(x.columns) == list(FEATURES.feature_names)
    assert set(cat_cols) == set(FEATURES.categorical_names)
    # categoricals as string, numerics as float
    for c in cat_cols:
        assert x[c].dtype == object
    for c in FEATURES.numeric_feature_names:
        assert np.issubdtype(x[c].dtype, np.floating)


def test_encode_derived_never_nan() -> None:
    """Derived features are NEVER NaN (even when focal_signs is missing → 0)."""
    df = generate(SPEC, FEATURES, seed=20260711)
    x, _ = encode(df, FEATURES)
    for d in FEATURES.derived:
        assert x[d.name].isna().sum() == 0, f"{d.name} has NaN"


def test_encode_missing_categorical_becomes_sentinel() -> None:
    rows = [{"trigger": None, "duration": "hours"}]
    x, _ = encode(rows, FEATURES)
    assert x["trigger"].iloc[0] == "__nan__"
    assert x["duration"].iloc[0] == "hours"


def test_encode_from_realistic_casefeatures_row() -> None:
    """'Serving' form: focal_signs as a LIST (not a count) → correct derived features."""
    row = {
        "trigger": "spontaneous", "timing_pattern": "acute_continuous",
        "head_impulse": "normal", "focal_signs": ["dysarthria", "diplopia"],
        "truncal_ataxia_severe": True, "vascular_risk_factors": ["hypertension"], "age_years": 70,
    }
    x, _ = encode([row], FEATURES)
    assert x["danger_sign_count"].iloc[0] == 3.0          # 2 focal + ataxia
    assert x["hints_central_pattern"].iloc[0] == 1.0      # normal + acute
    assert x["vascular_risk_count"].iloc[0] == 2.0        # 1 factor + age>=60
