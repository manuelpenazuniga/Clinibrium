"""TB1.2 — generador sintético + encode ciego."""
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
    assert (bppv["trigger"] == "positional_head").mean() > 0.7


def test_central_prior_has_danger_signal() -> None:
    df = generate(SPEC, FEATURES, seed=20260711)
    x, _ = encode(df, FEATURES)
    x["label"] = df["label"].values
    central = x[x["label"] == "central_suspected"]
    bppv = x[x["label"] == "bppv_posterior"]
    # el gate de peligro debe ver más señal de riesgo en central que en BPPV
    assert central["danger_sign_count"].mean() > bppv["danger_sign_count"].mean()
    assert central["hints_central_pattern"].mean() > bppv["hints_central_pattern"].mean()


def test_encode_shape_and_categoricals() -> None:
    df = generate(SPEC, FEATURES, seed=20260711)
    x, cat_cols = encode(df, FEATURES)
    assert list(x.columns) == list(FEATURES.feature_names)
    assert set(cat_cols) == set(FEATURES.categorical_names)
    # categóricas como string, numéricas como float
    for c in cat_cols:
        assert x[c].dtype == object
    for c in FEATURES.numeric_feature_names:
        assert np.issubdtype(x[c].dtype, np.floating)


def test_encode_derived_never_nan() -> None:
    """Las derivadas NUNCA son NaN (aun cuando focal_signs esté ausente → 0)."""
    df = generate(SPEC, FEATURES, seed=20260711)
    x, _ = encode(df, FEATURES)
    for d in FEATURES.derived:
        assert x[d.name].isna().sum() == 0, f"{d.name} tiene NaN"


def test_encode_missing_categorical_becomes_sentinel() -> None:
    rows = [{"trigger": None, "duration": "hours"}]
    x, _ = encode(rows, FEATURES)
    assert x["trigger"].iloc[0] == "__nan__"
    assert x["duration"].iloc[0] == "hours"


def test_encode_from_realistic_casefeatures_row() -> None:
    """Forma 'serving': focal_signs como LISTA (no conteo) → derivadas correctas."""
    row = {
        "trigger": "spontaneous", "timing_pattern": "acute_continuous",
        "head_impulse": "normal", "focal_signs": ["dysarthria", "diplopia"],
        "truncal_ataxia_severe": True, "vascular_risk_factors": ["hypertension"], "age_years": 70,
    }
    x, _ = encode([row], FEATURES)
    assert x["danger_sign_count"].iloc[0] == 3.0          # 2 focal + ataxia
    assert x["hints_central_pattern"].iloc[0] == 1.0      # normal + acute
    assert x["vascular_risk_count"].iloc[0] == 2.0        # 1 factor + edad≥60
