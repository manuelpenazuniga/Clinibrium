"""TB1.3 — HierarchicalCatBoost: monotone binary gate (INV-9) + path.

Small cached model (fewer rows/iterations) so the gate stays fast.
"""
import dataclasses
import functools

from ml_engine.core.encode import encode
from ml_engine.core.model import HierarchicalCatBoost
from ml_engine.core.synth import generate
from ml_engine.domains import vertigo

FEATURES = vertigo.FEATURES


@functools.lru_cache(maxsize=1)
def _small_model() -> HierarchicalCatBoost:
    spec = dataclasses.replace(vertigo.SYNTHETIC, n_samples=1500)
    df = generate(spec, FEATURES, seed=20260711)
    return HierarchicalCatBoost.train(
        vertigo.VERTIGO, df, seed=20260711, params={"iterations": 120, "depth": 4}
    )


def _grid_rows(n: int = 25):
    spec = dataclasses.replace(vertigo.SYNTHETIC, n_samples=n)
    df = generate(spec, FEATURES, seed=99)
    return encode(df, FEATURES)[0]


def test_leaf_probs_sum_to_one() -> None:
    model = _small_model()
    row = {"trigger": "positional_head", "timing_pattern": "episodic_triggered",
           "duration": "under_1min", "dix_hallpike": "right_positive"}
    p = model.predict_proba_one(row)
    assert set(p) == set(vertigo.HIERARCHY.leaves)
    assert abs(sum(p.values()) - 1.0) < 1e-6


def test_INV9_gate_monotone_in_risk_features() -> None:
    """Raising ANY risk feature (+1, ceteris paribus) NEVER lowers P(dangerous).

    Tested at the EXACT level CatBoost guarantees: the encoded matrix, a
    single feature moved, everything else fixed. This is the hard guarantee of
    INV-9 (gate, pre-abstention/pre-calibration).
    """
    model = _small_model()
    x = _grid_rows(25)
    risk = FEATURES.risk_features
    assert risk  # there are declared risk features
    for i in range(len(x)):
        base = x.iloc[[i]].copy()
        p_base = model.gate_danger_proba_encoded(base)
        for rf in risk:
            bumped = base.copy()
            col = bumped.columns.get_loc(rf)
            bumped.iloc[0, col] = float(base.iloc[0][rf]) + 1.0
            p_up = model.gate_danger_proba_encoded(bumped)
            assert p_up >= p_base - 1e-6, (
                f"row {i}: raising '{rf}' lowered P(dangerous) {p_base:.4f}→{p_up:.4f}"
            )


def test_INV9_holds_for_two_steps() -> None:
    """Cumulative monotonicity: +2 does not lower P(dangerous) either."""
    model = _small_model()
    x = _grid_rows(10)
    for i in range(len(x)):
        base = x.iloc[[i]].copy()
        p0 = model.gate_danger_proba_encoded(base)
        b2 = base.copy()
        col = b2.columns.get_loc("danger_sign_count")
        b2.iloc[0, col] = float(base.iloc[0]["danger_sign_count"]) + 2.0
        assert model.gate_danger_proba_encoded(b2) >= p0 - 1e-6


def test_clear_central_case_flags_danger() -> None:
    """Sanity: a clearly central case concentrates mass on the danger branch."""
    model = _small_model()
    central = {
        "timing_pattern": "acute_continuous", "head_impulse": "normal",
        "nystagmus_direction": "direction_changing", "skew_deviation": True,
        "focal_signs": ["dysarthria", "limb_weakness"], "truncal_ataxia_severe": True,
        "vascular_risk_factors": ["hypertension", "atrial_fibrillation"], "age_years": 72,
    }
    p = model.predict_proba_one(central)
    p_danger = p["central_suspected"] + p["cardiogenic_suspected"]
    assert p_danger > 0.5, f"P(danger)={p_danger:.3f} should dominate in a clear central case"


def test_clear_bppv_case_is_peripheral() -> None:
    """BPPV = POSITIONAL + fatigable pattern (dix_hallpike), NOT spontaneous torsional.

    (P0.5 fix: spontaneous torsional/vertical is CENTRAL — see the A↔B
    consistency test below; BPPV is represented by the positional pattern.)
    """
    model = _small_model()
    bppv = {
        "trigger": "positional_head", "timing_pattern": "episodic_triggered",
        "duration": "under_1min", "onset": "sudden", "dix_hallpike": "right_positive",
        "nystagmus_fatigable": True, "nystagmus_latency_s": 5.0,
    }
    p = model.predict_proba_one(bppv)
    p_danger = p["central_suspected"] + p["cardiogenic_suspected"]
    assert p_danger < 0.3, f"P(danger)={p_danger:.3f} should not dominate in a clear BPPV case"


def test_P05_spontaneous_torsional_not_classified_as_benign_bppv() -> None:
    """P0.5 fix — the SAFETY property that matters.

    The original bug: the generator put torsional_pure in the BPPV profile → B
    classified a SPONTANEOUS pure torsional/vertical nystagmus as **benign
    positional BPPV** (which would suggest Epley), contradicting A (red flag
    A2 → central).

    B is a soft PROBABILISTIC layer; A is the safety authority (hard rule,
    INV-1). We don't require B to replicate the hard rule, but it must NOT
    label the central sign as treatable BPPV. (The reconciliation direction
    toward danger is covered by the INV-9 grid test, which now includes
    `central_nystagmus_pattern` as a risk feature.)
    """
    model = _small_model()
    for direction in ("torsional_pure", "vertical_pure"):
        spont = {
            "trigger": "spontaneous", "timing_pattern": "acute_continuous",
            "onset": "sudden", "nystagmus_direction": direction,
        }
        p = model.predict_proba_one(spont)
        assert p["bppv_posterior"] < 0.15, (
            f"spontaneous {direction} → B calls it posterior BPPV "
            f"({p['bppv_posterior']:.3f}); contradicts A and would suggest Epley"
        )
        assert p["bppv_horizontal"] < 0.15


def test_P05_central_nystagmus_pattern_is_a_risk_feature() -> None:
    """The central nystagmus pattern feeds the danger gate (monotone)."""
    assert "central_nystagmus_pattern" in vertigo.FEATURES.risk_features
    # and pure torsional/vertical does NOT appear in the BPPV profile (tested in synth)
