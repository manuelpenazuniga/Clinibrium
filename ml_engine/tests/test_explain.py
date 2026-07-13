"""TB1.6 — local SHAP of the danger gate (bounded, single node)."""
import dataclasses
import functools

from ml_engine.core.model import HierarchicalCatBoost
from ml_engine.core.synth import generate
from ml_engine.domains import vertigo

FEATURES = vertigo.FEATURES


@functools.lru_cache(maxsize=1)
def _model() -> HierarchicalCatBoost:
    spec = dataclasses.replace(vertigo.SYNTHETIC, n_samples=1500)
    df = generate(spec, FEATURES, seed=20260711)
    return HierarchicalCatBoost.train(
        vertigo.VERTIGO, df, seed=20260711, params={"iterations": 120, "depth": 4}
    )


def test_shap_keys_are_known_features_and_top_k() -> None:
    model = _model()
    shap = model.explain_gate(
        {"trigger": "positional_head", "duration": "under_1min"}, top_k=6
    )
    assert 0 < len(shap) <= 6
    assert set(shap).issubset(set(FEATURES.feature_names))
    assert all(isinstance(v, float) for v in shap.values())


def test_central_case_risk_features_push_toward_danger() -> None:
    """In a central case, some risk feature must contribute POSITIVELY
    (push toward danger) — the explanation is coherent with the prediction."""
    model = _model()
    central = {
        "timing_pattern": "acute_continuous", "head_impulse": "normal",
        "skew_deviation": True, "focal_signs": ["dysarthria", "limb_weakness"],
        "truncal_ataxia_severe": True,
        "vascular_risk_factors": ["hypertension", "atrial_fibrillation"], "age_years": 74,
    }
    shap = model.explain_gate(central, top_k=8)
    risk = set(FEATURES.risk_features)
    risk_contribs = [v for k, v in shap.items() if k in risk]
    assert risk_contribs, "expected risk features among the top-k"
    assert max(risk_contribs) > 0, "some risk feature must push toward danger"


def test_shap_is_single_node_no_cross_aggregation() -> None:
    """Honesty contract: explain_gate explains ONLY the gate (one node), it
    does not aggregate across the tree. We verify it returns a single flat
    dict of features (no per-node/per-class structure)."""
    model = _model()
    shap = model.explain_gate({"trigger": "spontaneous"})
    assert isinstance(shap, dict)
    assert all(isinstance(k, str) and isinstance(v, float) for k, v in shap.items())
