"""TB1.9 — INV-12: a 2nd domain (toy, NOT isomorphic) trains+serves via config.

Platform proof: the SAME factory (`HierarchicalCatBoost.train`, `generate`,
`encode`, `predict_case`) is used with ANOTHER `LabelHierarchy`/`FeatureSpec`,
without touching `ml_engine.core.*`.
"""
import dataclasses

from ml_engine.core.model import HierarchicalCatBoost
from ml_engine.core.synth import generate
from ml_engine.domains import toy, vertigo
from ml_engine.train import train_domain


def test_toy_is_not_isomorphic_to_vertigo() -> None:
    # different tree shape and different leaf count → real agnosticism proof
    assert len(toy.HIERARCHY.leaves) != len(vertigo.HIERARCHY.leaves)
    assert toy.HIERARCHY.abstain_label != vertigo.HIERARCHY.abstain_label
    assert set(toy.FEATURES.feature_names).isdisjoint(set(vertigo.FEATURES.feature_names))


def test_INV12_second_domain_trains_and_serves_via_config_only() -> None:
    spec = dataclasses.replace(toy.SYNTHETIC, n_samples=1500)
    df = generate(spec, toy.FEATURES, seed=toy.SEED)
    model = HierarchicalCatBoost.train(
        toy.TOY, df, seed=toy.SEED, params={"iterations": 80, "depth": 3}
    )
    # trains and predicts over the toy's 3 leaves
    row = {"color": "blue", "flag_b": False, "signal_a": 1.0}
    p = model.predict_proba_one(row)
    assert set(p) == set(toy.HIERARCHY.leaves)
    assert abs(sum(p.values()) - 1.0) < 1e-6
    # clearly a z case (blue, no flag, low signal) → leaf_z dominates
    assert max(p, key=p.__getitem__) == "leaf_z"


def test_INV12_full_pipeline_with_calibration_and_abstention() -> None:
    """The full factory (calibration + abstention + predict_case) serves the toy."""
    model, metrics = train_domain(
        toy.TOY, seed=toy.SEED, n_samples=1500, params={"iterations": 80, "depth": 3}
    )
    assert model.calibrator is not None and model.abstainer is not None
    case = model.predict_case({"color": "red", "flag_b": True, "signal_a": 9.0})
    assert "unknown" in case and len(case) == len(toy.HIERARCHY.leaves) + 1
    assert abs(sum(case.values()) - 1.0) < 1e-6
    assert 0.0 <= metrics.leaf_accuracy <= 1.0
