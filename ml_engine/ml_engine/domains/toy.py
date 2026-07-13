"""TOY domain (instance #2) — platform agnosticism proof (INV-12).

It is NOT vertigo nor clinical: a deliberately different, **NON-isomorphic**
domain (different hierarchy, different feature names/types, different number
of classes and levels) that trains and serves through the SAME core factory,
changing ONLY this config file. Its existence proves that the core
(`ml_engine.core.*`) has nothing vertigo-specific hardcoded.

Hierarchy (non-isomorphic to vertigo: 3 leaves, mixed depth):
    g0 (binary gate) → {branch_high, leaf_z}
                          └ branch_high → {leaf_x, leaf_y}
"""
from __future__ import annotations

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

SEED = 7


def _score(row: Row) -> float:
    """Derived feature of the toy domain (pure): combines two raw signals."""
    a = row.get("signal_a")
    b = row.get("flag_b")
    val = (float(a) if isinstance(a, (int, float)) and not isinstance(a, bool) else 0.0)
    return val + (2.0 if b else 0.0)


_RAW = (
    RawFeature("color", "categorical", ("red", "green", "blue")),
    RawFeature("flag_b", "boolean"),
    RawFeature("signal_a", "numeric"),
)
_DERIVED = (DerivedFeature("score", _score),)
# risk feature (monotone toward the "high" branch): the derived score
_RISK = ("score",)

FEATURES = FeatureSpec(raw=_RAW, derived=_DERIVED, risk_features=_RISK)

HIERARCHY = LabelHierarchy(
    root="g0",
    nodes=(
        Node("g0", ("branch_high", "leaf_z")),
        Node("branch_high", ("leaf_x", "leaf_y")),
    ),
    leaves=("leaf_x", "leaf_y", "leaf_z"),
    danger_child="branch_high",
    abstain_label="unknown",
)

SYNTHETIC = SyntheticSpec(
    profiles=(
        LabelProfile(
            "leaf_x", prevalence=0.30,
            categorical={"color": {"red": 0.7, "green": 0.2, "blue": 0.1}},
            boolean={"flag_b": 0.8},
            numeric={"signal_a": NumericDist(8, 2, 0, 12)},
        ),
        LabelProfile(
            "leaf_y", prevalence=0.30,
            categorical={"color": {"green": 0.7, "red": 0.2, "blue": 0.1}},
            boolean={"flag_b": 0.7},
            numeric={"signal_a": NumericDist(6, 2, 0, 12)},
        ),
        LabelProfile(
            "leaf_z", prevalence=0.40,
            categorical={"color": {"blue": 0.8, "green": 0.1, "red": 0.1}},
            boolean={"flag_b": 0.05},
            numeric={"signal_a": NumericDist(1, 1, 0, 12)},
        ),
    ),
    n_samples=3000,
    seed=SEED,
)

TOY = Domain(name="toy", features=FEATURES, hierarchy=HIERARCHY, synthetic=SYNTHETIC)
