"""TB1.1 — tipos core (agnósticos) + config del dominio vértigo."""
import math

import pytest

from ml_engine.core.spec import Domain, FeatureSpec, LabelHierarchy, Node, RawFeature
from ml_engine.domains import vertigo


# --- FeatureSpec: validación ---------------------------------------------

def test_risk_features_must_be_numeric() -> None:
    with pytest.raises(ValueError):
        FeatureSpec(
            raw=(RawFeature("cat", "categorical", ("a", "b")),),
            risk_features=("cat",),  # categórica → inválida como risk
        )


def test_no_duplicate_feature_names() -> None:
    with pytest.raises(ValueError):
        FeatureSpec(raw=(RawFeature("x", "numeric"), RawFeature("x", "boolean")))


def test_feature_names_stable_order() -> None:
    fs = FeatureSpec(
        raw=(RawFeature("a", "numeric"), RawFeature("b", "boolean")),
        risk_features=("a",),
    )
    assert fs.feature_names == ("a", "b")
    assert "a" in fs.numeric_feature_names and "b" in fs.numeric_feature_names


# --- LabelHierarchy: buena formación y caminos ---------------------------

def test_root_gate_must_be_binary() -> None:
    with pytest.raises(ValueError):
        LabelHierarchy(
            root="r",
            nodes=(Node("r", ("a", "b", "c")),),
            leaves=("a", "b", "c"),
            danger_child="a",
        )


def test_vertigo_hierarchy_paths() -> None:
    h = vertigo.HIERARCHY
    # gate binario, danger_child válido
    assert len(h.node_by_id(h.root).children) == 2
    assert h.danger_child in h.node_by_id(h.root).children
    # camino a una hoja de peligro pasa por el branch de peligro
    path_central = h.path_to_leaf("central_suspected")
    assert path_central[0] == ("gate_danger", "branch_danger")
    # camino a BPPV posterior pasa por peripheral → node_bppv
    path_bppv = h.path_to_leaf("bppv_posterior")
    node_ids = [n for n, _ in path_bppv]
    assert node_ids == ["gate_danger", "branch_peripheral", "node_bppv"]
    # undetermined NO es hoja entrenada
    assert "undetermined" not in set(h.leaves)


# --- Domain: consistencia jerarquía ↔ generador --------------------------

def test_vertigo_domain_consistent() -> None:
    assert isinstance(vertigo.VERTIGO, Domain)
    assert set(vertigo.HIERARCHY.leaves) == set(vertigo.SYNTHETIC.labels)


def test_prevalences_sum_to_one() -> None:
    total = sum(p.prevalence for p in vertigo.SYNTHETIC.profiles)
    assert math.isclose(total, 1.0, abs_tol=1e-6)


# --- Derivadas: puras y NaN-safe -----------------------------------------

def test_derived_are_nan_safe_on_empty_row() -> None:
    empty: dict[str, object] = {}
    for d in vertigo.FEATURES.derived:
        val = d.fn(empty)
        assert isinstance(val, float) and not math.isnan(val)
        assert val == 0.0  # fila vacía → 0, nunca excepción


def test_danger_sign_count_list_and_count_forms() -> None:
    # forma serving (lista) y forma synth (conteo) dan lo mismo
    row_list = {"focal_signs": ["dysarthria", "diplopia"], "truncal_ataxia_severe": True}
    row_num = {"focal_signs": 2, "truncal_ataxia_severe": True}
    assert vertigo.danger_sign_count(row_list) == vertigo.danger_sign_count(row_num) == 3.0


def test_hints_central_pattern() -> None:
    assert vertigo.hints_central_pattern(
        {"head_impulse": "normal", "timing_pattern": "acute_continuous"}
    ) == 1.0
    # head-impulse normal pero NO en AVS → no dispara
    assert vertigo.hints_central_pattern(
        {"head_impulse": "normal", "timing_pattern": "episodic_triggered"}
    ) == 0.0


def test_vascular_risk_count_age_gate() -> None:
    assert vertigo.vascular_risk_count({"vascular_risk_factors": ["hypertension"], "age_years": 70}) == 2.0
    assert vertigo.vascular_risk_count({"vascular_risk_factors": ["hypertension"], "age_years": 40}) == 1.0
    # bool no debe contarse como edad
    assert vertigo.vascular_risk_count({"age_years": True}) == 0.0


def test_cardiogenic_cluster() -> None:
    assert vertigo.cardiogenic_cluster(
        {"presyncope_syncope": True, "palpitations": True, "chest_pain": True, "trigger": "orthostatic"}
    ) == 4.0
    assert vertigo.cardiogenic_cluster({}) == 0.0


def test_all_risk_features_are_numeric() -> None:
    numeric = set(vertigo.FEATURES.numeric_feature_names)
    assert set(vertigo.FEATURES.risk_features).issubset(numeric)
