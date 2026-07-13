"""TB1.8 — train CLI: train → save → load → predict (roundtrip)."""
import re

from ml_engine.core.model import HierarchicalCatBoost
from ml_engine.domains import vertigo
from ml_engine.train import build_and_save


def test_train_save_load_roundtrip(tmp_path) -> None:
    model, metrics = build_and_save(
        tmp_path, seed=20260711, n_samples=1200, params={"iterations": 100, "depth": 4}
    )
    # model_version declares synthetic + seed (AD-17)
    assert re.match(r"^synthetic-v\d+-seed\d+$", model.model_version)

    # artifacts written
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "MODEL_CARD.md").exists()
    card = (tmp_path / "MODEL_CARD.md").read_text()
    assert "SYNTHETIC" in card and "generator recovery" in card

    # calibration + abstention persisted (TB1.4/1.5)
    loaded = HierarchicalCatBoost.load(tmp_path, vertigo.VERTIGO)
    assert loaded.model_version == model.model_version
    assert loaded.calibrator is not None and loaded.calibrator.temperature > 0
    assert loaded.abstainer is not None
    assert abs(loaded.calibrator.temperature - model.calibrator.temperature) < 1e-9
    assert abs(loaded.abstainer.threshold - model.abstainer.threshold) < 1e-9

    row = {"trigger": "positional_head", "duration": "under_1min",
           "timing_pattern": "episodic_triggered", "dix_hallpike": "right_positive"}
    # raw 8-leaf identical mem↔load
    p_mem = model.predict_proba_one(row)
    p_load = loaded.predict_proba_one(row)
    assert set(p_load) == set(vertigo.HIERARCHY.leaves)
    for leaf in p_mem:
        assert abs(p_mem[leaf] - p_load[leaf]) < 1e-9
    # predict_case = 9 keys (with undetermined), Σ≈1
    case = loaded.predict_case(row)
    assert "undetermined" in case and len(case) == 9
    assert abs(sum(case.values()) - 1.0) < 1e-6


def test_metrics_report_ece_and_abstention() -> None:
    _, m = build_and_save(
        __import__("tempfile").mkdtemp(), seed=20260711, n_samples=1500,
        params={"iterations": 100, "depth": 4},
    )
    assert 0.0 <= m.ece_raw <= 1.0 and 0.0 <= m.ece_calibrated <= 1.0
    assert 0.0 <= m.abstain_rate <= 1.0
    assert m.temperature > 0.0


def test_metrics_are_sane() -> None:
    _, m = build_and_save(
        __import__("tempfile").mkdtemp(), seed=20260711, n_samples=1200,
        params={"iterations": 100, "depth": 4},
    )
    # generator recovery: high, but it is synthetic (not clinical)
    assert 0.0 <= m.leaf_accuracy <= 1.0
    assert m.gate_accuracy > 0.8
