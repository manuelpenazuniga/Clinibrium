"""TB1.8 — train CLI: entrenar → guardar → cargar → predecir (roundtrip)."""
import re

from ml_engine.core.model import HierarchicalCatBoost
from ml_engine.domains import vertigo
from ml_engine.train import build_and_save


def test_train_save_load_roundtrip(tmp_path) -> None:
    model, metrics = build_and_save(
        tmp_path, seed=20260711, n_samples=1200, params={"iterations": 100, "depth": 4}
    )
    # model_version declara sintético + seed (AD-17)
    assert re.match(r"^synthetic-v\d+-seed\d+$", model.model_version)

    # artifacts escritos
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "MODEL_CARD.md").exists()
    card = (tmp_path / "MODEL_CARD.md").read_text()
    assert "SINTÉTICO" in card and "recuperación del generador" in card

    # cargar y predecir: mismas probabilidades que el modelo en memoria
    loaded = HierarchicalCatBoost.load(tmp_path, vertigo.VERTIGO)
    assert loaded.model_version == model.model_version
    row = {"trigger": "positional_head", "duration": "under_1min",
           "timing_pattern": "episodic_triggered", "dix_hallpike": "right_positive"}
    p_mem = model.predict_proba_one(row)
    p_load = loaded.predict_proba_one(row)
    assert set(p_load) == set(vertigo.HIERARCHY.leaves)
    for leaf in p_mem:
        assert abs(p_mem[leaf] - p_load[leaf]) < 1e-9


def test_metrics_are_sane() -> None:
    _, m = build_and_save(
        __import__("tempfile").mkdtemp(), seed=20260711, n_samples=1200,
        params={"iterations": 100, "depth": 4},
    )
    # recuperación del generador: alta pero es sintético (no clínico)
    assert 0.0 <= m.leaf_accuracy <= 1.0
    assert m.gate_accuracy > 0.8
