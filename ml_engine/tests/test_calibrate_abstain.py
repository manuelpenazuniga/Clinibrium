"""TB1.4 (calibración + ECE) + TB1.5 (abstención)."""
import math

from ml_engine.core.abstain import ConfidenceGate
from ml_engine.core.calibrate import TemperatureCalibrator, ece

LEAVES = ("a", "b", "c")


def _p(a: float, b: float, c: float) -> dict[str, float]:
    return {"a": a, "b": b, "c": c}


# --- Calibración ----------------------------------------------------------

def test_calibrator_output_is_a_distribution_and_preserves_argmax() -> None:
    probs = [_p(0.7, 0.2, 0.1), _p(0.1, 0.8, 0.1), _p(0.2, 0.2, 0.6)]
    y = ["a", "b", "c"]
    cal = TemperatureCalibrator.fit(probs, y, LEAVES)
    assert cal.temperature > 0
    for p in probs:
        out = cal.transform_one(p)
        assert abs(sum(out.values()) - 1.0) < 1e-9
        # temperature scaling preserva el argmax
        assert max(out, key=out.__getitem__) == max(p, key=p.__getitem__)


def test_ece_in_range() -> None:
    probs = [_p(0.9, 0.05, 0.05), _p(0.4, 0.4, 0.2), _p(0.34, 0.33, 0.33)]
    y = ["a", "b", "c"]
    val = ece(probs, y, LEAVES, n_bins=5)
    assert 0.0 <= val <= 1.0 and math.isfinite(val)


def test_temperature_softens_overconfident_wrong_predictions() -> None:
    # modelo sobre-confiado y EQUIVOCADO → T>1 (ablanda)
    probs = [_p(0.98, 0.01, 0.01) for _ in range(50)]
    y = ["b"] * 50  # siempre se equivoca con altísima confianza
    cal = TemperatureCalibrator.fit(probs, y, LEAVES)
    assert cal.temperature > 1.0


# --- Abstención -----------------------------------------------------------

def test_gate_abstains_below_threshold() -> None:
    gate = ConfidenceGate(threshold=0.5)
    low = _p(0.34, 0.33, 0.33)  # max 0.34 < 0.5 → abstiene
    out = gate.apply(low)
    assert out["undetermined"] == 1.0
    assert sum(out.values()) == 1.0
    assert all(out[k] == 0.0 for k in LEAVES)


def test_gate_passes_confident() -> None:
    gate = ConfidenceGate(threshold=0.5)
    hi = _p(0.8, 0.1, 0.1)  # max 0.8 ≥ 0.5 → pasa
    out = gate.apply(hi)
    assert out["undetermined"] == 0.0
    assert abs(sum(out.values()) - 1.0) < 1e-9
    assert out["a"] == 0.8


def test_gate_fit_targets_coverage() -> None:
    # confidencias uniformes 0..1 → cobertura 0.9 ⇒ τ ≈ percentil 10
    probs = [_p(c, (1 - c) / 2, (1 - c) / 2) for c in [i / 100 for i in range(1, 101)]]
    gate = ConfidenceGate.fit(probs, target_coverage=0.90)
    abst = sum(gate.abstains(p) for p in probs) / len(probs)
    assert 0.05 <= abst <= 0.15  # ~10% abstiene
