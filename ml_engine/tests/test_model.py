"""TB1.3 — HierarchicalCatBoost: gate binario monótono (INV-9) + camino.

Modelo chico cacheado (menos filas/iteraciones) para que el gate sea rápido.
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
    """Subir CUALQUIER feature de riesgo (+1, ceteris paribus) NUNCA baja P(dangerous).

    Se testea al nivel EXACTO que CatBoost garantiza: la matriz codificada, un
    solo feature movido, el resto fijo. Es la garantía dura de INV-9 (gate,
    pre-abstención/pre-calibración).
    """
    model = _small_model()
    x = _grid_rows(25)
    risk = FEATURES.risk_features
    assert risk  # hay features de riesgo declaradas
    for i in range(len(x)):
        base = x.iloc[[i]].copy()
        p_base = model.gate_danger_proba_encoded(base)
        for rf in risk:
            bumped = base.copy()
            col = bumped.columns.get_loc(rf)
            bumped.iloc[0, col] = float(base.iloc[0][rf]) + 1.0
            p_up = model.gate_danger_proba_encoded(bumped)
            assert p_up >= p_base - 1e-6, (
                f"fila {i}: subir '{rf}' bajó P(dangerous) {p_base:.4f}→{p_up:.4f}"
            )


def test_INV9_holds_for_two_steps() -> None:
    """Monotonía acumulativa: +2 tampoco baja P(dangerous)."""
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
    """Sanidad: un caso claramente central concentra masa en la rama de peligro."""
    model = _small_model()
    central = {
        "timing_pattern": "acute_continuous", "head_impulse": "normal",
        "nystagmus_direction": "direction_changing", "skew_deviation": True,
        "focal_signs": ["dysarthria", "limb_weakness"], "truncal_ataxia_severe": True,
        "vascular_risk_factors": ["hypertension", "atrial_fibrillation"], "age_years": 72,
    }
    p = model.predict_proba_one(central)
    p_danger = p["central_suspected"] + p["cardiogenic_suspected"]
    assert p_danger > 0.5, f"P(peligro)={p_danger:.3f} debería dominar en caso central claro"


def test_clear_bppv_case_is_peripheral() -> None:
    """VPPB = patrón POSICIONAL + fatigable (dix_hallpike), NO torsional espontáneo.

    (Fix P0.5: el torsional/vertical espontáneo es CENTRAL — ver test de
    consistencia A↔B abajo; VPPB se representa por el patrón posicional.)
    """
    model = _small_model()
    bppv = {
        "trigger": "positional_head", "timing_pattern": "episodic_triggered",
        "duration": "under_1min", "onset": "sudden", "dix_hallpike": "right_positive",
        "nystagmus_fatigable": True, "nystagmus_latency_s": 5.0,
    }
    p = model.predict_proba_one(bppv)
    p_danger = p["central_suspected"] + p["cardiogenic_suspected"]
    assert p_danger < 0.3, f"P(peligro)={p_danger:.3f} no debería dominar en BPPV claro"


def test_P05_spontaneous_torsional_not_classified_as_benign_bppv() -> None:
    """Fix P0.5 — la propiedad de SEGURIDAD que importa.

    El bug original: el generador ponía torsional_pure en el perfil BPPV → B
    clasificaba un nistagmo ESPONTÁNEO puro torsional/vertical como **VPPB benigno
    posicional** (que sugeriría Epley), contradiciendo a A (red flag A2 → central).

    B es una capa PROBABILÍSTICA blanda; A es la autoridad de seguridad (regla
    dura, INV-1). No le exigimos a B replicar la regla dura, pero sí que NO
    etiquete el signo central como VPPB tratable. (La dirección de reconciliación
    hacia peligro está cubierta por el test-grilla INV-9, que ahora incluye
    `central_nystagmus_pattern` como feature de riesgo.)
    """
    model = _small_model()
    for direction in ("torsional_pure", "vertical_pure"):
        spont = {
            "trigger": "spontaneous", "timing_pattern": "acute_continuous",
            "onset": "sudden", "nystagmus_direction": direction,
        }
        p = model.predict_proba_one(spont)
        assert p["bppv_posterior"] < 0.15, (
            f"{direction} espontáneo → B lo llama VPPB posterior "
            f"({p['bppv_posterior']:.3f}); contradice a A y sugeriría Epley"
        )
        assert p["bppv_horizontal"] < 0.15


def test_P05_central_nystagmus_pattern_is_a_risk_feature() -> None:
    """El patrón de nistagmo central alimenta el gate de peligro (monótono)."""
    assert "central_nystagmus_pattern" in vertigo.FEATURES.risk_features
    # y NO aparece torsional/vertical puro en el perfil BPPV (se testea en synth)
