"""TB1.7 + TB1.10a — servicio /predict validado + contrato black-box (sin importar A)."""
import json
import os

import pytest
from fastapi.testclient import TestClient

from ml_engine.service.app import app, reset_model_cache
from ml_engine.train import build_and_save

# Shape del contrato CONGELADO (espejo LITERAL de clinibrium.contracts.PredictResponse,
# SIN importar A — AD-16). Si A cambiara el contrato, este literal debe actualizarse.
_FROZEN_CONTRACT_KEYS = {"probabilities", "shap", "model_version"}


@pytest.fixture(scope="module")
def client(tmp_path_factory) -> TestClient:
    art = tmp_path_factory.mktemp("model")
    build_and_save(art, seed=20260711, n_samples=1200, params={"iterations": 100, "depth": 4})
    os.environ["ML_ARTIFACTS_DIR"] = str(art)
    reset_model_cache()
    return TestClient(app)


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["service"] == "ml_engine"


def test_predict_matches_frozen_contract_shape(client: TestClient) -> None:
    body = {"trigger": "positional_head", "duration": "under_1min",
            "timing_pattern": "episodic_triggered", "dix_hallpike": "right_positive"}
    r = client.post("/predict", json=body)
    assert r.status_code == 200
    data = r.json()
    assert set(data) == _FROZEN_CONTRACT_KEYS
    assert isinstance(data["probabilities"], dict)
    assert abs(sum(data["probabilities"].values()) - 1.0) < 1e-3
    # shap: None o dict feature→float (TB1.6, opcional por contrato)
    assert data["shap"] is None or (
        isinstance(data["shap"], dict)
        and all(isinstance(v, float) for v in data["shap"].values())
    )
    assert data["model_version"].startswith("synthetic-")


def test_INV11_response_has_no_urgency(client: TestClient) -> None:
    r = client.post("/predict", json={"trigger": "spontaneous"})
    assert r.status_code == 200
    assert "urgency" not in json.dumps(r.json()).lower()
    assert "urgencia" not in json.dumps(r.json()).lower()


def test_extra_keys_rejected_privacy_frontier(client: TestClient) -> None:
    """Una clave fuera del allowlist (p.ej. PII) → 422 (no la acepta)."""
    r = client.post("/predict", json={"trigger": "spontaneous", "patient_name": "Juan Pérez"})
    assert r.status_code == 422
    assert "patient_name" in r.json()["detail"]


def test_realistic_central_case(client: TestClient) -> None:
    body = {
        "timing_pattern": "acute_continuous", "head_impulse": "normal",
        "nystagmus_direction": "direction_changing", "skew_deviation": True,
        "focal_signs": ["dysarthria"], "truncal_ataxia_severe": True,
        "vascular_risk_factors": ["hypertension", "atrial_fibrillation"], "age_years": 72,
    }
    r = client.post("/predict", json=body)
    assert r.status_code == 200
    probs = r.json()["probabilities"]
    # las 9 claves del vocabulario (8 hojas + undetermined)
    assert "undetermined" in probs and len(probs) == 9
    p_danger = probs["central_suspected"] + probs["cardiogenic_suspected"]
    assert p_danger > 0.5
