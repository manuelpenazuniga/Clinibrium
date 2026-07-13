"""``POST /predict`` service (Track B) — behind the FROZEN contract.

Returns EXACTLY ``{probabilities, shap, model_version}`` (the shape of
``clinibrium.contracts.PredictResponse``, verified by a black-box test without
importing A). Serves the real ``ml_engine`` engine; A's stub is untouched.

Guards (Codex fix #8/#9):
  - **Input** ``extra="forbid"``: rejects (422) any key outside the domain
    allowlist (privacy boundary — does not accept PII).
  - **Validated output**: keys ∈ domain vocabulary, finite values in [0,1],
    Σ≈1. On internal failure → **5xx** so A degrades (INV-6).
  - **The ML NEVER emits urgency** (INV-11): the response does not contain it.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from ml_engine.core.model import HierarchicalCatBoost
from ml_engine.core.spec import Domain
from ml_engine.domains import vertigo

_DEFAULT_ARTIFACTS = Path(__file__).parent.parent / "artifacts" / "model"

# Registry of servable domains (config, not A's code).
_DOMAIN: Domain = vertigo.VERTIGO

app = FastAPI(title="Clinibrium ml_engine (Track B)")

_cache: dict[str, HierarchicalCatBoost] = {}


def _artifacts_dir() -> Path:
    return Path(os.environ.get("ML_ARTIFACTS_DIR", str(_DEFAULT_ARTIFACTS)))


def get_model() -> HierarchicalCatBoost:
    key = str(_artifacts_dir())
    if key not in _cache:
        _cache[key] = HierarchicalCatBoost.load(_artifacts_dir(), _DOMAIN)
    return _cache[key]


def reset_model_cache() -> None:
    _cache.clear()


def _validate_output(probs: dict[str, float]) -> None:
    vocab = set(_DOMAIN.hierarchy.leaves) | {_DOMAIN.hierarchy.abstain_label}
    unknown = set(probs) - vocab
    if unknown:
        raise ValueError(f"unknown labels in the output: {unknown}")
    for k, v in probs.items():
        if not isinstance(v, float) or not math.isfinite(v) or not (0.0 <= v <= 1.0):
            raise ValueError(f"invalid probability for {k}: {v!r}")
    if abs(sum(probs.values()) - 1.0) > 1e-3:
        raise ValueError(f"probabilities do not sum to 1: Σ={sum(probs.values())}")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "ml_engine", "domain": _DOMAIN.name}


@app.post("/predict")
async def predict(request: Request) -> dict[str, Any]:
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="body must be a JSON object")

    # Privacy boundary: reject keys outside the allowlist (no PII).
    extra = set(body) - _DOMAIN.features.accepted_keys
    if extra:
        raise HTTPException(status_code=422, detail=f"keys outside the allowlist: {sorted(extra)}")

    try:
        model = get_model()
        # predict_case = calibrated + abstention (9 keys: 8 leaves + undetermined)
        probabilities: dict[str, float] = {k: float(v) for k, v in model.predict_case(body).items()}
        _validate_output(probabilities)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 — INV-6: internal failure ⇒ 5xx ⇒ A degrades
        raise HTTPException(
            status_code=503, detail=f"ml_engine degraded ({type(exc).__name__})"
        ) from exc

    # Local SHAP of the gate (TB1.6): optional explainability; on failure → None
    # (the explanation must NEVER take down the prediction).
    shap: dict[str, float] | None
    try:
        shap = model.explain_gate(body)
        for v in shap.values():
            if not math.isfinite(v):
                shap = None
                break
    except Exception:  # noqa: BLE001
        shap = None

    return {"probabilities": probabilities, "shap": shap, "model_version": model.model_version}
