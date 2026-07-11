"""Servicio ``POST /predict`` (Track B) — detrás del contrato CONGELADO.

Devuelve EXACTAMENTE ``{probabilities, shap, model_version}`` (el shape de
``clinibrium.contracts.PredictResponse``, verificado por un test black-box sin
importar A). Sirve el motor real de ``ml_engine``; el stub de A no se toca.

Guardas (fix Codex #8/#9):
  - **Input** ``extra="forbid"``: rechaza (422) cualquier clave fuera del
    allowlist del dominio (frontera de privacidad — no acepta PII).
  - **Output validado**: claves ∈ vocabulario del dominio, valores finitos en
    [0,1], Σ≈1. Ante fallo interno → **5xx** para que A degrade (INV-6).
  - **El ML NUNCA emite urgencia** (INV-11): la respuesta no la contiene.
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

# Registro de dominios servibles (config, no código de A).
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
        raise ValueError(f"labels desconocidos en la salida: {unknown}")
    for k, v in probs.items():
        if not isinstance(v, float) or not math.isfinite(v) or not (0.0 <= v <= 1.0):
            raise ValueError(f"probabilidad inválida para {k}: {v!r}")
    if abs(sum(probs.values()) - 1.0) > 1e-3:
        raise ValueError(f"las probabilidades no suman 1: Σ={sum(probs.values())}")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "ml_engine", "domain": _DOMAIN.name}


@app.post("/predict")
async def predict(request: Request) -> dict[str, Any]:
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="el body debe ser un objeto JSON")

    # Frontera de privacidad: rechazar claves fuera del allowlist (no PII).
    extra = set(body) - _DOMAIN.features.accepted_keys
    if extra:
        raise HTTPException(status_code=422, detail=f"claves fuera del allowlist: {sorted(extra)}")

    try:
        model = get_model()
        # predict_case = calibrado + abstención (9 claves: 8 hojas + undetermined)
        probabilities: dict[str, float] = {k: float(v) for k, v in model.predict_case(body).items()}
        _validate_output(probabilities)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 — INV-6: fallo interno ⇒ 5xx ⇒ A degrada
        raise HTTPException(
            status_code=503, detail=f"ml_engine degradado ({type(exc).__name__})"
        ) from exc

    return {"probabilities": probabilities, "shap": None, "model_version": model.model_version}
