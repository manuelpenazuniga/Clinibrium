"""Stub server de `POST /predict` (SOLO dev/demo).

NO se monta en la app principal de Clinibrium (ver `clinibrium.api`).
Es una app FastAPI separada, pensada para validar el contrato
congelado y la ruta feliz del `ml_client` sin depender del servicio
ML real (persona 2).

Cómo correrlo:

    cd backend && source .venv/bin/activate
    uvicorn clinibrium.ml_client.stub_server:app --port 8001 --reload

Y en otra terminal, apuntar el cliente:

    ML_PREDICT_URL=http://localhost:8001 pytest -q

La probabilidad que devuelve es una distribución fija sesgada hacia
BPPV posterior — suficiente para probar la ruta feliz y los SHAP, no
para uso clínico.
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from clinibrium.contracts import CaseFeatures, Diagnosis, PredictResponse

app = FastAPI(
    title="Clinibrium ML stub (dev only)",
    version="0.1.0",
    description="Stub de POST /predict — NO usar en producción.",
)

STUB_MODEL_VERSION = "stub-v0.1-bppv-biased"

# Distribución fija plausible: BPPV posterior como hipótesis principal.
STUB_PROBABILITIES: dict[str, float] = {
    Diagnosis.bppv_posterior.value: 0.62,
    Diagnosis.bppv_horizontal.value: 0.08,
    Diagnosis.vestibular_neuritis.value: 0.07,
    Diagnosis.vestibular_migraine.value: 0.06,
    Diagnosis.meniere.value: 0.05,
    Diagnosis.labyrinthitis.value: 0.04,
    Diagnosis.central_suspected.value: 0.03,
    Diagnosis.cardiogenic_suspected.value: 0.02,
    Diagnosis.undetermined.value: 0.03,
}

STUB_SHAP: dict[str, float] = {
    "dix_hallpike": 0.41,
    "nystagmus_mixed": 0.28,
    "nystagmus_latency_s": 0.12,
    "nystagmus_duration_s": 0.07,
    "trigger_positional_head": 0.05,
    "head_impulse_normal": -0.03,
    "hearing_loss_none": -0.02,
}


class _PredictRequest(BaseModel):
    """Acepta el body como dict plano (lo que el cliente envía con
    `model_dump(mode="json")`). Lo declaramos como `dict` para no
    re-validar dos veces el `CaseFeatures` (la fuente de verdad es el
    cliente)."""

    model_config = {"extra": "allow"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "clinibrium-ml-stub"}


@app.post("/predict", response_model=PredictResponse)
async def predict(_req: _PredictRequest) -> PredictResponse:
    """Devuelve probabilidades fijas plausibles (BPPV-biased)."""
    return PredictResponse(
        probabilities=STUB_PROBABILITIES,
        shap=STUB_SHAP,
        model_version=STUB_MODEL_VERSION,
    )


def _typecheck_only() -> None:
    """Helper para import-time: confirma que el shape de `CaseFeatures`
    es compatible con el stub (no se ejecuta, solo el anotado)."""
    _ = CaseFeatures  # noqa: F841
