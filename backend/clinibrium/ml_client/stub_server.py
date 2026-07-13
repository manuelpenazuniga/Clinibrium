"""Stub server for `POST /predict` (dev/demo ONLY).

NOT mounted on the main Clinibrium app (see `clinibrium.api`).
It is a separate FastAPI app, meant to validate the frozen contract
and the happy path of `ml_client` without depending on the real ML
service (person 2).

How to run it:

    cd backend && source .venv/bin/activate
    uvicorn clinibrium.ml_client.stub_server:app --port 8001 --reload

And in another terminal, point the client:

    ML_PREDICT_URL=http://localhost:8001 pytest -q

The probability it returns is a fixed distribution biased towards
posterior BPPV — enough to test the happy path and SHAP values, not
for clinical use.
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from clinibrium.contracts import CaseFeatures, Diagnosis, PredictResponse

app = FastAPI(
    title="Clinibrium ML stub (dev only)",
    version="0.1.0",
    description="POST /predict stub — do NOT use in production.",
)

STUB_MODEL_VERSION = "stub-v0.1-bppv-biased"

# Plausible fixed distribution: posterior BPPV as the leading hypothesis.
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
    """Accepts the body as a plain dict (what the client sends with
    `model_dump(mode="json")`). Declared as `dict` to avoid re-validating
    `CaseFeatures` twice (the source of truth is the client)."""

    model_config = {"extra": "allow"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "clinibrium-ml-stub"}


@app.post("/predict", response_model=PredictResponse)
async def predict(_req: _PredictRequest) -> PredictResponse:
    """Returns plausible fixed probabilities (BPPV-biased)."""
    return PredictResponse(
        probabilities=STUB_PROBABILITIES,
        shap=STUB_SHAP,
        model_version=STUB_MODEL_VERSION,
    )


def _typecheck_only() -> None:
    """Import-time helper: confirms that the `CaseFeatures` shape is
    compatible with the stub (not executed, only the annotation)."""
    _ = CaseFeatures  # noqa: F841
