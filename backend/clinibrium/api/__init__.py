"""FastAPI HTTP API — endpoints, (de)serialization; everything goes through the orchestrator."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from clinibrium.api.decision import router as decision_router
from clinibrium.api.evaluate import router as evaluate_router
from clinibrium.api.what_would_change import router as wwcm_router
from clinibrium.config import get_settings


def create_app() -> FastAPI:
    app = FastAPI(
        title="Clinibrium",
        version="0.1.0",
        description="Apoyo diagnóstico otoneurológico — el médico decide.",
    )

    origins = [o.strip() for o in get_settings().CORS_ORIGINS.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "clinibrium"}

    app.include_router(evaluate_router)
    app.include_router(decision_router)
    app.include_router(wwcm_router)

    return app


app = create_app()
