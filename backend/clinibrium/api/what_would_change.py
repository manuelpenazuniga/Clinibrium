"""Endpoint `POST /api/what-would-change` — análisis contrafactual determinista.

Toma un `CaseFeatures` base y devuelve qué ÚNICO hallazgo cambiaría el manejo
(urgencia / acciones forzadas), verificado por el núcleo determinista
(RedFlagEngine + rails). El LLM NO participa (INV-3): son contrafactuales duros.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from clinibrium.contracts.features import CaseFeatures
from clinibrium.counterfactual import analyze

router = APIRouter()


@router.post("/api/what-would-change")
async def what_would_change(features: CaseFeatures) -> dict[str, Any]:
    """¿Qué único hallazgo cambiaría el manejo de este paciente?

    Body: `CaseFeatures` (Pydantic valida → 422 si inválido/extra).
    Response: `{base_urgency, counterfactuals[], minimal_escalation}`.
    """
    return analyze(features).to_dict()
