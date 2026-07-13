"""`POST /api/what-would-change` endpoint — deterministic counterfactual analysis.

Takes a base `CaseFeatures` and returns which SINGLE finding would change the
management (urgency / forced actions), verified by the deterministic core
(RedFlagEngine + rails). The LLM does NOT participate (INV-3): these are hard
counterfactuals.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from clinibrium.contracts.features import CaseFeatures
from clinibrium.counterfactual import analyze

router = APIRouter()


@router.post("/api/what-would-change")
async def what_would_change(features: CaseFeatures) -> dict[str, Any]:
    """Which single finding would change the management of this patient?

    Body: `CaseFeatures` (Pydantic validates → 422 if invalid/extra fields).
    Response: `{base_urgency, counterfactuals[], minimal_escalation}`.
    """
    return analyze(features).to_dict()
