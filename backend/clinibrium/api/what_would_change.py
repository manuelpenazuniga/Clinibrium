"""`POST /api/what-would-change` endpoint — deterministic counterfactual analysis.

Takes a base `CaseFeatures` and returns which SINGLE finding would change the
management (urgency / forced actions), verified by the deterministic core
(RedFlagEngine + rails). The LLM does NOT participate (INV-3): these are hard
counterfactuals.
"""
from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Query

from clinibrium.contracts.features import CaseFeatures
from clinibrium.counterfactual import analyze
from clinibrium.i18n import localize_counterfactual_change

router = APIRouter()


def _localize_counterfactuals(payload: dict[str, Any], lang: str) -> dict[str, Any]:
    """Swap each counterfactual `change` for `lang` in place (Spanish = no-op).

    PRESENTATION ONLY, keyed by the stable `change_key`. Enums (`base_urgency`,
    `new_urgency`, `rails_fired`, `forced_actions_added`) are untouched — this
    is a fully deterministic (INV-3) endpoint and localization changes only the
    displayed prose, never the analysis.
    """
    if lang != "en":
        return payload

    def _swap(cf: Any) -> None:
        if isinstance(cf, dict) and "change" in cf:
            cf["change"] = localize_counterfactual_change(
                cf.get("change_key", ""), cf["change"], "en"
            )

    for cf in payload.get("counterfactuals", []):
        _swap(cf)
    _swap(payload.get("minimal_escalation"))
    return payload


@router.post("/api/what-would-change")
async def what_would_change(
    features: CaseFeatures,
    lang: Literal["es", "en"] = Query("es"),
) -> dict[str, Any]:
    """Which single finding would change the management of this patient?

    Body: `CaseFeatures` (Pydantic validates → 422 if invalid/extra fields).
    Query: `lang` ("es" default | "en") localizes the `change` labels ONLY —
    the deterministic analysis (RedFlagEngine + rails, INV-3) is identical in
    both languages. Response: `{base_urgency, counterfactuals[], minimal_escalation}`.
    """
    return _localize_counterfactuals(analyze(features).to_dict(), lang)
