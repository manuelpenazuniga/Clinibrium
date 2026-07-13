"""`AuditEvent`: immutable audit event (INV-4).

Leaf: this model imports NOTHING from `clinibrium.*` except within the
`clinibrium.contracts` package itself.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from clinibrium.contracts.enums import ActorType, ForcedAction, Urgency


class AuditEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    occurred_at: datetime
    event_type: str  # e.g. "pipeline_evaluation"
    actor: ActorType = ActorType.system
    model_used: str | None = None
    input_features_hash: str  # hash of the structured features (NOT of PII)
    urgency: Urgency
    forced_actions: list[ForcedAction] = []
    red_flag_activa: bool
    outcome_summary: str
    reasoner_status: Literal["ok", "degraded"] = "ok"  # INV-8 marker
    outcome: str = "evaluation"
