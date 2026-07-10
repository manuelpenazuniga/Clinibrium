"""`AuditEvent`: evento inmutable de auditoría (INV-4).

Hoja: este modelo NO importa nada de `clinibrium.*` salvo dentro del propio
paquete `clinibrium.contracts`.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from clinibrium.contracts.enums import ActorType, ForcedAction, Urgency


class AuditEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    occurred_at: datetime
    event_type: str  # p.ej. "pipeline_evaluation"
    actor: ActorType = ActorType.system
    model_used: str | None = None
    input_features_hash: str  # hash de las features estructuradas (NO del PII)
    urgency: Urgency
    forced_actions: list[ForcedAction] = []
    red_flag_activa: bool
    outcome_summary: str
