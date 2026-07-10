"""Contratos de salida del pipeline (red flag, diferencial, ML, reasoner).

Hoja: SIN lógica, SIN cómputo — solo formas de datos. Ningún import fuera
del paquete `clinibrium.contracts`.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from clinibrium.contracts.enums import Diagnosis, ForcedAction, Urgency


class RedFlagHit(BaseModel):
    id: str
    label: str
    forced_actions: list[ForcedAction]
    severity: Literal["high", "medium"]


class RedFlagResult(BaseModel):
    red_flag_activa: bool
    hits: list[RedFlagHit] = []
    forced_actions: set[ForcedAction] = set()


class DifferentialCandidate(BaseModel):
    diagnosis: Diagnosis
    score: float  # rango [0.0, 1.0]; convención: ordenadas desc por score
    rule_ids: list[str] = []


class DifferentialResult(BaseModel):
    # Ordenadas desc por `score` (convención del productor; el validador no
    # re-ordena).
    candidates: list[DifferentialCandidate] = []


class PredictResponse(BaseModel):
    """Contrato del endpoint `POST /predict` (track B — ML opcional)."""

    probabilities: dict[str, float]
    shap: dict[str, float] | None = None
    model_version: str


class ReasonerOutput(BaseModel):
    """Salida del razonador (Claude).

    El reasoner EXPLICA y CONCILIA; NO fija `urgency` ni `diagnosis`
    vinculante — esos los sellan RedFlagEngine + rails (INV-1).
    """

    explanation: str
    reconciliation: str
    suggested_next_steps: list[str] = []
    model_used: str
    reasoner_suggested_urgency: Urgency | None = None  # AD-11: estructurada (enum), no texto


class PipelineResult(BaseModel):
    case_id: str
    urgency: Urgency
    red_flag: RedFlagResult
    differential: DifferentialResult
    ml: PredictResponse | None = None
    reasoning: ReasonerOutput | None = None
    forced_actions: set[ForcedAction] = set()
    applied_rails: list[str] = []
    audit_event_id: str | None = None
