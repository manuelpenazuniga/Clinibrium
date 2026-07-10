"""Modelos Pydantic compartidos (features desidentificadas, resultados, AuditEvent).

Hoja del grafo `clinibrium.*`: este paquete NO importa nada de otros
submódulos de `clinibrium`. Cualquier dependencia entre submódulos internos
de `clinibrium.contracts` es válida.
"""
from __future__ import annotations

from clinibrium.contracts.audit import AuditEvent
from clinibrium.contracts.enums import (
    ActorType,
    Diagnosis,
    DixHallpikeResult,
    FocalSign,
    ForcedAction,
    HeadImpulse,
    HearingLoss,
    NystagmusDirection,
    Onset,
    SymptomDuration,
    TimingPattern,
    Trigger,
    Urgency,
    VascularRiskFactor,
)
from clinibrium.contracts.features import NETWORK_SAFE_FIELDS, CaseFeatures
from clinibrium.contracts.results import (
    DifferentialCandidate,
    DifferentialResult,
    PipelineResult,
    PredictResponse,
    ReasonerOutput,
    RedFlagHit,
    RedFlagResult,
)

__all__ = [
    "NETWORK_SAFE_FIELDS",
    "ActorType",
    "AuditEvent",
    "CaseFeatures",
    "Diagnosis",
    "DifferentialCandidate",
    "DifferentialResult",
    "DixHallpikeResult",
    "FocalSign",
    "ForcedAction",
    "HeadImpulse",
    "HearingLoss",
    "NystagmusDirection",
    "Onset",
    "PipelineResult",
    "PredictResponse",
    "ReasonerOutput",
    "RedFlagHit",
    "RedFlagResult",
    "SymptomDuration",
    "TimingPattern",
    "Trigger",
    "Urgency",
    "VascularRiskFactor",
]
