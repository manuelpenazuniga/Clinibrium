"""Shared Pydantic models (de-identified features, results, AuditEvent).

Leaf of the `clinibrium.*` graph: this package imports NOTHING from other
`clinibrium` submodules. Any dependency between internal submodules of
`clinibrium.contracts` is valid.
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
