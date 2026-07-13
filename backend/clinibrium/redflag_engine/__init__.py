"""Deterministic RedFlagEngine — is it an emergency? (separate due to regulatory regime).

Public API:
  - `evaluate(features)` → `RedFlagResult`
  - `RULES` (rule table, editable by the validating clinician)
  - `RedFlagRule`, `AGE_CENTRAL_THRESHOLD`

INV-5: this package ONLY imports `contracts`. Never `differential_engine`,
`reasoner`, `ml_client` or `orchestrator`. Its verdict cannot be
overridden by anyone downstream.
"""
from __future__ import annotations

from clinibrium.redflag_engine.engine import evaluate
from clinibrium.redflag_engine.rules import (
    AGE_CENTRAL_THRESHOLD,
    RULES,
    RedFlagRule,
)

__all__ = [
    "AGE_CENTRAL_THRESHOLD",
    "RULES",
    "RedFlagRule",
    "evaluate",
]
