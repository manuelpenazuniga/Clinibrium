"""Deterministic DifferentialEngine — ICVD rules, candidate pool.

Leaf of the `clinibrium.*` graph (INV-5): this package ONLY imports from
`clinibrium.contracts`. It does NOT import `redflag_engine`, `reasoner`,
`ml_client` or `orchestrator`. The separation is law: urgency and
safety are sealed by RedFlagEngine + rails, not by this module.

Public API:
    evaluate(features) -> DifferentialResult
    CRITERIA           -> list[DiagnosisCriterion]  (data table)
"""
from __future__ import annotations

from clinibrium.differential_engine.criteria import CRITERIA, DiagnosisCriterion
from clinibrium.differential_engine.engine import evaluate

__all__ = ["CRITERIA", "DiagnosisCriterion", "evaluate"]
