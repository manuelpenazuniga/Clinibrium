"""DifferentialEngine determinista — reglas ICVD, pool de candidatos.

Hoja del grafo `clinibrium.*` (INV-5): este paquete SOLO importa de
`clinibrium.contracts`. NO importa `redflag_engine`, `reasoner`,
`ml_client` ni `orchestrator`. La separación es ley: la urgencia y la
seguridad las sellan RedFlagEngine + rails, no este módulo.

API pública:
    evaluate(features) -> DifferentialResult
    CRITERIA           -> list[DiagnosisCriterion]  (tabla de datos)
"""
from __future__ import annotations

from clinibrium.differential_engine.criteria import CRITERIA, DiagnosisCriterion
from clinibrium.differential_engine.engine import evaluate

__all__ = ["CRITERIA", "DiagnosisCriterion", "evaluate"]
