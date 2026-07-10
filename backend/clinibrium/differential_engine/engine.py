"""DifferentialEngine — scoring determinista y puro.

Toma `CaseFeatures` y devuelve un `DifferentialResult` con candidatos
ordenados desc por score (0..1). Es una función PURA:

  - mismas features ⇒ mismo resultado (mismas listas, mismo orden),
  - sin I/O, sin reloj, sin LLM, sin random,
  - sin acoplamiento a `redflag_engine` / `reasoner` / `ml_client` /
    `orchestrator` (INV-5: separación regulatoria),
  - no detecta red flags, no fija urgencia y no recomienda tratamiento
    (eso es RedFlagEngine + rails + reasoner; INV-1, INV-3).
"""
from __future__ import annotations

from clinibrium.contracts.enums import Diagnosis
from clinibrium.contracts.features import CaseFeatures
from clinibrium.contracts.results import DifferentialCandidate, DifferentialResult
from clinibrium.differential_engine.criteria import CRITERIA, DiagnosisCriterion

# Precomputado en import-time: `Diagnosis → peso total posible`. Es
# determinista (deriva 1:1 de CRITERIA) y deja la evaluación como un
# solo pass lineal por diagnóstico.
_MAX_POSSIBLE_BY_DIAGNOSIS: dict[Diagnosis, float] = {}


def _build_max_possible() -> dict[Diagnosis, float]:
    out: dict[Diagnosis, float] = {}
    for c in CRITERIA:
        out[c.diagnosis] = out.get(c.diagnosis, 0.0) + c.weight
    return out


_MAX_POSSIBLE_BY_DIAGNOSIS = _build_max_possible()


def _score_one(
    diagnosis: Diagnosis,
    features: CaseFeatures,
    criteria: list[DiagnosisCriterion],
) -> DifferentialCandidate | None:
    """Suma los pesos de los criterios matcheados y normaliza por el
    total posible para ese diagnóstico. Devuelve `None` si nada matchea
    o si el diagnóstico no tiene criterios definidos.
    """
    max_possible = _MAX_POSSIBLE_BY_DIAGNOSIS.get(diagnosis, 0.0)
    if max_possible <= 0.0:
        return None

    raw = 0.0
    rule_ids: list[str] = []
    for c in criteria:
        if c.diagnosis != diagnosis:
            continue
        if c.predicate(features):
            raw += c.weight
            rule_ids.append(c.id)

    if raw <= 0.0:
        return None

    return DifferentialCandidate(
        diagnosis=diagnosis,
        score=raw / max_possible,
        rule_ids=rule_ids,
    )


def evaluate(features: CaseFeatures) -> DifferentialResult:
    """Evalúa `features` y devuelve el pool priorizado de diagnósticos
    diferenciales. Mismas features ⇒ mismo resultado, siempre.

    Convenciones:
      - Itera `Diagnosis` en su orden de definición (estable) y aplica
        `sorted(..., key=-score)` para el orden descendente por score.
        Python `sorted` es estable ⇒ empate de score se desempata por
        el orden de inserción, que es el orden de enum.
      - Excluye candidatos con `score == 0` (ningún criterio matcheó).
    """
    candidates: list[DifferentialCandidate] = []
    for dx in Diagnosis:
        cand = _score_one(dx, features, CRITERIA)
        if cand is not None:
            candidates.append(cand)

    candidates.sort(key=lambda c: -c.score)
    return DifferentialResult(candidates=candidates)
