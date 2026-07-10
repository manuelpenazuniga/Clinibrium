"""Evaluación del RedFlagEngine — pura, sin estado, sin I/O.

INV-5: este módulo SOLO importa `contracts` y la tabla `RULES` local. NO
importa `differential_engine`, `reasoner`, `ml_client` ni `orchestrator`.
El veredicto de `evaluate` no puede ser anulado por nadie aguas abajo.

Invariantes de la evaluación:
  - Mismo `CaseFeatures` ⇒ mismo `RedFlagResult` (puro y determinista).
  - `red_flag_activa == True` ssi algún hit trae `DERIVAR_URGENTE`.
    `ESCALAR` / `PRECAUCION_EXAMEN` solos NO activan `red_flag_activa`,
    pero sí quedan en `forced_actions` del resultado.
  - El orden de `hits` sigue el orden estable de `RULES`.
"""
from __future__ import annotations

from clinibrium.contracts.enums import ForcedAction
from clinibrium.contracts.features import CaseFeatures
from clinibrium.contracts.results import RedFlagHit, RedFlagResult
from clinibrium.redflag_engine.rules import RULES, RedFlagRule


def _hit_for(rule: RedFlagRule) -> RedFlagHit:
    return RedFlagHit(
        id=rule.id,
        label=rule.label,
        forced_actions=list(rule.forced_actions),
        severity=rule.severity,
    )


def evaluate(features: CaseFeatures) -> RedFlagResult:
    """Evalúa las red flags sobre `features`. Función pura.

    Recorre `RULES` en orden estable; para cada regla cuyo predicado sea
    verdadero, agrega un `RedFlagHit`. Devuelve:

      - `red_flag_activa` True si algún hit trae `DERIVAR_URGENTE`.
      - `forced_actions` = unión (set) de las acciones forzadas de todos los hits.
      - `hits` en el mismo orden de `RULES`.
    """
    hits: list[RedFlagHit] = []
    forced: set[ForcedAction] = set()

    for rule in RULES:
        if not rule.predicate(features):
            continue
        hit = _hit_for(rule)
        hits.append(hit)
        forced.update(hit.forced_actions)

    red_flag_activa = ForcedAction.DERIVAR_URGENTE in forced

    return RedFlagResult(
        red_flag_activa=red_flag_activa,
        hits=hits,
        forced_actions=forced,
    )


__all__ = ["evaluate"]
