"""RedFlagEngine evaluation — pure, stateless, no I/O.

INV-5: this module ONLY imports `contracts` and the local `RULES` table. It
does NOT import `differential_engine`, `reasoner`, `ml_client` or
`orchestrator`. The verdict of `evaluate` cannot be overridden by anyone
downstream.

Evaluation invariants:
  - Same `CaseFeatures` ⇒ same `RedFlagResult` (pure and deterministic).
  - `red_flag_activa == True` iff some hit carries `DERIVAR_URGENTE`.
    `ESCALAR` / `PRECAUCION_EXAMEN` alone do NOT activate `red_flag_activa`,
    but they do end up in the result's `forced_actions`.
  - The order of `hits` follows the stable order of `RULES`.
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
    """Evaluate the red flags over `features`. Pure function.

    Walks `RULES` in stable order; for each rule whose predicate is
    true, appends a `RedFlagHit`. Returns:

      - `red_flag_activa` True if some hit carries `DERIVAR_URGENTE`.
      - `forced_actions` = union (set) of the forced actions of all hits.
      - `hits` in the same order as `RULES`.
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
