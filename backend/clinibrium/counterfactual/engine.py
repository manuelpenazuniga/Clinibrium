"""Deterministic counterfactual engine (What Would Change My Mind?).

For a base case, applies SINGLE-variable perturbations (allowlisted,
clinically reviewed) and runs each variant through the DETERMINISTIC core
(RedFlagEngine + DifferentialEngine + rails). Returns the minimal changes that
alter the urgency or the forced actions, with the rail that fired.

Guiding principle (INV-3): the LLM does NOT decide what is urgent. The
deterministic core seals the urgency of each counterfactual; Claude
(upstream) only explains them.

Import graph: counterfactual → engines + rails + contracts (like the
orchestrator). Does NOT import reasoner/ml/audit (pure analysis, no effects).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from clinibrium.contracts.enums import (
    FocalSign,
    HearingLoss,
    NystagmusDirection,
    Urgency,
)
from clinibrium.contracts.features import CaseFeatures
from clinibrium.contracts.results import PipelineResult
from clinibrium.differential_engine import evaluate as differential_evaluate
from clinibrium.rails import apply_rails
from clinibrium.redflag_engine import evaluate as redflag_evaluate

# Severity order used to rank escalations.
_URGENCY_RANK: dict[Urgency, int] = {
    Urgency.ambulatoria: 0,
    Urgency.prioritaria: 1,
    Urgency.inmediata: 2,
}


@dataclass
class _Perturbation:
    field: str
    value: Any
    label: str  # human-readable description of the SINGLE-variable change (UI-facing, Spanish)


# SINGLE-variable perturbations, clinically meaningful (they map to rails).
# Labels are returned through the API and shown to the clinician — keep Spanish.
# TODO(clinical): provisional list to confirm/expand with the specialist (T-CLIN r2).
_PERTURBATIONS: tuple[_Perturbation, ...] = (
    _Perturbation("focal_signs", {FocalSign.diplopia}, "Nuevo signo focal: diplopía"),
    _Perturbation("focal_signs", {FocalSign.dysarthria}, "Nuevo signo focal: disartria"),
    _Perturbation("skew_deviation", True, "Skew deviation presente"),
    _Perturbation("truncal_ataxia_severe", True, "Ataxia troncal severa (no puede caminar)"),
    _Perturbation(
        "nystagmus_direction",
        NystagmusDirection.direction_changing,
        "Nistagmo que cambia de dirección con la mirada",
    ),
    _Perturbation(
        "nystagmus_direction",
        NystagmusDirection.torsional_pure,
        "Nistagmo espontáneo puro torsional",
    ),
    _Perturbation(
        "headache_neck_pain_sudden_severe",
        True,
        "Cefalea/cervicalgia súbita e intensa",
    ),
    _Perturbation(
        "hearing_loss",
        HearingLoss.sudden_unilateral,
        "Hipoacusia súbita unilateral",
    ),
    _Perturbation("altered_consciousness", True, "Compromiso de conciencia"),
    _Perturbation("presyncope_syncope", True, "Presíncope o síncope"),
    _Perturbation("neck_stiffness", True, "Rigidez de nuca"),
    _Perturbation("recent_head_neck_trauma", True, "Trauma craneal/cervical reciente"),
)


@dataclass
class Counterfactual:
    feature: str
    change: str  # human-readable description
    base_urgency: str
    new_urgency: str
    urgency_changed: bool
    escalates: bool  # new_urgency more urgent than base
    forced_actions_added: list[str]
    rails_fired: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "change": self.change,
            "base_urgency": self.base_urgency,
            "new_urgency": self.new_urgency,
            "urgency_changed": self.urgency_changed,
            "escalates": self.escalates,
            "forced_actions_added": self.forced_actions_added,
            "rails_fired": self.rails_fired,
        }


@dataclass
class WhatWouldChangeResult:
    base_urgency: str
    counterfactuals: list[Counterfactual] = field(default_factory=list)

    @property
    def minimal_escalation(self) -> Counterfactual | None:
        """The escalating counterfactual with the LOWEST final urgency (the 'minimal change')."""
        escalating = [c for c in self.counterfactuals if c.escalates]
        if not escalating:
            return None
        return min(escalating, key=lambda c: _URGENCY_RANK[Urgency(c.new_urgency)])

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_urgency": self.base_urgency,
            "counterfactuals": [c.to_dict() for c in self.counterfactuals],
            "minimal_escalation": (
                self.minimal_escalation.to_dict() if self.minimal_escalation else None
            ),
        }


def _deterministic_seal(
    features: CaseFeatures,
) -> tuple[Urgency, set[str], list[str]]:
    """Runs ONLY the deterministic core (redflag → differential → rails)."""
    red_flag = redflag_evaluate(features)
    differential = differential_evaluate(features)
    prelim = PipelineResult(
        case_id="cf",
        urgency=Urgency.ambulatoria,
        red_flag=red_flag,
        differential=differential,
        forced_actions=set(),
        applied_rails=[],
    )
    sealed = apply_rails(prelim, features)
    forced = {a.value for a in sealed.forced_actions}
    return sealed.urgency, forced, list(sealed.applied_rails)


def _current_value(features: CaseFeatures, pert: _Perturbation) -> Any:
    return getattr(features, pert.field)


def _is_noop(features: CaseFeatures, pert: _Perturbation) -> bool:
    """True if the perturbation changes nothing (the case already has that value)."""
    current = _current_value(features, pert)
    if isinstance(pert.value, set):
        # focal_signs: no-op if the sign is already present
        return bool(pert.value) and pert.value.issubset(current or set())
    return current == pert.value


def analyze(features: CaseFeatures) -> WhatWouldChangeResult:
    """Counterfactual analysis: which single finding changes the management."""
    base_urgency, base_actions, _ = _deterministic_seal(features)
    base_rank = _URGENCY_RANK[base_urgency]

    out: list[Counterfactual] = []
    for pert in _PERTURBATIONS:
        if _is_noop(features, pert):
            continue
        # SINGLE-variable change (fix P1.1: exactly one feature)
        cf_features = features.model_copy(update={pert.field: pert.value})
        new_urgency, new_actions, rails = _deterministic_seal(cf_features)
        actions_added = sorted(new_actions - base_actions)
        urgency_changed = new_urgency != base_urgency
        if not urgency_changed and not actions_added:
            continue
        out.append(
            Counterfactual(
                feature=pert.field,
                change=pert.label,
                base_urgency=base_urgency.value,
                new_urgency=new_urgency.value,
                urgency_changed=urgency_changed,
                escalates=_URGENCY_RANK[new_urgency] > base_rank,
                forced_actions_added=actions_added,
                rails_fired=rails,
            )
        )

    # sort by final urgency desc (those escalating to inmediata first)
    out.sort(key=lambda c: _URGENCY_RANK[Urgency(c.new_urgency)], reverse=True)
    return WhatWouldChangeResult(base_urgency=base_urgency.value, counterfactuals=out)


def analyze_from_mapping(payload: Mapping[str, Any]) -> WhatWouldChangeResult:
    """Helper: validates the payload as CaseFeatures and analyzes it."""
    return analyze(CaseFeatures.model_validate(payload))
