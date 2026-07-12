"""Motor contrafactual determinista (What Would Change My Mind?).

Para un caso base, aplica perturbaciones de UNA sola variable (allowlisted,
clínicamente revisadas) y corre cada variante por el núcleo DETERMINISTA
(RedFlagEngine + DifferentialEngine + rails). Devuelve los cambios mínimos que
alteran la urgencia o las acciones forzadas, con el riel que disparó.

Principio rector (INV-3): el LLM NO decide qué es urgente. El core determinista
sella la urgencia de cada contrafactual; Claude (aguas arriba) solo los explica.

Grafo de imports: counterfactual → engines + rails + contracts (como el
orchestrator). NO importa reasoner/ml/audit (es análisis puro, sin efectos).
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

# Orden de severidad para rankear escalamientos.
_URGENCY_RANK: dict[Urgency, int] = {
    Urgency.ambulatoria: 0,
    Urgency.prioritaria: 1,
    Urgency.inmediata: 2,
}


@dataclass
class _Perturbation:
    field: str
    value: Any
    label: str  # descripción humana del cambio de UNA variable


# Perturbaciones de UNA sola variable, clínicamente significativas (mapean a rieles).
# TODO(clinical): lista provisional a confirmar/expandir con el especialista (T-CLIN r2).
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
    change: str  # descripción humana
    base_urgency: str
    new_urgency: str
    urgency_changed: bool
    escalates: bool  # new_urgency más urgente que base
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
        """El contrafactual escalante de MENOR urgencia final (el 'mínimo cambio')."""
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
    """Corre SOLO el núcleo determinista (redflag → differential → rails)."""
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
    """True si la perturbación no cambia nada (el caso ya tiene ese valor)."""
    current = _current_value(features, pert)
    if isinstance(pert.value, set):
        # focal_signs: no-op si el signo ya está presente
        return bool(pert.value) and pert.value.issubset(current or set())
    return current == pert.value


def analyze(features: CaseFeatures) -> WhatWouldChangeResult:
    """Análisis contrafactual: qué único hallazgo cambia el manejo."""
    base_urgency, base_actions, _ = _deterministic_seal(features)
    base_rank = _URGENCY_RANK[base_urgency]

    out: list[Counterfactual] = []
    for pert in _PERTURBATIONS:
        if _is_noop(features, pert):
            continue
        # cambio de UNA sola variable (fix P1.1: exactamente una feature)
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

    # ordenar por urgencia final desc (los que escalan a inmediata primero)
    out.sort(key=lambda c: _URGENCY_RANK[Urgency(c.new_urgency)], reverse=True)
    return WhatWouldChangeResult(base_urgency=base_urgency.value, counterfactuals=out)


def analyze_from_mapping(payload: Mapping[str, Any]) -> WhatWouldChangeResult:
    """Helper: valida el payload como CaseFeatures y analiza."""
    return analyze(CaseFeatures.model_validate(payload))
