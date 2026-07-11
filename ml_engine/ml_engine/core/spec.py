"""Tipos AGNÓSTICOS de dominio para la capa ML (la frontera de plataforma).

El core NO conoce vértigo. Un dominio se define (en ``ml_engine.domains.*``)
escribiendo tres objetos de config:

  - ``FeatureSpec``   — qué features existen, sus tipos, qué transformadores
    derivados PUROS provee el dominio, y cuáles features numéricas son "de
    riesgo" (dirección monótona en el gate de peligro).
  - ``LabelHierarchy`` — la forma del árbol de etiquetas + qué rama es "peligro".
  - ``SyntheticSpec`` — priors/distribuciones del generador sintético.

``encode.py`` aplica los transformadores derivados **a ciegas** (no contiene
constantes de dominio). Así un 2º dominio se conecta cambiando solo
``domains/*`` (INV-12).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal

FeatureKind = Literal["categorical", "boolean", "numeric"]
Row = Mapping[str, object]
"""Fila cruda de input: dict feature→valor (categórica=str, bool, num, o None)."""


@dataclass(frozen=True)
class RawFeature:
    """Feature cruda declarada por el dominio."""

    name: str
    kind: FeatureKind
    categories: tuple[str, ...] = ()  # solo para categorical


@dataclass(frozen=True)
class DerivedFeature:
    """Feature numérica derivada por un transformador PURO del dominio.

    ``fn`` recibe la fila cruda y devuelve un ``float`` (NaN-safe: nunca
    levanta ante ``None`` o claves ausentes). El core la ejecuta a ciegas.
    """

    name: str
    fn: Callable[[Row], float]


@dataclass(frozen=True)
class FeatureSpec:
    raw: tuple[RawFeature, ...]
    derived: tuple[DerivedFeature, ...] = ()
    # Features numéricas (raw o derivadas) que empujan monótonamente al gate
    # de peligro (CatBoost monotone_constraints=+1). DEBEN ser numéricas.
    risk_features: tuple[str, ...] = ()
    # Claves de input aceptadas por el servicio (extra="forbid", frontera de
    # privacidad). Si está vacío, se deriva de las raw features.
    input_allowlist: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        names = [f.name for f in self.raw] + [d.name for d in self.derived]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"FeatureSpec: nombres duplicados {dupes}")
        numeric = set(self.numeric_feature_names)
        bad = [r for r in self.risk_features if r not in numeric]
        if bad:
            raise ValueError(
                f"FeatureSpec: risk_features deben ser numéricas (raw numeric o derived): {bad}"
            )

    @property
    def raw_by_name(self) -> dict[str, RawFeature]:
        return {f.name: f for f in self.raw}

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Orden estable: raw (en orden) + derivadas (en orden)."""
        return tuple(f.name for f in self.raw) + tuple(d.name for d in self.derived)

    @property
    def categorical_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.raw if f.kind == "categorical")

    @property
    def numeric_feature_names(self) -> tuple[str, ...]:
        """Numéricas para el modelo: raw numeric/boolean (encode a 0/1) + todas las derivadas."""
        raw_num = tuple(f.name for f in self.raw if f.kind in ("numeric", "boolean"))
        return raw_num + tuple(d.name for d in self.derived)

    @property
    def accepted_keys(self) -> frozenset[str]:
        return self.input_allowlist or frozenset(f.name for f in self.raw)


@dataclass(frozen=True)
class Node:
    """Nodo interno de la jerarquía: elige entre ``children`` (leaf o node_id)."""

    node_id: str
    children: tuple[str, ...]


@dataclass(frozen=True)
class LabelHierarchy:
    """Árbol de etiquetas (LCPN). ``root`` es el gate binario de peligro."""

    root: str
    nodes: tuple[Node, ...]
    leaves: tuple[str, ...]
    danger_child: str  # child del root que representa "peligro" (node_id o leaf)
    abstain_label: str = "undetermined"

    def __post_init__(self) -> None:
        ids = {n.node_id for n in self.nodes}
        if self.root not in ids:
            raise ValueError(f"root '{self.root}' no está en nodes")
        leaf_set = set(self.leaves)
        for n in self.nodes:
            for c in n.children:
                if c not in ids and c not in leaf_set:
                    raise ValueError(f"child '{c}' de '{n.node_id}' no es node ni leaf")
        root_children = self.node_by_id(self.root).children
        if len(root_children) != 2:
            raise ValueError("el gate raíz debe ser binario (2 children)")
        if self.danger_child not in root_children:
            raise ValueError(f"danger_child '{self.danger_child}' no es child del root")
        if self.abstain_label in leaf_set:
            raise ValueError("abstain_label no debe ser una hoja entrenada")

    def node_by_id(self, node_id: str) -> Node:
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        raise KeyError(node_id)

    @property
    def internal_node_ids(self) -> tuple[str, ...]:
        return tuple(n.node_id for n in self.nodes)

    def is_leaf(self, name: str) -> bool:
        return name in set(self.leaves)

    def path_to_leaf(self, leaf: str) -> tuple[tuple[str, str], ...]:
        """Camino raíz→hoja como tupla de (node_id, child elegido)."""
        if leaf not in set(self.leaves):
            raise KeyError(leaf)

        def _search(node_id: str) -> tuple[tuple[str, str], ...] | None:
            node = self.node_by_id(node_id)
            for child in node.children:
                if child == leaf:
                    return ((node_id, child),)
                if not self.is_leaf(child):
                    sub = _search(child)
                    if sub is not None:
                        return ((node_id, child),) + sub
            return None

        result = _search(self.root)
        if result is None:
            raise KeyError(f"hoja '{leaf}' inalcanzable")
        return result


@dataclass(frozen=True)
class NumericDist:
    mean: float
    std: float
    lo: float
    hi: float


@dataclass(frozen=True)
class LabelProfile:
    """Distribuciones condicionales de features dado un label (generador)."""

    label: str
    prevalence: float
    categorical: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    boolean: Mapping[str, float] = field(default_factory=dict)
    numeric: Mapping[str, NumericDist] = field(default_factory=dict)


@dataclass(frozen=True)
class SyntheticSpec:
    profiles: tuple[LabelProfile, ...]
    n_samples: int = 8000
    seed: int = 20260711

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(p.label for p in self.profiles)


@dataclass(frozen=True)
class Domain:
    """Bundle de config que instancia la plataforma para un dominio."""

    name: str
    features: FeatureSpec
    hierarchy: LabelHierarchy
    synthetic: SyntheticSpec

    def __post_init__(self) -> None:
        leaves = set(self.hierarchy.leaves)
        labels = set(self.synthetic.labels)
        if leaves != labels:
            raise ValueError(
                f"[{self.name}] hojas de la jerarquía {leaves} ≠ labels del generador {labels}"
            )
