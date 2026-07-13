"""Domain-AGNOSTIC types for the ML layer (the platform boundary).

The core knows NOTHING about vertigo. A domain is defined (in
``ml_engine.domains.*``) by writing three config objects:

  - ``FeatureSpec``   — which features exist, their types, which PURE derived
    transformers the domain provides, and which numeric features are "risk"
    features (monotone direction in the danger gate).
  - ``LabelHierarchy`` — the shape of the label tree + which branch is "danger".
  - ``SyntheticSpec`` — priors/distributions for the synthetic generator.

``encode.py`` applies the derived transformers **blindly** (it contains no
domain constants). This way a 2nd domain plugs in by changing only
``domains/*`` (INV-12).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal

FeatureKind = Literal["categorical", "boolean", "numeric"]
Row = Mapping[str, object]
"""Raw input row: feature→value dict (categorical=str, bool, num, or None)."""


@dataclass(frozen=True)
class RawFeature:
    """Raw feature declared by the domain."""

    name: str
    kind: FeatureKind
    categories: tuple[str, ...] = ()  # only for categorical


@dataclass(frozen=True)
class DerivedFeature:
    """Numeric feature derived by a PURE transformer from the domain.

    ``fn`` receives the raw row and returns a ``float`` (NaN-safe: it never
    raises on ``None`` or missing keys). The core runs it blindly.
    """

    name: str
    fn: Callable[[Row], float]


@dataclass(frozen=True)
class FeatureSpec:
    raw: tuple[RawFeature, ...]
    derived: tuple[DerivedFeature, ...] = ()
    # Numeric features (raw or derived) that push monotonically toward the
    # danger gate (CatBoost monotone_constraints=+1). They MUST be numeric.
    risk_features: tuple[str, ...] = ()
    # Input keys accepted by the service (extra="forbid", privacy boundary).
    # If empty, it is derived from the raw features.
    input_allowlist: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        names = [f.name for f in self.raw] + [d.name for d in self.derived]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"FeatureSpec: duplicate names {dupes}")
        numeric = set(self.numeric_feature_names)
        bad = [r for r in self.risk_features if r not in numeric]
        if bad:
            raise ValueError(
                f"FeatureSpec: risk_features must be numeric (raw numeric or derived): {bad}"
            )

    @property
    def raw_by_name(self) -> dict[str, RawFeature]:
        return {f.name: f for f in self.raw}

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Stable order: raw (in order) + derived (in order)."""
        return tuple(f.name for f in self.raw) + tuple(d.name for d in self.derived)

    @property
    def categorical_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.raw if f.kind == "categorical")

    @property
    def numeric_feature_names(self) -> tuple[str, ...]:
        """Numerics for the model: raw numeric/boolean (encoded to 0/1) + all derived."""
        raw_num = tuple(f.name for f in self.raw if f.kind in ("numeric", "boolean"))
        return raw_num + tuple(d.name for d in self.derived)

    @property
    def accepted_keys(self) -> frozenset[str]:
        return self.input_allowlist or frozenset(f.name for f in self.raw)


@dataclass(frozen=True)
class Node:
    """Internal node of the hierarchy: chooses among ``children`` (leaf or node_id)."""

    node_id: str
    children: tuple[str, ...]


@dataclass(frozen=True)
class LabelHierarchy:
    """Label tree (LCPN). ``root`` is the binary danger gate."""

    root: str
    nodes: tuple[Node, ...]
    leaves: tuple[str, ...]
    danger_child: str  # child of the root that represents "danger" (node_id or leaf)
    abstain_label: str = "undetermined"

    def __post_init__(self) -> None:
        ids = {n.node_id for n in self.nodes}
        if self.root not in ids:
            raise ValueError(f"root '{self.root}' is not in nodes")
        leaf_set = set(self.leaves)
        for n in self.nodes:
            for c in n.children:
                if c not in ids and c not in leaf_set:
                    raise ValueError(f"child '{c}' of '{n.node_id}' is neither node nor leaf")
        root_children = self.node_by_id(self.root).children
        if len(root_children) != 2:
            raise ValueError("the root gate must be binary (2 children)")
        if self.danger_child not in root_children:
            raise ValueError(f"danger_child '{self.danger_child}' is not a child of the root")
        if self.abstain_label in leaf_set:
            raise ValueError("abstain_label must not be a trained leaf")

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
        """Root→leaf path as a tuple of (node_id, chosen child)."""
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
            raise KeyError(f"leaf '{leaf}' is unreachable")
        return result


@dataclass(frozen=True)
class NumericDist:
    mean: float
    std: float
    lo: float
    hi: float


@dataclass(frozen=True)
class LabelProfile:
    """Conditional feature distributions given a label (generator)."""

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
    # Fraction of features dropped (→ missing) per case, so that the synthetic
    # data resembles real SPARSE inputs (Codex: stress with missingness).
    # Avoids over-confidence/over-abstention on incomplete inputs.
    missing_rate: float = 0.0

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(p.label for p in self.profiles)


@dataclass(frozen=True)
class Domain:
    """Config bundle that instantiates the platform for one domain."""

    name: str
    features: FeatureSpec
    hierarchy: LabelHierarchy
    synthetic: SyntheticSpec

    def __post_init__(self) -> None:
        leaves = set(self.hierarchy.leaves)
        labels = set(self.synthetic.labels)
        if leaves != labels:
            raise ValueError(
                f"[{self.name}] hierarchy leaves {leaves} != generator labels {labels}"
            )
