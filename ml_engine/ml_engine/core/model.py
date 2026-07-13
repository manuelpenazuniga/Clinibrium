"""Hierarchical model (LCPN) on CatBoost — domain-agnostic.

One CatBoost classifier per internal node of the ``LabelHierarchy``. The leaf
probability is the product of the conditionals along the path.

**Safety inside the ML (INV-9):** the root node is a BINARY GATE
``dangerous vs peripheral`` whose target is encoded 1=dangerous, with
``monotone_constraints=+1`` on the numeric risk features. Hard guarantee:
raising a risk feature (ceteris paribus) NEVER lowers the gate's
``P(dangerous)``. The guarantee is ONLY on the gate (pre-abstention); the
children do not have it (CatBoost constrains per-feature, not per-class → the
multiclass version was unimplementable, Codex+Gemini fix).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from ml_engine.core.abstain import ConfidenceGate
from ml_engine.core.calibrate import TemperatureCalibrator
from ml_engine.core.encode import encode
from ml_engine.core.explain import top_shap_for_node
from ml_engine.core.spec import Domain, LabelHierarchy

_MANIFEST = "manifest.json"

_DEFAULT_PARAMS = {
    "depth": 5,
    "iterations": 250,
    "learning_rate": 0.08,
    "loss_function": "Logloss",  # overridden to MultiClass on n-ary nodes
    "verbose": False,
    "allow_writing_files": False,
    "thread_count": 1,  # reproducible determinism
}


def _danger_leaves(h: LabelHierarchy) -> set[str]:
    """Leaves reachable under the root gate's ``danger_child``."""
    out: set[str] = set()

    def _collect(name: str) -> None:
        if h.is_leaf(name):
            out.add(name)
            return
        for c in h.node_by_id(name).children:
            _collect(c)

    _collect(h.danger_child)
    return out


def _child_on_path(h: LabelHierarchy, node_id: str, leaf: str) -> str | None:
    """Child of ``node_id`` on the path toward ``leaf`` (or None if not applicable)."""
    for nid, child in h.path_to_leaf(leaf):
        if nid == node_id:
            return child
    return None


@dataclass
class _NodeModel:
    node_id: str
    classes: list[str]  # children of the node (leaf or node_id), stable order
    model: CatBoostClassifier
    is_gate: bool = False  # binary gate with target 1=dangerous


class HierarchicalCatBoost:
    """LCPN ensemble. Train with :meth:`train`; predict with :meth:`predict_proba`."""

    def __init__(
        self,
        domain: Domain,
        nodes: dict[str, _NodeModel],
        model_version: str,
        *,
        calibrator: TemperatureCalibrator | None = None,
        abstainer: ConfidenceGate | None = None,
    ) -> None:
        self.domain = domain
        self.features = domain.features
        self.hierarchy = domain.hierarchy
        self._nodes = nodes
        self.model_version = model_version
        self.calibrator = calibrator
        self.abstainer = abstainer

    # ---- training --------------------------------------------------------

    @classmethod
    def train(
        cls,
        domain: Domain,
        df: pd.DataFrame,
        *,
        seed: int = 20260711,
        params: dict | None = None,
    ) -> HierarchicalCatBoost:
        h = domain.hierarchy
        x, cat_cols = encode(df, domain.features)
        y_leaf = df["label"].to_numpy()
        base = {**_DEFAULT_PARAMS, **(params or {}), "random_seed": seed}
        risk = list(domain.features.risk_features)

        nodes: dict[str, _NodeModel] = {}
        for node in h.nodes:
            nid = node.node_id
            children = list(node.children)
            # subset of rows that pass through this node + their child-target
            targets: list[str] = []
            mask: list[bool] = []
            for leaf in y_leaf:
                child = _child_on_path(h, nid, str(leaf))
                mask.append(child is not None)
                targets.append(child if child is not None else "")
            mask_arr = np.array(mask)
            x_node = x.loc[mask_arr]
            t_node = np.array(targets)[mask_arr]

            is_gate = nid == h.root
            if is_gate:
                # binary target 1=dangerous (controls the sign of the monotonicity)
                y = (t_node == h.danger_child).astype(int)
                mono = {r: 1 for r in risk}
                model = CatBoostClassifier(
                    **{**base, "loss_function": "Logloss", "monotone_constraints": mono}
                )
            elif len(children) == 2:
                y = t_node
                model = CatBoostClassifier(**{**base, "loss_function": "Logloss"})
            else:
                y = t_node
                model = CatBoostClassifier(**{**base, "loss_function": "MultiClass"})

            pool = Pool(x_node, y, cat_features=cat_cols)
            model.fit(pool)
            nodes[nid] = _NodeModel(nid, children, model, is_gate=is_gate)

        version = f"synthetic-v1-seed{seed}"
        return cls(domain, nodes, version)

    # ---- persistence (TB1.8) ---------------------------------------------

    def save(self, out_dir: str | Path) -> None:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, Any] = {
            "domain": self.domain.name,
            "model_version": self.model_version,
            "feature_names": list(self.features.feature_names),
            "nodes": [],
        }
        for nid, nm in self._nodes.items():
            fname = f"node_{nid}.cbm"
            nm.model.save_model(str(d / fname))
            manifest["nodes"].append(
                {"node_id": nid, "classes": nm.classes, "is_gate": nm.is_gate, "file": fname}
            )
        if self.calibrator is not None:
            manifest["calibration"] = {
                "temperature": self.calibrator.temperature,
                "leaves": list(self.calibrator.leaves),
            }
        if self.abstainer is not None:
            manifest["abstention"] = {
                "threshold": self.abstainer.threshold,
                "abstain_label": self.abstainer.abstain_label,
            }
        (d / _MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True))

    @classmethod
    def load(cls, in_dir: str | Path, domain: Domain) -> HierarchicalCatBoost:
        d = Path(in_dir)
        manifest = json.loads((d / _MANIFEST).read_text())
        if manifest["domain"] != domain.name:
            raise ValueError(
                f"manifest domain '{manifest['domain']}' != '{domain.name}'"
            )
        nodes: dict[str, _NodeModel] = {}
        for entry in manifest["nodes"]:
            model = CatBoostClassifier()
            model.load_model(str(d / entry["file"]))
            nodes[entry["node_id"]] = _NodeModel(
                entry["node_id"], list(entry["classes"]), model, is_gate=entry["is_gate"]
            )
        cal = None
        if "calibration" in manifest:
            c = manifest["calibration"]
            cal = TemperatureCalibrator(tuple(c["leaves"]), float(c["temperature"]))
        ab = None
        if "abstention" in manifest:
            a = manifest["abstention"]
            ab = ConfidenceGate(float(a["threshold"]), a["abstain_label"])
        return cls(domain, nodes, manifest["model_version"], calibrator=cal, abstainer=ab)

    # ---- inference ---------------------------------------------------------

    def _node_conditional(self, nm: _NodeModel, x_row: pd.DataFrame) -> dict[str, float]:
        """P(child | node) for one row (1×features)."""
        proba = nm.model.predict_proba(x_row)[0]
        if nm.is_gate:
            # class 1 = dangerous = danger_child; class 0 = the other child
            p_danger = float(proba[1])
            other = [c for c in nm.classes if c != self.hierarchy.danger_child][0]
            return {self.hierarchy.danger_child: p_danger, other: 1.0 - p_danger}
        classes = [str(c) for c in nm.model.classes_]
        return {cls_: float(p) for cls_, p in zip(classes, proba, strict=True)}

    def gate_danger_proba_encoded(self, x_row: pd.DataFrame) -> float:
        """Root gate's P(dangerous) on an ALREADY encoded row (for INV-9)."""
        gate = self._nodes[self.hierarchy.root]
        return float(gate.model.predict_proba(x_row)[0][1])

    def predict_proba_one(self, row: dict) -> dict[str, float]:
        """Distribution over the 8 leaves (Σ≈1). Does NOT include abstention."""
        x_row, _ = encode([row], self.features)
        leaf_probs: dict[str, float] = {}
        # BFS from the root propagating the probability mass
        node_mass: dict[str, float] = {self.hierarchy.root: 1.0}
        queue = [self.hierarchy.root]
        while queue:
            nid = queue.pop(0)
            nm = self._nodes[nid]
            cond = self._node_conditional(nm, x_row)
            for child, p_cond in cond.items():
                p = node_mass[nid] * p_cond
                if self.hierarchy.is_leaf(child):
                    leaf_probs[child] = leaf_probs.get(child, 0.0) + p
                else:
                    node_mass[child] = p
                    queue.append(child)
        # normalize for numeric safety
        total = sum(leaf_probs.values())
        if total > 0:
            leaf_probs = {k: v / total for k, v in leaf_probs.items()}
        return leaf_probs

    def predict_proba(self, rows: list[dict]) -> list[dict[str, float]]:
        return [self.predict_proba_one(r) for r in rows]

    def explain_gate(self, row: dict, *, top_k: int = 6) -> dict[str, float]:
        """Local SHAP of the danger GATE (TB1.6): {feature → contribution to P(dangerous)}.

        Attribution of ONE node (no cross-node aggregation). Positive ⇒ pushes
        toward danger. On synthetic data it explains the generator, not clinical
        causality.
        """
        x_row, cat = encode([row], self.features)
        gate = self._nodes[self.hierarchy.root]
        return top_shap_for_node(
            gate.model, x_row, cat, self.features.feature_names, top_k=top_k
        )

    def predict_case(self, row: dict) -> dict[str, float]:
        """FINAL distribution over the 9 keys (8 leaves + abstention, Σ≈1).

        Applies calibration (if present) and abstention (if present). This is
        what the ``/predict`` endpoint serves. If there is no abstainer,
        ``undetermined`` stays at 0.
        """
        p = self.predict_proba_one(row)
        if self.calibrator is not None:
            p = self.calibrator.transform_one(p)
        if self.abstainer is not None:
            return self.abstainer.apply(p)
        out = dict(p)
        out.setdefault(self.hierarchy.abstain_label, 0.0)
        return out
