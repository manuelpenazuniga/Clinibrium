"""Modelo jerárquico (LCPN) sobre CatBoost — agnóstico de dominio.

Un clasificador CatBoost por nodo interno de la ``LabelHierarchy``. La
probabilidad de hoja es el producto de las condicionales del camino.

**Seguridad dentro del ML (INV-9):** el nodo raíz es un GATE BINARIO
``dangerous vs peripheral`` cuyo target se codifica 1=dangerous, con
``monotone_constraints=+1`` sobre las features numéricas de riesgo. Garantía
dura: subir una feature de riesgo (ceteris paribus) NUNCA baja ``P(dangerous)``
del gate. La garantía es SOLO sobre el gate (pre-abstención); los hijos no la
tienen (CatBoost restringe por-feature, no per-clase → la versión multiclase
era inimplementable, fix Codex+Gemini).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from ml_engine.core.encode import encode
from ml_engine.core.spec import Domain, LabelHierarchy

_DEFAULT_PARAMS = {
    "depth": 5,
    "iterations": 250,
    "learning_rate": 0.08,
    "loss_function": "Logloss",  # se sobreescribe a MultiClass en nodos n-arios
    "verbose": False,
    "allow_writing_files": False,
    "thread_count": 1,  # determinismo reproducible
}


def _danger_leaves(h: LabelHierarchy) -> set[str]:
    """Hojas alcanzables bajo el ``danger_child`` del gate raíz."""
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
    """Child de ``node_id`` en el camino hacia ``leaf`` (o None si no aplica)."""
    for nid, child in h.path_to_leaf(leaf):
        if nid == node_id:
            return child
    return None


@dataclass
class _NodeModel:
    node_id: str
    classes: list[str]  # children del nodo (leaf o node_id), en orden estable
    model: CatBoostClassifier
    is_gate: bool = False  # gate binario con target 1=dangerous


class HierarchicalCatBoost:
    """Ensamble LCPN. Entrenar con :meth:`train`; predecir con :meth:`predict_proba`."""

    def __init__(self, domain: Domain, nodes: dict[str, _NodeModel]) -> None:
        self.domain = domain
        self.features = domain.features
        self.hierarchy = domain.hierarchy
        self._nodes = nodes

    # ---- entrenamiento --------------------------------------------------

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
            # subconjunto de filas que pasan por este nodo + su child-target
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
                # target binario 1=dangerous (controla el signo de la monotonía)
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

        return cls(domain, nodes)

    # ---- inferencia -----------------------------------------------------

    def _node_conditional(self, nm: _NodeModel, x_row: pd.DataFrame) -> dict[str, float]:
        """P(child | node) para una fila (1×features)."""
        proba = nm.model.predict_proba(x_row)[0]
        if nm.is_gate:
            # class 1 = dangerous = danger_child; class 0 = el otro child
            p_danger = float(proba[1])
            other = [c for c in nm.classes if c != self.hierarchy.danger_child][0]
            return {self.hierarchy.danger_child: p_danger, other: 1.0 - p_danger}
        classes = [str(c) for c in nm.model.classes_]
        return {cls_: float(p) for cls_, p in zip(classes, proba, strict=True)}

    def gate_danger_proba_encoded(self, x_row: pd.DataFrame) -> float:
        """P(dangerous) del gate raíz sobre una fila YA codificada (para INV-9)."""
        gate = self._nodes[self.hierarchy.root]
        return float(gate.model.predict_proba(x_row)[0][1])

    def predict_proba_one(self, row: dict) -> dict[str, float]:
        """Distribución sobre las 8 hojas (Σ≈1). NO incluye abstención."""
        x_row, _ = encode([row], self.features)
        leaf_probs: dict[str, float] = {}
        # BFS desde la raíz propagando la masa de probabilidad
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
        # normaliza por seguridad numérica
        total = sum(leaf_probs.values())
        if total > 0:
            leaf_probs = {k: v / total for k, v in leaf_probs.items()}
        return leaf_probs

    def predict_proba(self, rows: list[dict]) -> list[dict[str, float]]:
        return [self.predict_proba_one(r) for r in rows]
