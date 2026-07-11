"""Explicabilidad SHAP LOCAL de un solo nodo (TB1.6, acotado).

Fix Codex/Gemini: NO se agregan valores SHAP a través de los nodos del árbol
(rompe la aditividad — cada nodo tiene su propio baseline/escala logit). Se
explica UN nodo: el **gate de peligro** (``dangerous vs peripheral``), que es la
decisión más relevante. TreeSHAP exacto de CatBoost sobre el score crudo (logit)
de la clase "dangerous": contribución positiva ⇒ empuja hacia peligro.

HONESTIDAD (AD-17): sobre datos sintéticos esto explica el GENERADOR, no
causalidad clínica. Se presenta como "atribución local, no causal, sobre el
generador sintético".
"""
from __future__ import annotations

from catboost import CatBoostClassifier, Pool


def top_shap_for_node(
    cb_model: CatBoostClassifier,
    x_row,
    cat_features: list[str],
    feature_names: tuple[str, ...],
    *,
    top_k: int = 6,
) -> dict[str, float]:
    """Top-k contribuciones SHAP (por |valor|) de UN nodo binario, una fila.

    Devuelve ``{feature: contribución_al_logit_de_dangerous}``. Se descarta la
    última columna (expected value / base). Para un modelo binario Logloss el
    SHAP es sobre el score crudo de la clase positiva (= dangerous, por diseño
    del gate).
    """
    pool = Pool(x_row, cat_features=cat_features)
    values = cb_model.get_feature_importance(pool, type="ShapValues")
    contribs = values[0][:-1]  # última columna = expected value (base)
    pairs = sorted(
        zip(feature_names, contribs, strict=True), key=lambda kv: -abs(kv[1])
    )
    return {name: round(float(val), 4) for name, val in pairs[:top_k]}
