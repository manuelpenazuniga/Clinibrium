"""LOCAL SHAP explainability for a single node (TB1.6, bounded).

Codex/Gemini fix: SHAP values are NOT aggregated across the tree's nodes (it
breaks additivity — each node has its own baseline/logit scale). ONE node is
explained: the **danger gate** (``dangerous vs peripheral``), which is the
most relevant decision. Exact CatBoost TreeSHAP on the raw score (logit) of
the "dangerous" class: positive contribution ⇒ pushes toward danger.

HONESTY (AD-17): on synthetic data this explains the GENERATOR, not clinical
causality. It is presented as "local, non-causal attribution over the
synthetic generator".
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
    """Top-k SHAP contributions (by |value|) of ONE binary node, one row.

    Returns ``{feature: contribution_to_dangerous_logit}``. The last column
    (expected value / base) is discarded. For a binary Logloss model the SHAP
    is over the raw score of the positive class (= dangerous, by gate design).
    """
    pool = Pool(x_row, cat_features=cat_features)
    values = cb_model.get_feature_importance(pool, type="ShapValues")
    contribs = values[0][:-1]  # last column = expected value (base)
    pairs = sorted(
        zip(feature_names, contribs, strict=True), key=lambda kv: -abs(kv[1])
    )
    return {name: round(float(val), 4) for name, val in pairs[:top_k]}
