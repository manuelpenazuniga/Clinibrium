"""Presentation-layer i18n (canonical IDs → localized labels).

This package is PRESENTATION ONLY. It exists so that clinician-facing strings
produced by the deterministic core can be rendered in the judge's chosen
language WITHOUT the core ever depending on language.

Hard boundary (safety):
  - The deterministic engines (`redflag_engine`, `differential_engine`,
    `rails`, `counterfactual`) keep producing their canonical Spanish labels
    and stable IDs/keys. They NEVER import this package.
  - Localization happens at the API serialization boundary (`api/*`), by
    swapping a label for its English translation keyed by the STABLE id/key
    when `lang == "en"`. Spanish (`lang == "es"`, the default) is a no-op, so
    the recorded Spanish demo is byte-identical.

`Lang` is the only accepted vocabulary: "es" (default) or "en".
"""
from __future__ import annotations

from clinibrium.i18n.labels import (
    COUNTERFACTUAL_LABELS_EN,
    REDFLAG_LABELS_EN,
    Lang,
    localize_counterfactual_change,
    localize_redflag_label,
)

__all__ = [
    "COUNTERFACTUAL_LABELS_EN",
    "REDFLAG_LABELS_EN",
    "Lang",
    "localize_counterfactual_change",
    "localize_redflag_label",
]
