"""English label maps for backend-produced clinician-facing strings.

Canonical vocabulary lives in the engines (Spanish label + stable id/key).
Here we ONLY hold the English rendering, keyed by that stable id/key, plus
pure helpers that resolve id → localized text at the presentation boundary.

Adding/renaming a rule or a perturbation is a table edit in the engine; if a
key is missing here we fall back to the canonical Spanish label (never crash,
never blank) so a new clinical rule is safe by default.
"""
from __future__ import annotations

from typing import Literal

Lang = Literal["es", "en"]

# Red-flag rule id (redflag_engine/rules.py `RedFlagRule.id`) → English label.
# The Spanish source of truth stays in rules.py; these are hand-authored
# clinical translations. Keep this dict in sync with RULES ids.
REDFLAG_LABELS_EN: dict[str, str] = {
    "A1": "AVS with HINTS suspicious for a central cause",
    "A2": "Pure vertical or torsional nystagmus",
    "A3": "Direction-changing nystagmus",
    "A4": "Severe truncal ataxia",
    "A5": "Focal neurological signs",
    "A6": "Sudden severe headache or neck pain",
    "A7": "AVS + age + vascular risk",
    "A8": "Sudden unilateral hearing loss + acute vertigo (AICA)",
    "A9": "Altered consciousness with acute vertigo",
    "A10": "Nystagmus not suppressed by fixation in AVS",
    "B1": "Isolated sudden sensorineural hearing loss (priority ENT, 48h)",
    "B2": "Fever with neck stiffness or altered consciousness",
    "B3": "Cardiogenic pattern (syncope, palpitations, chest pain)",
    "B4": "Otitis or mastoiditis with vertigo",
    "B5": "Recent head or neck trauma",
    "C1": "Significant cervical pathology",
    "C2": "Known carotid or vertebrobasilar disease",
    "C3": "Cardiovascular instability",
    "E4": "Worsening during the flow",
}

# Counterfactual perturbation key (counterfactual/engine.py `_Perturbation.key`)
# → English label. The Spanish source of truth stays in engine.py.
COUNTERFACTUAL_LABELS_EN: dict[str, str] = {
    "focal_signs.diplopia": "New focal sign: diplopia",
    "focal_signs.dysarthria": "New focal sign: dysarthria",
    "skew_deviation": "Skew deviation present",
    "truncal_ataxia_severe": "Severe truncal ataxia (cannot walk)",
    "nystagmus_direction.direction_changing": "Gaze-direction-changing nystagmus",
    "nystagmus_direction.torsional_pure": "Pure torsional spontaneous nystagmus",
    "headache_neck_pain_sudden_severe": "Sudden, severe headache / neck pain",
    "hearing_loss.sudden_unilateral": "Sudden unilateral hearing loss",
    "altered_consciousness": "Altered consciousness",
    "presyncope_syncope": "Presyncope or syncope",
    "neck_stiffness": "Neck stiffness",
    "recent_head_neck_trauma": "Recent head / neck trauma",
}


def localize_redflag_label(rule_id: str, canonical_label: str, lang: Lang) -> str:
    """Resolve a red-flag hit label for `lang`.

    Spanish is a no-op (returns the canonical label unchanged). English falls
    back to the canonical Spanish label if the id is not translated yet.
    """
    if lang == "en":
        return REDFLAG_LABELS_EN.get(rule_id, canonical_label)
    return canonical_label


def localize_counterfactual_change(
    change_key: str, canonical_change: str, lang: Lang
) -> str:
    """Resolve a counterfactual `change` label for `lang` (Spanish = no-op)."""
    if lang == "en":
        return COUNTERFACTUAL_LABELS_EN.get(change_key, canonical_change)
    return canonical_change


__all__ = [
    "COUNTERFACTUAL_LABELS_EN",
    "REDFLAG_LABELS_EN",
    "Lang",
    "localize_counterfactual_change",
    "localize_redflag_label",
]
