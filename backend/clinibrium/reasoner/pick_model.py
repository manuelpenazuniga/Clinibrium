"""Único punto de decisión de modelo (AD-6). No se filtra al frontend."""
from __future__ import annotations

from clinibrium.contracts import RedFlagResult

OPUS = "claude-opus-4-8"
HAIKU = "claude-haiku-4-5-20251001"


def pick_model(red_flag: RedFlagResult, *, recording_mode: bool = False) -> str:
    if recording_mode or red_flag.red_flag_activa:
        return OPUS
    return HAIKU
