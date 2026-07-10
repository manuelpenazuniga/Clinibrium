"""Validador INV-2 (fail-closed): choke point del payload que cruza la red."""
from __future__ import annotations

from clinibrium.contracts import NETWORK_SAFE_FIELDS, CaseFeatures


class PrivacyViolation(Exception):
    """Un campo fuera del allowlist NETWORK_SAFE_FIELDS intentó cruzar la red."""


def build_network_payload(features: CaseFeatures) -> dict:
    """ÚNICO constructor del payload que cruza la red a Claude. Fail-closed.

    Si aparece cualquier clave fuera de NETWORK_SAFE_FIELDS, LEVANTA (bug de
    seguridad, no algo a filtrar en silencio). Devuelve el dict validado que
    engine.py usa para armar el prompt.
    """
    payload = features.model_dump(mode="json")
    extra = set(payload.keys()) - NETWORK_SAFE_FIELDS
    if extra:
        raise PrivacyViolation(
            f"Campos fuera del allowlist intentando cruzar la red: {extra}"
        )
    return payload
