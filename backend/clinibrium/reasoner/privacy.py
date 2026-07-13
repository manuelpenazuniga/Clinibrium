"""INV-2 validator (fail-closed): choke point for the payload that crosses the network."""
from __future__ import annotations

from clinibrium.contracts import NETWORK_SAFE_FIELDS, CaseFeatures


class PrivacyViolation(Exception):
    """A field outside the NETWORK_SAFE_FIELDS allowlist tried to cross the network."""


def build_network_payload(features: CaseFeatures) -> dict:
    """The ONLY builder of the payload that crosses the network to Claude. Fail-closed.

    If any key outside NETWORK_SAFE_FIELDS appears, it RAISES (a security
    bug, not something to filter silently). Returns the validated dict that
    engine.py uses to build the prompt.
    """
    payload = features.model_dump(mode="json")
    extra = set(payload.keys()) - NETWORK_SAFE_FIELDS
    if extra:
        raise PrivacyViolation(
            f"Fields outside the allowlist attempting to cross the network: {extra}"
        )
    return payload
