"""Claude reasoner: pick_model + RAG-grounded; it explains, does NOT classify."""
from __future__ import annotations

from clinibrium.reasoner.engine import reason
from clinibrium.reasoner.pick_model import HAIKU, OPUS, pick_model
from clinibrium.reasoner.privacy import PrivacyViolation, build_network_payload

__all__ = ["HAIKU", "OPUS", "PrivacyViolation", "build_network_payload", "pick_model", "reason"]
