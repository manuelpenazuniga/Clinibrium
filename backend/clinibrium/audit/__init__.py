"""Construcción y emisión del AuditEvent (INV-4)."""
from __future__ import annotations

from clinibrium.audit.engine import build_audit_event, emit, emit_decision

__all__ = ["build_audit_event", "emit", "emit_decision"]
