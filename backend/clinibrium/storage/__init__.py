"""Persistencia del AuditEvent: Postgres si está, JSONL si no (best-effort)."""
from __future__ import annotations

from clinibrium.storage.persist import persist_audit

__all__ = ["persist_audit"]
