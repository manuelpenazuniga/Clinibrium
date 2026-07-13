"""AuditEvent persistence: Postgres if available, JSONL otherwise (best-effort)."""
from __future__ import annotations

from clinibrium.storage.persist import persist_audit

__all__ = ["persist_audit"]
