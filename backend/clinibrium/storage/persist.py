"""`AuditEvent` persistence: Postgres if available, JSONL otherwise.

INV-4: NEVER raises on persistence failure — logs and continues.
Emitting the event is an invariant; persistence is best-effort with fallback.
"""
from __future__ import annotations

import json
import logging

import asyncpg

from clinibrium.config import get_settings
from clinibrium.contracts.audit import AuditEvent

logger = logging.getLogger(__name__)

_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS audit_events (
    id          TEXT PRIMARY KEY,
    occurred_at TIMESTAMPTZ NOT NULL,
    event_type  TEXT NOT NULL,
    actor       TEXT NOT NULL,
    model_used  TEXT,
    input_features_hash TEXT NOT NULL,
    urgency     TEXT NOT NULL,
    forced_actions JSONB NOT NULL DEFAULT '[]',
    red_flag_activa  BOOLEAN NOT NULL,
    outcome_summary  TEXT NOT NULL,
    reasoner_status  TEXT NOT NULL DEFAULT 'ok',
    outcome    TEXT NOT NULL DEFAULT 'evaluation'
)
"""


def _event_to_row(event: AuditEvent) -> tuple:
    return (
        event.id,
        event.occurred_at,
        event.event_type,
        event.actor.value,
        event.model_used,
        event.input_features_hash,
        event.urgency.value,
        json.dumps([fa.value for fa in event.forced_actions]),
        event.red_flag_activa,
        event.outcome_summary,
        event.reasoner_status,
        event.outcome,
    )


async def _persist_postgres(event: AuditEvent, database_url: str) -> None:
    conn = await asyncpg.connect(database_url)
    try:
        await conn.execute(_TABLE_DDL)
        await conn.execute(
            """INSERT INTO audit_events (
                id, occurred_at, event_type, actor, model_used,
                input_features_hash, urgency, forced_actions,
                red_flag_activa, outcome_summary, reasoner_status, outcome
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)""",
            *_event_to_row(event),
        )
    finally:
        await conn.close()


def _persist_jsonl(event: AuditEvent, path: str) -> None:
    with open(path, "a") as fh:
        fh.write(event.model_dump_json() + "\n")


async def persist_audit(event: AuditEvent) -> None:
    """Persists the `AuditEvent` with automatic degradation.

    - If `DATABASE_URL` is configured AND the DB is reachable → INSERT into
      `audit_events` (creates the table if it does not exist).
    - Otherwise → immutable append to JSONL (`AUDIT_LOG_PATH`).
    - NEVER raises: logs and continues.
    """
    settings = get_settings()
    database_url = settings.DATABASE_URL

    if database_url:
        try:
            await _persist_postgres(event, database_url)
            return
        except Exception:
            logger.warning(
                "persist_audit: Postgres failed → fallback JSONL (%s)",
                settings.AUDIT_LOG_PATH,
                exc_info=True,
            )

    try:
        _persist_jsonl(event, settings.AUDIT_LOG_PATH)
    except Exception:
        logger.exception(
            "persist_audit: JSONL failed too — event NOT persisted "
            "(INV-4: 1 AuditEvent was ALREADY built; persistence is best-effort)"
        )
