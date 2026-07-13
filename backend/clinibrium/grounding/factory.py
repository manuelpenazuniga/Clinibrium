"""`get_grounding()` — factory with graceful degradation (AD-10).

Same pattern as `ml_client`: never raises because the DB is absent;
returns `InlineGrounding` (the reliable demo path) when the DB is
unavailable. The reasoner (T7) consumes the grounding through the
`Grounding` interface and does not know which implementation is active.

Rules:
    - `DATABASE_URL` falsy (None or "") → `InlineGrounding`.
    - `DATABASE_URL` set but DB does not answer the probe → `InlineGrounding`.
    - `DATABASE_URL` set and DB answers → `PgvectorGrounding`
      (which re-validates in `retrieve()` in case the DB goes down at runtime).
"""
from __future__ import annotations

import logging
from urllib.parse import urlparse

from clinibrium.config import get_settings
from clinibrium.grounding.base import Grounding
from clinibrium.grounding.inline import InlineGrounding
from clinibrium.grounding.pgvector import PgvectorGrounding

logger = logging.getLogger(__name__)


# Timeout for the TCP probe to host:port. Deliberately short — the factory
# runs on the hot boot path.
_PROBE_TIMEOUT_S: float = 0.5


def _tcp_probe(database_url: str, timeout_s: float = _PROBE_TIMEOUT_S) -> bool:
    """Reachability probe: opens a TCP socket to the DSN's host:port.

    Deliberately **weak** (TCP only, no Postgres handshake): an open
    socket does NOT guarantee the server is ready for queries, but it
    is a cheap signal to avoid returning `PgvectorGrounding` when the
    server is not running. `PgvectorGrounding.retrieve()` performs its
    own async probe with a full handshake, so the weak signal here does
    not produce harmful false positives.
    """
    try:
        parsed = urlparse(database_url)
    except Exception:  # noqa: BLE001
        return False
    host = parsed.hostname
    port = parsed.port or 5432
    if not host:
        return False

    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError as exc:
        logger.info(
            "get_grounding: DB not reachable at %s:%s (%s) → InlineGrounding",
            host,
            port,
            exc.strerror or type(exc).__name__,
        )
        return False


def get_grounding() -> Grounding:
    """Returns the active `Grounding` implementation.

    - `DATABASE_URL` falsy → `InlineGrounding`.
    - `DATABASE_URL` set + DB reachable (TCP) → `PgvectorGrounding`
      (async probe on the first `retrieve()` confirms the handshake).
    - any failure → `InlineGrounding` (graceful degradation,
      `ml_client` pattern).

    This function NEVER raises exceptions; it is safe to call at
    import time of downstream modules.
    """
    database_url = get_settings().DATABASE_URL
    if not database_url:
        logger.debug("get_grounding: DATABASE_URL not configured → InlineGrounding")
        return InlineGrounding()

    if not _tcp_probe(database_url):
        return InlineGrounding()

    return PgvectorGrounding(database_url)
