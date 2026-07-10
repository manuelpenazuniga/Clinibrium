"""`get_grounding()` — factory con degradación elegante (AD-10).

Patrón idéntico al de `ml_client`: nunca levanta por la ausencia de la
DB; devuelve `InlineGrounding` (path demo confiable) cuando la DB no
está disponible. El reasoner (T7) consume el grounding a través de la
interfaz `Grounding` y no sabe cuál implementación está activa.

Reglas:
    - `DATABASE_URL` falsy (None o "") → `InlineGrounding`.
    - `DATABASE_URL` seteado pero DB no responde al probe → `InlineGrounding`.
    - `DATABASE_URL` seteado y DB responde → `PgvectorGrounding`
      (que a su vez revalida en `retrieve()` por si la DB cae en runtime).
"""
from __future__ import annotations

import logging
from urllib.parse import urlparse

from clinibrium.config import get_settings
from clinibrium.grounding.base import Grounding
from clinibrium.grounding.inline import InlineGrounding
from clinibrium.grounding.pgvector import PgvectorGrounding

logger = logging.getLogger(__name__)


# Timeout del probe de TCP a host:port. Corto a propósito — la factory
# corre en el path caliente del boot.
_PROBE_TIMEOUT_S: float = 0.5


def _tcp_probe(database_url: str, timeout_s: float = _PROBE_TIMEOUT_S) -> bool:
    """Probe de reachability: abre un socket TCP a host:port del DSN.

    Es deliberadamente **débil** (TCP only, no handshake de Postgres):
    un socket abierto NO garantiza que el server esté listo para
    queries, pero es una señal barata para evitar devolver
    `PgvectorGrounding` cuando el server no está corriendo. El
    `PgvectorGrounding.retrieve()` hace su propia probe async con
    handshake completo, así que la señal débil acá no genera falsos
    positivos dañinos.
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
            "get_grounding: DB no reachable en %s:%s (%s) → InlineGrounding",
            host,
            port,
            exc.strerror or type(exc).__name__,
        )
        return False


def get_grounding() -> Grounding:
    """Devuelve la implementación de `Grounding` activa.

    - `DATABASE_URL` falsy → `InlineGrounding`.
    - `DATABASE_URL` set + DB reachable (TCP) → `PgvectorGrounding`
      (probe async al primer `retrieve()` confirma handshake).
    - cualquier falla → `InlineGrounding` (degradación elegante,
      patrón `ml_client`).

    Esta función NUNCA levanta excepciones; es segura de llamar en
    import-time de los módulos aguas abajo.
    """
    database_url = get_settings().DATABASE_URL
    if not database_url:
        logger.debug("get_grounding: DATABASE_URL no configurada → InlineGrounding")
        return InlineGrounding()

    if not _tcp_probe(database_url):
        return InlineGrounding()

    return PgvectorGrounding(database_url)
