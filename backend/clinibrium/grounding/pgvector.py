"""`PgvectorGrounding` — cosine-similarity retrieval on pgvector.

AD-10: enriched `Grounding` implementation when PostgreSQL + pgvector
are available. When the DB is absent, the `get_grounding()` factory
degrades to `InlineGrounding` (reliable demo path, same mechanism as
`ml_client`).

Design notes:

- **Lightweight, deterministic embedder** (hashing trick / bag-of-words
  to a fixed-dimension dense vector). Zero torch, zero network APIs,
  zero hidden state. Retrieval quality is **deliberately secondary**:
  RAG is not demo-critical. What matters is demonstrating the
  **pgvector mechanics** (DDL + ingest + top-k by cosine). In
  production it gets replaced by a real embeddings model
  (`SentenceTransformers`, `Ollama`, etc.) — the `Grounding`
  interface contract does not change.
- **Idempotent ingestion** from `inline.CORPUS` (same source of truth).
- **Broad degradation**: if the DB does not answer or the embedder
  fails, the methods that need it do NOT break the gate — `retrieve()`
  returns an empty list and `ingest()` reports the error in logs.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re
from typing import Iterable, Sequence

from clinibrium.contracts import CaseFeatures, Diagnosis, DifferentialResult
from clinibrium.grounding.base import GroundingChunk
from clinibrium.grounding.inline import CORPUS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedder: hashing trick to a dense vector (256 dims) + L2 norm.
# ---------------------------------------------------------------------------

EMBED_DIM = 256

# Minimal (Spanish) stopwords to reduce the weight of tokens that pollute
# the bag. NOT exhaustive — just enough so similarity does not concentrate
# on "el/la/de/que" when clinical terms are present. (Spanish because the
# corpus text is in Spanish.)
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "al", "algo", "algunas", "algunos", "ante", "antes", "como",
        "con", "contra", "cual", "cuando", "de", "del", "desde", "donde",
        "durante", "e", "el", "ella", "ellas", "ellos", "en", "entre",
        "era", "eran", "es", "esa", "esas", "ese", "eso", "esos", "esta",
        "estaba", "estar", "estas", "este", "esto", "estos", "fue", "fueron",
        "ha", "había", "han", "has", "hasta", "hay", "había", "la", "las",
        "le", "les", "lo", "los", "más", "me", "mi", "mis", "mismo", "mucho",
        "muy", "nada", "ni", "no", "nos", "nosotros", "o", "os", "otra",
        "otras", "otro", "otros", "para", "pero", "poco", "por", "que",
        "quien", "quienes", "se", "sea", "sean", "seas", "ser", "si",
        "sido", "siempre", "siendo", "sin", "sobre", "sois", "somos", "son",
        "soy", "su", "sus", "también", "tanto", "te", "tener", "ti", "tiene",
        "tienen", "todo", "todos", "tu", "tus", "un", "una", "uno", "unos",
        "y", "ya", "yo",
    }
)

_TOKEN_RE = re.compile(r"[a-záéíóúñü]+", flags=re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    """Tokenizes to lowercase, filters stopwords and very short tokens."""
    out: list[str] = []
    for match in _TOKEN_RE.finditer(text.lower()):
        tok = match.group(0)
        if len(tok) < 3:
            continue
        if tok in _STOPWORDS:
            continue
        out.append(tok)
    return out


def _hash_to_dim(token: str, dim: int = EMBED_DIM) -> int:
    """Deterministic hash of a token → index in [0, dim).

    Uses SHA-256, takes the first 4 bytes as a little-endian uint32,
    and maps them modulo `dim`. Stable across processes and platforms.
    """
    h = hashlib.sha256(token.encode("utf-8")).digest()
    val = int.from_bytes(h[:4], byteorder="little", signed=False)
    return val % dim


def embed_text(text: str, dim: int = EMBED_DIM) -> list[float]:
    """Fixed-dimension dense vector via hashing trick (TF + L2 norm).

    Deterministic: same `text` and same `dim` ⇒ same vector, in any
    process / platform. It is a **hashed bag-of-words** embedder —
    useful to demonstrate pgvector mechanics, NOT for clinical-quality
    retrieval. See module docstring.
    """
    counts = [0.0] * dim
    for tok in _tokenize(text):
        counts[_hash_to_dim(tok, dim)] += 1.0

    # L2 norm (zero vector if counts=0)
    norm = math.sqrt(sum(c * c for c in counts))
    if norm == 0.0:
        return counts
    return [c / norm for c in counts]


# ---------------------------------------------------------------------------
# Query text construction (features + candidate diagnoses).
# ---------------------------------------------------------------------------


def _features_to_text(features: CaseFeatures) -> str:
    """Converts the set fields of `CaseFeatures` into a string.

    Only fields whose value differs from the default (i.e. carrying
    clinical information) are included. Enums serialize to their
    `.value`; sets are iterated; bools serialize as their names.
    """
    parts: list[str] = []
    # `model_fields` is stable (declaration order in Pydantic v2).
    for fname in type(features).model_fields:
        value = getattr(features, fname)
        if value is None:
            continue
        default = type(features).model_fields[fname].default
        if default is not None and value == default:
            continue
        if isinstance(value, bool):
            parts.append(fname if value else f"no_{fname}")
            continue
        if isinstance(value, set):
            for v in sorted(value, key=lambda x: getattr(x, "value", str(x))):
                parts.append(f"{fname}={getattr(v, 'value', str(v))}")
            continue
        if hasattr(value, "value"):  # enums
            parts.append(f"{fname}={value.value}")
            continue
        if isinstance(value, (int, float)):
            parts.append(f"{fname}={value}")
            continue
    return " ".join(parts)


def _candidates_to_text(candidates: DifferentialResult) -> str:
    """Top candidate diagnoses → text, to anchor the retrieval."""
    return " ".join(c.diagnosis.value for c in candidates.candidates[:5])


def build_query_text(
    candidates: DifferentialResult,
    features: CaseFeatures,
) -> str:
    """Composes the retrieval query: structured features + candidate dx."""
    return f"{_features_to_text(features)} {_candidates_to_text(candidates)}".strip()


# ---------------------------------------------------------------------------
# asyncpg + pgvector connection.
# ---------------------------------------------------------------------------

_TABLE = "clinibrium_grounding_chunks"


def _to_pgvector_literal(vec: Sequence[float]) -> str:
    """Serializes a vector to pgvector's literal syntax: `[v1,v2,...]`."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


async def _connect(database_url: str):  # type: ignore[no-untyped-def]
    """Opens an asyncpg connection and registers the pgvector codec.

    Returns the connection. On failure it raises — the caller degrades.
    """
    import asyncpg
    from pgvector.asyncpg import register_vector

    conn = await asyncpg.connect(database_url, timeout=3.0)
    try:
        await register_vector(conn)
    except Exception:  # noqa: BLE001
        # If pgvector is not installed on the server, close cleanly.
        await conn.close()
        raise
    return conn


async def _ensure_schema(conn) -> None:  # type: ignore[no-untyped-def]
    """Creates the extension and table if they do not exist. Idempotent."""
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_TABLE} (
            id          BIGSERIAL PRIMARY KEY,
            diagnosis   TEXT NOT NULL,
            source_id   TEXT NOT NULL UNIQUE,
            text        TEXT NOT NULL,
            embedding   VECTOR({EMBED_DIM}) NOT NULL
        )
        """
    )


async def _fetch_table_columns(conn) -> set[str]:  # type: ignore[no-untyped-def]
    """Returns the set of existing columns in `_TABLE`.

    Used in `ingest()` to tolerate tables created by previous versions
    of the module (schema evolving during the hackathon).
    """
    rows = await conn.fetch(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = $1",
        _TABLE,
    )
    return {r["column_name"] for r in rows}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _corpus_to_records() -> list[dict[str, object]]:
    """Flattens `CORPUS` into a list of dicts (one per chunk)."""
    records: list[dict[str, object]] = []
    for dx, chunks in CORPUS.items():
        for chunk in chunks:
            records.append(
                {
                    "diagnosis": dx.value,
                    "source_id": chunk.source_id,
                    "text": chunk.text,
                    "embedding": embed_text(chunk.text),
                }
            )
    return records


async def ingest(
    database_url: str,
    *,
    corpus: dict[Diagnosis, list[GroundingChunk]] | None = None,
) -> int:
    """Ingests the corpus into the pgvector table. Idempotent.

    Strategy:
      1. Opens a connection + registers the pgvector codec.
      2. Ensures the schema (extension + table). Idempotent.
      3. UPSERT by `source_id` (unique key) ⇒ calling `ingest()` multiple
         times is safe and leaves the table consistent with `CORPUS`.

    Returns the number of chunks written. If the DB does not answer,
    it **degrades gracefully** (log + return 0) — does not break the gate.
    """
    try:
        conn = await _connect(database_url)
    except Exception as exc:  # noqa: BLE001
        logger.info(
            "grounding.ingest: DB unavailable (%s) → skip ingest",
            type(exc).__name__,
        )
        return 0

    try:
        await _ensure_schema(conn)
        records = _corpus_to_records() if corpus is None else [
            {
                "diagnosis": dx.value,
                "source_id": chunk.source_id,
                "text": chunk.text,
                "embedding": embed_text(chunk.text),
            }
            for dx, chunks in corpus.items()
            for chunk in chunks
        ]

        # Tolerant schema detection: if the table already existed with
        # `embedding VECTOR(N)`, we compare dimensions.
        columns = await _fetch_table_columns(conn)
        # We do not force dimension verification on this path: if N does
        # not match, the ingest query will fail with a pgvector type
        # error and we catch it below as degradation.

        if "embedding" not in columns:
            # Re-creating the table with the embedder's current dimension
            # is not trivial without DROP; we rely on _ensure_schema for
            # a fresh table. If the table existed without an embedding
            # column, we create it now.
            await conn.execute(
                f"ALTER TABLE {_TABLE} ADD COLUMN embedding VECTOR({EMBED_DIM})"
            )

        # UPSERT: if a row with that source_id already exists, update it.
        # `ON CONFLICT (source_id) DO UPDATE` requires pgvector to be able
        # to reassign the vector — it works because the column is of the
        # `vector` type registered by `register_vector`.
        for rec in records:
            await conn.execute(
                f"""
                INSERT INTO {_TABLE} (diagnosis, source_id, text, embedding)
                VALUES ($1, $2, $3, $4::vector)
                ON CONFLICT (source_id) DO UPDATE
                SET diagnosis = EXCLUDED.diagnosis,
                    text = EXCLUDED.text,
                    embedding = EXCLUDED.embedding
                """,
                rec["diagnosis"],
                rec["source_id"],
                rec["text"],
                rec["embedding"],
            )
        return len(records)
    except Exception as exc:  # noqa: BLE001
        logger.info(
            "grounding.ingest: ingestion error (%s) → skip",
            type(exc).__name__,
        )
        return 0
    finally:
        await conn.close()


async def _retrieve_raw(
    database_url: str,
    query_vec: Sequence[float],
    k: int,
) -> list[dict[str, object]]:
    """Top-k by cosine similarity (pgvector's `<=>` operator)."""
    conn = await _connect(database_url)
    try:
        await _ensure_schema(conn)
        rows = await conn.fetch(
            f"""
            SELECT diagnosis, source_id, text,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM {_TABLE}
            ORDER BY embedding <=> $1::vector
            LIMIT $2
            """,
            list(query_vec),
            int(k),
        )
        return [
            {
                "diagnosis": r["diagnosis"],
                "source_id": r["source_id"],
                "text": r["text"],
                "similarity": float(r["similarity"]) if r["similarity"] is not None else 0.0,
            }
            for r in rows
        ]
    finally:
        await conn.close()


class PgvectorGrounding:
    """`Grounding` with cosine-similarity retrieval on pgvector.

    Exposes TWO equivalent entry points:

    - `retrieve(...)` — **sync**, satisfies the `Grounding` Protocol.
      Internally spawns an `asyncio.run()`. Meant for tests, CLI, and
      sync contexts. Do NOT use inside an active event loop.
    - `retrieve_async(...)` — async, for FastAPI / any existing event
      loop. A single `await`, no nested loop.

    Either way: if the DB is unavailable or fails, it **degrades
    gracefully to `[]`** — never raises. That is the invariant of the
    `Grounding` interface (reliable demo path).
    """

    def __init__(self, database_url: str) -> None:
        self._database_url = database_url
        self._available: bool | None = None  # lazy probe

    async def _probe(self) -> bool:
        """Lazy availability check: ping + check the table.

        Caches the result while the DB answers. If the first probe
        fails, it stays at `_available=False` and retries on every
        `retrieve()` (the DB may come up after boot).
        """
        if self._available is True:
            return True
        try:
            conn = await _connect(self._database_url)
            try:
                await _ensure_schema(conn)
                self._available = True
                return True
            finally:
                await conn.close()
        except Exception as exc:  # noqa: BLE001
            logger.info(
                "PgvectorGrounding: DB unavailable (%s) → degrades to []",
                type(exc).__name__,
            )
            self._available = False
            return False

    async def retrieve_async(
        self,
        candidates: DifferentialResult,
        features: CaseFeatures,
        k: int = 4,
    ) -> list[GroundingChunk]:
        """Async version of the retrieval (for use inside an event loop)."""
        if not await self._probe():
            return []
        query_text = build_query_text(candidates, features)
        query_vec = embed_text(query_text)
        try:
            rows = await _retrieve_raw(self._database_url, query_vec, k)
        except Exception as exc:  # noqa: BLE001
            logger.info(
                "PgvectorGrounding.retrieve_async: error (%s) → []",
                type(exc).__name__,
            )
            self._available = False
            return []

        out: list[GroundingChunk] = []
        for row in rows:
            dx_value = row["diagnosis"]
            try:
                dx = Diagnosis(dx_value)
            except ValueError:
                dx = None
            out.append(
                GroundingChunk(
                    text=str(row["text"]),
                    diagnosis=dx,
                    source_id=str(row["source_id"]),
                )
            )
        return out

    def retrieve(
        self,
        candidates: DifferentialResult,
        features: CaseFeatures,
        k: int = 4,
    ) -> list[GroundingChunk]:
        """Sync wrapper over `retrieve_async` (satisfies the Protocol).

        Spawns an internal `asyncio.run()`. Meant for sync usage
        (tests, scripts, gates). In a FastAPI handler use
        `retrieve_async` directly to avoid nesting event loops.
        """
        return asyncio.run(self.retrieve_async(candidates, features, k=k))


# ---------------------------------------------------------------------------
# Sync helpers (test/dev) — tests only. They produce the numeric
# vector, no DB required.
# ---------------------------------------------------------------------------


def embed_query_sync(
    candidates: DifferentialResult,
    features: CaseFeatures,
) -> list[float]:
    """Sync version of query-embedding construction (no DB)."""
    return embed_text(build_query_text(candidates, features))


def all_chunks_flat(
    corpus: dict[Diagnosis, list[GroundingChunk]],
) -> Iterable[tuple[Diagnosis, GroundingChunk]]:
    """Iterates all corpus chunks in stable order."""
    for dx in Diagnosis:
        if dx in corpus:
            for chunk in corpus[dx]:
                yield dx, chunk
