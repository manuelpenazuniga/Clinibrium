"""`PgvectorGrounding` — retrieval por similitud coseno en pgvector.

AD-10: implementación enriquecida del `Grounding` cuando hay
PostgreSQL + pgvector disponibles. Cuando la DB no está, la factory
`get_grounding()` degrada a `InlineGrounding` (path demo confiable,
mecanismo idéntico al de `ml_client`).

Notas de diseño:

- **Embedder liviano y determinista** (hashing trick / bag-of-words a
  vector denso de dimensión fija). Cero torch, cero APIs de red, cero
  estado oculto. La calidad del retrieval es **deliberadamente
  secundaria**: RAG no es demo-crítico. Lo que importa es demostrar
  la **mecánica pgvector** (DDL + ingest + top-k por coseno). En
  producción se reemplaza por un modelo de embeddings real
  (`SentenceTransformers`, `Ollama`, etc.) — el contrato de la
  interfaz `Grounding` no cambia.
- **Ingesta idempotente** desde `inline.CORPUS` (mismo source of truth).
- **Degradación amplia**: si la DB no responde o el embedder falla, los
  métodos que la necesitan NO rompen el gate — `retrieve()` devuelve
  lista vacía y `ingest()` reporta el error en logs.
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
# Embedder: hashing trick a vector denso (256 dims) + L2 norm.
# ---------------------------------------------------------------------------

EMBED_DIM = 256

# Stopwords mínimas (español) para reducir peso de tokens que ensucian
# la bolsa. NO es exhaustivo — es solo para que la similitud no se
# concentre en "el/la/de/que" cuando hay términos clínicos.
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
    """Tokeniza a lowercase, filtra stopwords y tokens de 1 sola letra."""
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
    """Hash determinista de un token → índice en [0, dim).

    Usa SHA-256, toma los primeros 4 bytes como uint32 little-endian,
    y los mapea módulo `dim`. Estable entre procesos y plataformas.
    """
    h = hashlib.sha256(token.encode("utf-8")).digest()
    val = int.from_bytes(h[:4], byteorder="little", signed=False)
    return val % dim


def embed_text(text: str, dim: int = EMBED_DIM) -> list[float]:
    """Vector denso de dimensión fija por hashing-trick (TF + L2 norm).

    Determinista: mismo `text` y misma `dim` ⇒ mismo vector, en cualquier
    proceso / plataforma. Es un embedder de **bolsa de palabras hasheada**
    — útil para mostrar la mecánica de pgvector, NO para retrieval de
    calidad clínica. Ver docstring del módulo.
    """
    counts = [0.0] * dim
    for tok in _tokenize(text):
        counts[_hash_to_dim(tok, dim)] += 1.0

    # L2 norm (vector zero si counts=0)
    norm = math.sqrt(sum(c * c for c in counts))
    if norm == 0.0:
        return counts
    return [c / norm for c in counts]


# ---------------------------------------------------------------------------
# Construcción del texto de query (features + diagnósticos candidatos).
# ---------------------------------------------------------------------------


def _features_to_text(features: CaseFeatures) -> str:
    """Convierte los campos seteados de `CaseFeatures` a una cadena.

    Solo se incluyen los campos con valor distinto al default (i.e. con
    información clínica). Enums se serializan a su `.value`; sets se
    iteran; bools se serializan como sus nombres.
    """
    parts: list[str] = []
    # `model_fields` es estable (orden de declaración en Pydantic v2).
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
    """Top diagnósticos candidatos → texto, para anclar el retrieval."""
    return " ".join(c.diagnosis.value for c in candidates.candidates[:5])


def build_query_text(
    candidates: DifferentialResult,
    features: CaseFeatures,
) -> str:
    """Compone la query de retrieval: features estructuradas + dx candidatos."""
    return f"{_features_to_text(features)} {_candidates_to_text(candidates)}".strip()


# ---------------------------------------------------------------------------
# Conexión asyncpg + pgvector.
# ---------------------------------------------------------------------------

_TABLE = "clinibrium_grounding_chunks"


def _to_pgvector_literal(vec: Sequence[float]) -> str:
    """Serializa un vector a la sintaxis literal de pgvector: `[v1,v2,...]`."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


async def _connect(database_url: str):  # type: ignore[no-untyped-def]
    """Abre una conexión asyncpg y registra el codec de pgvector.

    Devuelve la conexión. Si falla, lanza — el caller degrada.
    """
    import asyncpg
    from pgvector.asyncpg import register_vector

    conn = await asyncpg.connect(database_url, timeout=3.0)
    try:
        await register_vector(conn)
    except Exception:  # noqa: BLE001
        # Si pgvector no está instalado en el server, cerramos limpio.
        await conn.close()
        raise
    return conn


async def _ensure_schema(conn) -> None:  # type: ignore[no-untyped-def]
    """Crea la extensión y la tabla si no existen. Idempotente."""
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
    """Devuelve el set de columnas existentes en `_TABLE`.

    Se usa en `ingest()` para tolerar tablas creadas por versiones
    previas del módulo (esquema evolving durante el hackathon).
    """
    rows = await conn.fetch(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = $1",
        _TABLE,
    )
    return {r["column_name"] for r in rows}


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------


def _corpus_to_records() -> list[dict[str, object]]:
    """Aplana `CORPUS` a una lista de dicts (uno por chunk)."""
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
    """Ingesta el corpus en la tabla pgvector. Idempotente.

    Estrategia:
      1. Abre conexión + registra codec de pgvector.
      2. Asegura esquema (extensión + tabla). Idempotente.
      3. UPSERT por `source_id` (clave única) ⇒ llamar `ingest()` varias
         veces es seguro y deja la tabla consistente con el `CORPUS`.

    Devuelve la cantidad de chunks escritos. Si la DB no responde,
    **degrada elegante** (log + return 0) — no rompe el gate.
    """
    try:
        conn = await _connect(database_url)
    except Exception as exc:  # noqa: BLE001
        logger.info(
            "grounding.ingest: DB no disponible (%s) → skip ingest",
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

        # Detección de esquema tolerante: si la tabla ya existía con
        # `embedding VECTOR(N)`, comparamos dimensión.
        columns = await _fetch_table_columns(conn)
        # No forzamos verificación de dimensión en este path: si N no
        # coincide, la query de ingest fallará con un error de tipo
        # pgvector y la capturamos abajo como degradación.

        if "embedding" not in columns:
            # Re-crear la tabla con la dimensión actual del embedder
            # no es trivial sin DROP; confiamos en _ensure_schema para
            # una tabla fresca. Si la tabla existía sin embedding,
            # creamos la columna ahora.
            await conn.execute(
                f"ALTER TABLE {_TABLE} ADD COLUMN embedding VECTOR({EMBED_DIM})"
            )

        # UPSERT: si la fila con ese source_id ya existe, la actualizamos.
        # `ON CONFLICT (source_id) DO UPDATE` requiere pgvector pueda
        # reasignar el vector — funciona porque la columna es del tipo
        # `vector` registrado por `register_vector`.
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
            "grounding.ingest: error en ingesta (%s) → skip",
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
    """Top-k por similitud coseno (operador `<=>` de pgvector)."""
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
    """`Grounding` con retrieval por similitud coseno en pgvector.

    Expone DOS puntos de entrada equivalentes:

    - `retrieve(...)` — **sync**, satisface el `Protocol Grounding`.
      Internamente lanza un `asyncio.run()`. Pensado para tests, CLI,
      y contextos sync. NO usar dentro de un event loop activo.
    - `retrieve_async(...)` — async, para FastAPI / cualquier event
      loop existente. Un solo `await`, sin loop anidado.

    En cualquier caso: si la DB no está disponible o falla, **degrada
    elegante a `[]`** — nunca levanta. Esa es la invariante de la
    interfaz `Grounding` (path demo confiable).
    """

    def __init__(self, database_url: str) -> None:
        self._database_url = database_url
        self._available: bool | None = None  # lazy probe

    async def _probe(self) -> bool:
        """Chequeo de disponibilidad lazy: ping + ver tabla.

        Cachea el resultado mientras la DB responda. Si la primera
        probe falla, queda en `_available=False` y reintenta en cada
        `retrieve()` (la DB puede levantarse después del boot).
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
                "PgvectorGrounding: DB no disponible (%s) → degrada a []",
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
        """Versión async del retrieval (para usar dentro de un event loop)."""
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
        """Sync wrapper sobre `retrieve_async` (satisface el Protocol).

        Lanza un `asyncio.run()` interno. Pensado para uso sync
        (tests, scripts, gates). En un handler de FastAPI usar
        `retrieve_async` directamente para no anidar event loops.
        """
        return asyncio.run(self.retrieve_async(candidates, features, k=k))


# ---------------------------------------------------------------------------
# Helpers de sync (test/dev) — solo para tests. Producen el vector
# numérico, no requieren DB.
# ---------------------------------------------------------------------------


def embed_query_sync(
    candidates: DifferentialResult,
    features: CaseFeatures,
) -> list[float]:
    """Versión sync de construcción de query-embedding (sin DB)."""
    return embed_text(build_query_text(candidates, features))


def all_chunks_flat(
    corpus: dict[Diagnosis, list[GroundingChunk]],
) -> Iterable[tuple[Diagnosis, GroundingChunk]]:
    """Itera todos los chunks del corpus en orden estable."""
    for dx in Diagnosis:
        if dx in corpus:
            for chunk in corpus[dx]:
                yield dx, chunk
