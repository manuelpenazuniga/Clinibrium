"""Tests del módulo `grounding` (T6) — RAG por paráfrasis propia.

Cubre los criterios de aceptación de la tarea:
  (1) `InlineGrounding.retrieve` con candidatos de BPPV posterior +
      Ménière devuelve los chunks de esos diagnósticos, ordenados por
      score del pool, ≤k, determinista.
  (2) El `CORPUS` cubre los 8 diagnósticos documentados con al menos
      1 chunk cada uno.
  (3) `get_grounding()` con `DATABASE_URL=None` devuelve
      `InlineGrounding` (no rompe el gate).
  (4) El embedder es determinista (mismo texto → mismo vector).
  (5) Tests de pgvector que requieren DB se SKIPPEAN si no hay DB
      (gate NO depende de una DB corriendo).
  (6) INV: el módulo `grounding` SOLO importa de `contracts` (+
      `config` + su propia conexión DB). NO importa `reasoner`,
      motores, `orchestrator`, `rails` ni `api`.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from clinibrium.contracts import (
    CaseFeatures,
    Diagnosis,
    DifferentialCandidate,
    DifferentialResult,
    Trigger,
    TimingPattern,
)
from clinibrium.grounding import GroundingChunk, InlineGrounding, get_grounding
from clinibrium.grounding.base import Grounding
from clinibrium.grounding.inline import CORPUS, SUPPORTED_DIAGNOSES
from clinibrium.grounding.pgvector import (
    EMBED_DIM,
    PgvectorGrounding,
    _tokenize,
    build_query_text,
    embed_text,
)


# =========================================================================
# Helpers
# =========================================================================


def _diff(
    *pairs: tuple[Diagnosis, float],
) -> DifferentialResult:
    return DifferentialResult(
        candidates=[
            DifferentialCandidate(diagnosis=dx, score=score) for dx, score in pairs
        ]
    )


def _has_db() -> bool:
    """Devuelve True solo si Postgres responde al TCP probe del DSN.

    Usado por la fixture `_require_db` para skippear tests de pgvector
    cuando la DB no está corriendo (el gate NO debe depender de ella).
    """
    from urllib.parse import urlparse

    import socket

    from clinibrium.config import get_settings

    url = get_settings().DATABASE_URL
    if not url:
        return False
    try:
        parsed = urlparse(url)
    except Exception:  # noqa: BLE001
        return False
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


@pytest.fixture(scope="module")
def db_available() -> bool:
    return _has_db()


@pytest.fixture()
def require_db(db_available: bool) -> None:
    if not db_available:
        pytest.skip("Postgres no disponible — test de pgvector skippeado")


# =========================================================================
# (1) InlineGrounding.retrieve — orden, k, determinismo
# =========================================================================


def test_inline_retrieve_returns_chunks_for_top_candidates() -> None:
    """BPPV posterior + Ménière en el pool ⇒ retrieve devuelve chunks de
    ESOS diagnósticos, en el orden del pool, hasta k.
    """
    g = InlineGrounding()
    result = g.retrieve(
        _diff(
            (Diagnosis.bppv_posterior, 0.95),
            (Diagnosis.meniere, 0.40),
        ),
        CaseFeatures(),
        k=5,
    )
    assert len(result) <= 5
    assert len(result) > 0
    # Todos los chunks devueltos son de BPPV posterior o Ménière
    assert all(
        c.diagnosis in {Diagnosis.bppv_posterior, Diagnosis.meniere} for c in result
    )
    # Los de BPPV van antes que los de Ménière (orden del pool)
    seen_meniere = False
    for c in result:
        if c.diagnosis == Diagnosis.meniere:
            seen_meniere = True
        else:
            assert not seen_meniere, "Ménière apareció antes que BPPV posterior"


def test_inline_retrieve_respects_k_limit() -> None:
    """Con k=2 sobre el pool de BPPV (3 chunks) ⇒ exactamente 2 chunks."""
    g = InlineGrounding()
    result = g.retrieve(
        _diff((Diagnosis.bppv_posterior, 0.99)),
        CaseFeatures(),
        k=2,
    )
    assert len(result) == 2
    assert all(c.diagnosis == Diagnosis.bppv_posterior for c in result)


def test_inline_retrieve_deterministic_two_calls_equal() -> None:
    """Determinismo: dos `retrieve()` con los mismos args ⇒ mismo resultado."""
    g = InlineGrounding()
    args = (
        _diff(
            (Diagnosis.bppv_posterior, 0.95),
            (Diagnosis.meniere, 0.40),
            (Diagnosis.vestibular_neuritis, 0.20),
        ),
        CaseFeatures(trigger=Trigger.positional_head),
        4,
    )
    r1 = g.retrieve(*args)
    r2 = g.retrieve(*args)
    assert r1 == r2


def test_inline_retrieve_deterministic_independent_instances() -> None:
    """Determinismo: dos instancias nuevas del grounding ⇒ mismo resultado."""
    g1 = InlineGrounding()
    g2 = InlineGrounding()
    args = (
        _diff(
            (Diagnosis.meniere, 0.8),
            (Diagnosis.vestibular_migraine, 0.5),
        ),
        CaseFeatures(),
        3,
    )
    assert g1.retrieve(*args) == g2.retrieve(*args)


def test_inline_retrieve_empty_pool_returns_empty() -> None:
    """Sin candidatos, retrieve devuelve lista vacía (no rompe)."""
    g = InlineGrounding()
    result = g.retrieve(DifferentialResult(), CaseFeatures(), k=4)
    assert result == []


def test_inline_retrieve_with_unknown_dx_skips_silently() -> None:
    """Un candidato cuyo diagnóstico no está en el CORPUS no rompe
    retrieve — simplemente no aporta chunks. (No debería pasar: el
    DifferentialEngine solo emite `Diagnosis` válidos, pero la
    interfaz es defensiva.)"""
    g = InlineGrounding()
    # Forzamos un Diagnosis que NO está en CORPUS (`undetermined`).
    result = g.retrieve(
        _diff(
            (Diagnosis.undetermined, 0.5),
            (Diagnosis.bppv_posterior, 0.9),
        ),
        CaseFeatures(),
        k=3,
    )
    # Solo BPPV aporta; `undetermined` no figura en el CORPUS.
    assert all(c.diagnosis == Diagnosis.bppv_posterior for c in result)


def test_inline_satisfies_protocol() -> None:
    """`InlineGrounding` satisface el `Protocol Grounding` (duck typing)."""
    g: Grounding = InlineGrounding()
    assert hasattr(g, "retrieve")
    assert callable(g.retrieve)


# =========================================================================
# (2) Cobertura del CORPUS — los 8 diagnósticos
# =========================================================================


def test_corpus_covers_eight_diagnoses() -> None:
    """El CORPUS cubre los 8 diagnósticos documentados en el spec de T6."""
    assert SUPPORTED_DIAGNOSES == frozenset(
        {
            Diagnosis.bppv_posterior,
            Diagnosis.bppv_horizontal,
            Diagnosis.meniere,
            Diagnosis.vestibular_migraine,
            Diagnosis.vestibular_neuritis,
            Diagnosis.labyrinthitis,
            Diagnosis.central_suspected,
            Diagnosis.cardiogenic_suspected,
        }
    )


def test_corpus_has_at_least_one_chunk_per_diagnosis() -> None:
    """Cada diagnóstico cubierto tiene ≥1 chunk."""
    for dx in SUPPORTED_DIAGNOSES:
        assert CORPUS.get(dx), f"sin chunks para {dx.value}"


def test_corpus_chunks_have_well_formed_source_ids() -> None:
    """`source_id` sigue la convención `clinibrium-paraphrase:<dx>-<n>`."""
    pattern_ok = True
    for dx, chunks in CORPUS.items():
        for i, chunk in enumerate(chunks, start=1):
            assert chunk.source_id == f"clinibrium-paraphrase:{dx.value}-{i}", (
                f"source_id inesperado: {chunk.source_id}"
            )
            assert chunk.diagnosis == dx
            assert pattern_ok
            assert chunk.text  # no vacío


def test_corpus_chunks_are_paramfrasis_propia_marker() -> None:
    """Cada chunk debe tener un texto no trivial (>50 chars) — esto es un
    test de humo de la cantidad de paráfrasis. La auditoría de calidad
    clínica de las paráfrasis es tarea `T-CLIN`; acá solo garantizamos
    que hay contenido原创 (no son placeholders vacíos)."""
    for dx, chunks in CORPUS.items():
        for chunk in chunks:
            assert len(chunk.text) > 50, f"chunk de {dx.value} parece vacío"
            # Heurística suave: ningún chunk repite la marca ICVD literal
            # (no debería — son paráfrasis propias).
            assert "ICVD" not in chunk.text.upper(), (
                f"chunk de {dx.value} contiene literal 'ICVD' — debería ser "
                f"paráfrasis"
            )


# =========================================================================
# (3) Factory — get_grounding() con DATABASE_URL=None ⇒ InlineGrounding
# =========================================================================


def test_get_grounding_with_no_database_url_returns_inline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sin DATABASE_URL: la factory degrada a `InlineGrounding` (no rompe)."""
    from clinibrium.config import get_settings

    monkeypatch.setattr(get_settings(), "DATABASE_URL", None)
    g = get_grounding()
    assert isinstance(g, InlineGrounding)


def test_get_grounding_with_empty_database_url_returns_inline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DATABASE_URL='': la factory también degrada (falsy)."""
    from clinibrium.config import get_settings

    monkeypatch.setattr(get_settings(), "DATABASE_URL", "")
    g = get_grounding()
    assert isinstance(g, InlineGrounding)


def test_get_grounding_with_unreachable_db_returns_inline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DATABASE_URL set pero host:port no reachable ⇒ InlineGrounding."""
    from clinibrium.config import get_settings

    # 127.0.0.1:1 es prácticamente siempre unreachable
    monkeypatch.setattr(
        get_settings(), "DATABASE_URL", "postgresql://x:x@127.0.0.1:1/x"
    )
    g = get_grounding()
    assert isinstance(g, InlineGrounding)


def test_get_grounding_does_not_raise_on_any_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`get_grounding()` NUNCA levanta — ni con URL vacía, ni con DSN malformado."""
    from clinibrium.config import get_settings

    for url in (None, "", "not a url", "postgresql://", "postgresql://x@"):
        monkeypatch.setattr(get_settings(), "DATABASE_URL", url)
        # Solo constatamos que no levanta
        _ = get_grounding()


# =========================================================================
# (4) Embedder — determinista (mismo texto → mismo vector)
# =========================================================================


def test_embedder_is_deterministic() -> None:
    """Mismo texto + misma dim ⇒ mismo vector."""
    a = embed_text("vértigo posicional breve con nistagmo torsional")
    b = embed_text("vértigo posicional breve con nistagmo torsional")
    assert a == b


def test_embedder_is_deterministic_across_calls() -> None:
    """100 llamadas idénticas ⇒ 100 vectores idénticos."""
    text = "ataxia troncal severa con nistagmo vertical puro"
    first = embed_text(text)
    for _ in range(100):
        assert embed_text(text) == first


def test_embedder_is_deterministic_independent_process_objects() -> None:
    """Determinismo via el mismo texto en dos instantes distantes (mismo
    proceso) — confirma que no hay estado oculto."""
    v1 = embed_text("hipoacusia fluctuante con tinnitus y plenitud aural")
    # ... (imaginemos código de cliente en el medio) ...
    v2 = embed_text("hipoacusia fluctuante con tinnitus y plenitud aural")
    assert v1 == v2


def test_embedder_default_dim_is_256() -> None:
    assert EMBED_DIM == 256
    v = embed_text("cualquier texto clínico de al menos tres palabras")
    assert len(v) == 256


def test_embedder_returns_l2_normalized_vector() -> None:
    """El vector es L2-normalizado: ||v||_2 ≈ 1 (salvo texto vacío)."""
    import math

    v = embed_text("vértigo continuo espontáneo con náuseas y vómitos")
    norm = math.sqrt(sum(c * c for c in v))
    assert 0.99 <= norm <= 1.01


def test_embedder_empty_text_returns_zero_vector() -> None:
    """Texto vacío / solo stopwords ⇒ vector cero (sin normalizar)."""
    v_empty = embed_text("")
    v_stop = embed_text("de la el y a")
    assert v_empty == [0.0] * EMBED_DIM
    assert v_stop == [0.0] * EMBED_DIM


def test_embedder_different_texts_produce_different_vectors() -> None:
    """Textos distintos producen vectores distintos (sanity)."""
    a = embed_text("vértigo posicional breve con nistagmo torsional")
    b = embed_text("hipoacusia fluctuante con tinnitus y plenitud aural")
    assert a != b


def test_embedder_tokenize_basic() -> None:
    """Tokenizador: lowercase, drop de stopwords, drop de tokens <3 chars."""
    toks = _tokenize("El vértigo POSICIONAL es breve y se asocia a nistagmo")
    assert "vértigo" in toks
    assert "posicional" in toks
    assert "breve" in toks
    assert "asocia" in toks
    assert "nistagmo" in toks
    # Stopwords filtradas
    assert "el" not in toks
    assert "y" not in toks
    assert "a" not in toks


def test_build_query_text_includes_candidates_and_features() -> None:
    """`build_query_text` compone features estructuradas + top diagnósticos."""
    f = CaseFeatures(
        trigger=Trigger.positional_head,
        timing_pattern=TimingPattern.episodic_triggered,
    )
    q = build_query_text(
        _diff(
            (Diagnosis.bppv_posterior, 0.9),
            (Diagnosis.meniere, 0.3),
        ),
        f,
    )
    assert "bppv_posterior" in q
    assert "meniere" in q
    # El feature `trigger=positional_head` aparece por nombre
    assert "positional_head" in q or "trigger" in q


# =========================================================================
# (5) Tests de pgvector — skippean si no hay DB
# =========================================================================


def test_pgvector_ingest_and_retrieve(require_db: None) -> None:
    """Si hay DB: ingest escribe el CORPUS y retrieve devuelve chunks
    relevantes para el query. Skippea si no hay DB.
    """
    import asyncio

    from clinibrium.config import get_settings
    from clinibrium.grounding.pgvector import ingest

    url = get_settings().DATABASE_URL
    assert url

    async def _run() -> tuple[int, list[GroundingChunk]]:
        n = await ingest(url)
        g = PgvectorGrounding(url)
        chunks = await g.retrieve_async(
            _diff(
                (Diagnosis.bppv_posterior, 0.95),
                (Diagnosis.vestibular_neuritis, 0.4),
            ),
            CaseFeatures(trigger=Trigger.positional_head),
            k=3,
        )
        return n, chunks

    n, chunks = asyncio.run(_run())
    assert n > 0, "ingest no escribió filas"
    assert len(chunks) > 0, "retrieve no devolvió chunks"
    # Los chunks vienen del CORPUS (cualquiera de los 8 diagnósticos
    # es válido, pero el espacio de búsqueda está sobre el universo
    # indexado)
    assert all(isinstance(c, GroundingChunk) for c in chunks)
    assert all(c.source_id.startswith("clinibrium-paraphrase:") for c in chunks)


def test_pgvector_ingest_is_idempotent(require_db: None) -> None:
    """`ingest()` puede llamarse varias veces: la tabla queda consistente
    con el CORPUS (mismo número de filas, mismo source_ids)."""
    import asyncio

    from clinibrium.config import get_settings
    from clinibrium.grounding.pgvector import ingest

    url = get_settings().DATABASE_URL
    assert url

    async def _run() -> tuple[int, int]:
        n1 = await ingest(url)
        n2 = await ingest(url)
        return n1, n2

    n1, n2 = asyncio.run(_run())
    assert n1 == n2
    # Total de chunks del CORPUS (≥8, los 8 diagnósticos con 1+ cada uno)
    assert n1 >= 8


def test_pgvector_retrieve_degrades_when_db_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PgvectorGrounding.retrieve contra DB no reachable ⇒ [] (no rompe)."""
    g = PgvectorGrounding("postgresql://x:x@127.0.0.1:1/x")
    chunks = g.retrieve(  # sync entry point (Protocol)
        _diff((Diagnosis.bppv_posterior, 0.9)),
        CaseFeatures(),
        k=4,
    )
    assert chunks == []


# =========================================================================
# (6) INV — grounding solo importa contracts (+ config + DB driver).
# Verificación por AST (como en otros tests del repo).
# =========================================================================


_FORBIDDEN_TOP_LEVEL = {
    "clinibrium.engines",
    "clinibrium.redflag_engine",
    "clinibrium.differential_engine",
    "clinibrium.reasoner",
    "clinibrium.orchestrator",
    "clinibrium.rails",
    "clinibrium.api",
    "clinibrium.audit",
    "clinibrium.ml_client",
    "clinibrium.storage",
    "clinibrium.fhir",
}


def _iter_imports(py_file: Path) -> list[tuple[int, str]]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                out.append((node.lineno, node.module))
    return out


def test_grounding_does_not_import_forbidden_modules() -> None:
    """El paquete `grounding` SOLO importa de `clinibrium.contracts` y
    `clinibrium.config` (settings) — nunca motores, reasoner, orchestrator,
    rails, api, audit, ml_client, storage, fhir.
    """
    pkg_root = Path(inspect.getfile(InlineGrounding)).parent  # grounding/
    allowed = {"clinibrium.contracts", "clinibrium.config", "clinibrium.grounding"}
    offenders: list[str] = []
    for py in sorted(pkg_root.glob("*.py")):
        for lineno, mod in _iter_imports(py):
            if not mod.startswith("clinibrium."):
                continue
            if not any(mod == a or mod.startswith(a + ".") for a in allowed):
                offenders.append(f"{py.name}:{lineno} → {mod}")
    assert not offenders, (
        "grounding importa módulos prohibidos:\n  "
        + "\n  ".join(offenders)
    )


def test_grounding_does_not_call_claude_or_set_diagnosis() -> None:
    """Sanity: el módulo NO importa el SDK de Anthropic ni define campos
    de diagnosis vinculante. Es solo retrieval.
    """
    pkg_root = Path(inspect.getfile(InlineGrounding)).parent
    forbidden_libs = ("anthropic", "openai", "google.generativeai")
    offenders: list[str] = []
    for py in sorted(pkg_root.glob("*.py")):
        for lineno, mod in _iter_imports(py):
            if any(mod == bad or mod.startswith(bad + ".") for bad in forbidden_libs):
                offenders.append(f"{py.name}:{lineno} → {mod}")
    assert not offenders, (
        "grounding no debe llamar a proveedores de LLM:\n  "
        + "\n  ".join(offenders)
    )
