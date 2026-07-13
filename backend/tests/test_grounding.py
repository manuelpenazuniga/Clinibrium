"""Tests for the `grounding` module (T6) — RAG via own paraphrase.

Covers the task's acceptance criteria:
  (1) `InlineGrounding.retrieve` with posterior BPPV + Ménière candidates
      returns the chunks for those diagnoses, ordered by pool score,
      ≤k, deterministic.
  (2) The `CORPUS` covers the 8 documented diagnoses with at least
      1 chunk each.
  (3) `get_grounding()` with `DATABASE_URL=None` returns
      `InlineGrounding` (does not break the gate).
  (4) The embedder is deterministic (same text → same vector).
  (5) pgvector tests that require a DB are SKIPPED when no DB is
      available (the gate does NOT depend on a running DB).
  (6) INV: the `grounding` module ONLY imports from `contracts` (+
      `config` + its own DB connection). It does NOT import `reasoner`,
      engines, `orchestrator`, `rails` or `api`.
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
    """Returns True only if Postgres answers the DSN's TCP probe.

    Used by the `_require_db` fixture to skip pgvector tests when the
    DB is not running (the gate must NOT depend on it).
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
        pytest.skip("Postgres not available — pgvector test skipped")


# =========================================================================
# (1) InlineGrounding.retrieve — order, k, determinism
# =========================================================================


def test_inline_retrieve_returns_chunks_for_top_candidates() -> None:
    """Posterior BPPV + Ménière in the pool ⇒ retrieve returns chunks for
    THOSE diagnoses, in pool order, up to k.
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
    # All returned chunks belong to posterior BPPV or Ménière
    assert all(
        c.diagnosis in {Diagnosis.bppv_posterior, Diagnosis.meniere} for c in result
    )
    # BPPV chunks come before Ménière chunks (pool order)
    seen_meniere = False
    for c in result:
        if c.diagnosis == Diagnosis.meniere:
            seen_meniere = True
        else:
            assert not seen_meniere, "Ménière appeared before posterior BPPV"


def test_inline_retrieve_respects_k_limit() -> None:
    """With k=2 over the BPPV pool (3 chunks) ⇒ exactly 2 chunks."""
    g = InlineGrounding()
    result = g.retrieve(
        _diff((Diagnosis.bppv_posterior, 0.99)),
        CaseFeatures(),
        k=2,
    )
    assert len(result) == 2
    assert all(c.diagnosis == Diagnosis.bppv_posterior for c in result)


def test_inline_retrieve_deterministic_two_calls_equal() -> None:
    """Determinism: two `retrieve()` calls with the same args ⇒ same result."""
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
    """Determinism: two fresh grounding instances ⇒ same result."""
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
    """With no candidates, retrieve returns an empty list (does not break)."""
    g = InlineGrounding()
    result = g.retrieve(DifferentialResult(), CaseFeatures(), k=4)
    assert result == []


def test_inline_retrieve_with_unknown_dx_skips_silently() -> None:
    """A candidate whose diagnosis is not in the CORPUS does not break
    retrieve — it simply contributes no chunks. (Should not happen: the
    DifferentialEngine only emits valid `Diagnosis` values, but the
    interface is defensive.)"""
    g = InlineGrounding()
    # Force a Diagnosis that is NOT in the CORPUS (`undetermined`).
    result = g.retrieve(
        _diff(
            (Diagnosis.undetermined, 0.5),
            (Diagnosis.bppv_posterior, 0.9),
        ),
        CaseFeatures(),
        k=3,
    )
    # Only BPPV contributes; `undetermined` is not in the CORPUS.
    assert all(c.diagnosis == Diagnosis.bppv_posterior for c in result)


def test_inline_satisfies_protocol() -> None:
    """`InlineGrounding` satisfies the `Grounding` Protocol (duck typing)."""
    g: Grounding = InlineGrounding()
    assert hasattr(g, "retrieve")
    assert callable(g.retrieve)


# =========================================================================
# (2) CORPUS coverage — all 8 diagnoses
# =========================================================================


def test_corpus_covers_eight_diagnoses() -> None:
    """The CORPUS covers the 8 diagnoses documented in the T6 spec."""
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
    """Every covered diagnosis has ≥1 chunk."""
    for dx in SUPPORTED_DIAGNOSES:
        assert CORPUS.get(dx), f"no chunks for {dx.value}"


def test_corpus_chunks_have_well_formed_source_ids() -> None:
    """`source_id` follows the `clinibrium-paraphrase:<dx>-<n>` convention."""
    pattern_ok = True
    for dx, chunks in CORPUS.items():
        for i, chunk in enumerate(chunks, start=1):
            assert chunk.source_id == f"clinibrium-paraphrase:{dx.value}-{i}", (
                f"unexpected source_id: {chunk.source_id}"
            )
            assert chunk.diagnosis == dx
            assert pattern_ok
            assert chunk.text  # not empty


def test_corpus_chunks_are_own_paraphrase_marker() -> None:
    """Every chunk must have non-trivial text (>50 chars) — this is a
    smoke test of the amount of paraphrase. The clinical quality audit
    of the paraphrases is task `T-CLIN`; here we only guarantee there
    is original content (no empty placeholders)."""
    for dx, chunks in CORPUS.items():
        for chunk in chunks:
            assert len(chunk.text) > 50, f"chunk for {dx.value} looks empty"
            # Soft heuristic: no chunk repeats the literal ICVD marker
            # (it shouldn't — they are our own paraphrases).
            assert "ICVD" not in chunk.text.upper(), (
                f"chunk for {dx.value} contains literal 'ICVD' — it should be "
                f"a paraphrase"
            )


# =========================================================================
# (3) Factory — get_grounding() with DATABASE_URL=None ⇒ InlineGrounding
# =========================================================================


def test_get_grounding_with_no_database_url_returns_inline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without DATABASE_URL: the factory degrades to `InlineGrounding` (does not break)."""
    from clinibrium.config import get_settings

    monkeypatch.setattr(get_settings(), "DATABASE_URL", None)
    g = get_grounding()
    assert isinstance(g, InlineGrounding)


def test_get_grounding_with_empty_database_url_returns_inline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DATABASE_URL='': the factory also degrades (falsy)."""
    from clinibrium.config import get_settings

    monkeypatch.setattr(get_settings(), "DATABASE_URL", "")
    g = get_grounding()
    assert isinstance(g, InlineGrounding)


def test_get_grounding_with_unreachable_db_returns_inline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DATABASE_URL set but host:port unreachable ⇒ InlineGrounding."""
    from clinibrium.config import get_settings

    # 127.0.0.1:1 is virtually always unreachable
    monkeypatch.setattr(
        get_settings(), "DATABASE_URL", "postgresql://x:x@127.0.0.1:1/x"
    )
    g = get_grounding()
    assert isinstance(g, InlineGrounding)


def test_get_grounding_does_not_raise_on_any_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`get_grounding()` NEVER raises — not with an empty URL nor a malformed DSN."""
    from clinibrium.config import get_settings

    for url in (None, "", "not a url", "postgresql://", "postgresql://x@"):
        monkeypatch.setattr(get_settings(), "DATABASE_URL", url)
        # We only assert it does not raise
        _ = get_grounding()


# =========================================================================
# (4) Embedder — deterministic (same text → same vector)
# =========================================================================


def test_embedder_is_deterministic() -> None:
    """Same text + same dim ⇒ same vector."""
    a = embed_text("vértigo posicional breve con nistagmo torsional")
    b = embed_text("vértigo posicional breve con nistagmo torsional")
    assert a == b


def test_embedder_is_deterministic_across_calls() -> None:
    """100 identical calls ⇒ 100 identical vectors."""
    text = "ataxia troncal severa con nistagmo vertical puro"
    first = embed_text(text)
    for _ in range(100):
        assert embed_text(text) == first


def test_embedder_is_deterministic_independent_process_objects() -> None:
    """Determinism via the same text at two distant moments (same
    process) — confirms there is no hidden state."""
    v1 = embed_text("hipoacusia fluctuante con tinnitus y plenitud aural")
    # ... (imagine client code in between) ...
    v2 = embed_text("hipoacusia fluctuante con tinnitus y plenitud aural")
    assert v1 == v2


def test_embedder_default_dim_is_256() -> None:
    assert EMBED_DIM == 256
    v = embed_text("cualquier texto clínico de al menos tres palabras")
    assert len(v) == 256


def test_embedder_returns_l2_normalized_vector() -> None:
    """The vector is L2-normalized: ||v||_2 ≈ 1 (except for empty text)."""
    import math

    v = embed_text("vértigo continuo espontáneo con náuseas y vómitos")
    norm = math.sqrt(sum(c * c for c in v))
    assert 0.99 <= norm <= 1.01


def test_embedder_empty_text_returns_zero_vector() -> None:
    """Empty text / stopwords only ⇒ zero vector (not normalized)."""
    v_empty = embed_text("")
    v_stop = embed_text("de la el y a")
    assert v_empty == [0.0] * EMBED_DIM
    assert v_stop == [0.0] * EMBED_DIM


def test_embedder_different_texts_produce_different_vectors() -> None:
    """Different texts produce different vectors (sanity)."""
    a = embed_text("vértigo posicional breve con nistagmo torsional")
    b = embed_text("hipoacusia fluctuante con tinnitus y plenitud aural")
    assert a != b


def test_embedder_tokenize_basic() -> None:
    """Tokenizer: lowercase, stopword drop, drop of tokens <3 chars."""
    toks = _tokenize("El vértigo POSICIONAL es breve y se asocia a nistagmo")
    assert "vértigo" in toks
    assert "posicional" in toks
    assert "breve" in toks
    assert "asocia" in toks
    assert "nistagmo" in toks
    # Stopwords filtered out
    assert "el" not in toks
    assert "y" not in toks
    assert "a" not in toks


def test_build_query_text_includes_candidates_and_features() -> None:
    """`build_query_text` composes structured features + top diagnoses."""
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
    # The `trigger=positional_head` feature appears by name
    assert "positional_head" in q or "trigger" in q


# =========================================================================
# (5) pgvector tests — skipped when no DB is available
# =========================================================================


def test_pgvector_ingest_and_retrieve(require_db: None) -> None:
    """With a DB: ingest writes the CORPUS and retrieve returns chunks
    relevant to the query. Skipped when no DB is available.
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
    assert n > 0, "ingest wrote no rows"
    assert len(chunks) > 0, "retrieve returned no chunks"
    # The chunks come from the CORPUS (any of the 8 diagnoses is
    # valid, but the search space is over the indexed universe)
    assert all(isinstance(c, GroundingChunk) for c in chunks)
    assert all(c.source_id.startswith("clinibrium-paraphrase:") for c in chunks)


def test_pgvector_ingest_is_idempotent(require_db: None) -> None:
    """`ingest()` can be called multiple times: the table stays consistent
    with the CORPUS (same row count, same source_ids)."""
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
    # Total CORPUS chunks (≥8, the 8 diagnoses with 1+ each)
    assert n1 >= 8


def test_pgvector_retrieve_degrades_when_db_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PgvectorGrounding.retrieve against an unreachable DB ⇒ [] (does not break)."""
    g = PgvectorGrounding("postgresql://x:x@127.0.0.1:1/x")
    chunks = g.retrieve(  # sync entry point (Protocol)
        _diff((Diagnosis.bppv_posterior, 0.9)),
        CaseFeatures(),
        k=4,
    )
    assert chunks == []


# =========================================================================
# (6) INV — grounding only imports contracts (+ config + DB driver).
# Verified via AST (as in other tests in this repo).
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
    """The `grounding` package ONLY imports from `clinibrium.contracts` and
    `clinibrium.config` (settings) — never engines, reasoner, orchestrator,
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
        "grounding imports forbidden modules:\n  "
        + "\n  ".join(offenders)
    )


def test_grounding_does_not_call_claude_or_set_diagnosis() -> None:
    """Sanity: the module does NOT import the Anthropic SDK nor define
    binding diagnosis fields. It is retrieval only.
    """
    pkg_root = Path(inspect.getfile(InlineGrounding)).parent
    forbidden_libs = ("anthropic", "openai", "google.generativeai")
    offenders: list[str] = []
    for py in sorted(pkg_root.glob("*.py")):
        for lineno, mod in _iter_imports(py):
            if any(mod == bad or mod.startswith(bad + ".") for bad in forbidden_libs):
                offenders.append(f"{py.name}:{lineno} → {mod}")
    assert not offenders, (
        "grounding must not call LLM providers:\n  "
        + "\n  ".join(offenders)
    )
