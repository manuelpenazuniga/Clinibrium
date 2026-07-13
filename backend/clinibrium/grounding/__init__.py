"""Public interface of the `grounding` module (clinical criteria RAG).

Leaf of the `clinibrium.*` graph (AD-10): this package ONLY imports from
`clinibrium.contracts` + libs + `clinibrium.config` (settings) + its own
Postgres/pgvector connection. It does NOT import `reasoner`, the engines
(`redflag_engine`, `differential_engine`), `orchestrator`, `rails` or
`api`. That boundary is what lets the reasoner (T7) consume the
grounding **through an interface** and degrade gracefully to `rag_inline`
when pgvector is unavailable.

AD-5 / hard rule 3: the RAG corpus is built from **OUR OWN PARAPHRASE**
of the ICVD criteria — NEVER verbatim text (ICVD is CC BY-NC; the
restricted text is not used, while the rules as facts are rewritable).
The paraphrases live in `inline.CORPUS` and are documented there as the
team's own authorship.

Does NOT call Claude and does NOT set diagnoses: it provides *chunks*
(criteria snippets) for the reasoner to ground its explanation.
"""
from __future__ import annotations

from clinibrium.grounding.base import Grounding, GroundingChunk
from clinibrium.grounding.factory import get_grounding
from clinibrium.grounding.inline import InlineGrounding

__all__ = ["Grounding", "GroundingChunk", "InlineGrounding", "get_grounding"]
