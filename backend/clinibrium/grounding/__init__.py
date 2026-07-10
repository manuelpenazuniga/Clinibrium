"""Interfaz pública del módulo `grounding` (RAG de criterios clínicos).

Hoja del grafo `clinibrium.*` (AD-10): este paquete SOLO importa de
`clinibrium.contracts` + libs + `clinibrium.config` (settings) + su
propia conexión a Postgres/pgvector. NO importa `reasoner`, los motores
(`redflag_engine`, `differential_engine`), `orchestrator`, `rails` ni
`api`. Esa frontera es la que le permite al reasoner (T7) consumir el
grounding **a través de una interfaz** y degradar elegante a `rag_inline`
cuando pgvector no está disponible.

AD-5 / regla dura 3: el corpus RAG es por **PARÁFRASIS PROPIA** de los
criterios ICVD — NUNCA texto verbatim (ICVD es CC BY-NC; el texto
restringido no se usa, las reglas como hechos sí son re-escribibles).
Las paráfrasis viven en `inline.CORPUS` y se documentan allí como
autoría propia del equipo.

NO llama a Claude y NO fija diagnósticos: provee *chunks* (snippets de
criterios) para que el reasoner fundamente su explicación.
"""
from __future__ import annotations

from clinibrium.grounding.base import Grounding, GroundingChunk
from clinibrium.grounding.factory import get_grounding
from clinibrium.grounding.inline import InlineGrounding

__all__ = ["Grounding", "GroundingChunk", "InlineGrounding", "get_grounding"]
