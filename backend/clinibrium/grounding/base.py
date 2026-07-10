"""Interfaz `Grounding` + tipo `GroundingChunk`.

AD-10: el reasoner consume grounding **vía interfaz** (no pgvector
directo). Esto permite que la implementación degrade elegante a
`InlineGrounding` cuando pgvector no está disponible, sin que el
reasoner tenga que conocer el detalle de la implementación.

Reglas de diseño:

- `GroundingChunk.text` es siempre **paráfrasis propia** del equipo
  (AD-5). No se acepta texto verbatim de ICVD.
- `GroundingChunk.diagnosis` es opcional: un chunk puede ser genérico
  (p.ej. "red flags en AVS") o específico de un diagnóstico.
- `GroundingChunk.source_id` identifica al chunk para trazabilidad
  (AuditEvent, debugging, evaluación de retrieval). Convención:
  `clinibrium-paraphrase:<diagnóstico>-<n>`.
- `Grounding.retrieve(...)` es **determinista** para una misma
  `(candidates, features, k)`. Sin random, sin reloj.
"""
from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict

from clinibrium.contracts import CaseFeatures, Diagnosis, DifferentialResult


class GroundingChunk(BaseModel):
    """Snippet de criterio clínico (paráfrasis propia) que el reasoner
    consume como contexto.

    Atributos:
        text:        Paráfrasis ORIGINAL del equipo (NO verbatim ICVD).
        diagnosis:   Diagnóstico al que aplica (None = genérico / cross-cutting).
        source_id:   Identificador trazable del chunk. Convención:
                     `clinibrium-paraphrase:<diagnóstico>-<n>`.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    diagnosis: Diagnosis | None = None
    source_id: str


class Grounding(Protocol):
    """Protocol de retrieval de chunks de criterios.

    Una implementación (`InlineGrounding`, `PgvectorGrounding`) devuelve
    hasta `k` chunks relevantes para los diagnósticos candidatos de un
    `DifferentialResult`, opcionalmente ponderados por las features
    presentes en el caso.
    """

    def retrieve(
        self,
        candidates: DifferentialResult,
        features: CaseFeatures,
        k: int = 4,
    ) -> list[GroundingChunk]:
        ...
