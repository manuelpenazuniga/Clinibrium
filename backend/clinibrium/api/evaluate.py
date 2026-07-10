"""Endpoint `POST /api/evaluate` — Server-Sent Events del pipeline VertigoDx.

CABLEADO sobre `orchestrator.evaluate` (mapa: api → orchestrator, contracts).
PROHIBIDO importar engines / reasoner / rails / grounding directo: todo pasa
por el orchestrator.

Diseño (v7.3 §10 demo):
    - Body: `CaseFeatures` (Pydantic valida; inválido → 422).
    - `recording_mode` se lee de `Settings` (AD-6 — server-side only).
    - `asyncio.Queue` puentea el `on_stage` del orchestrator con el generador
      SSE. Cada evento se serializa como `event: <stage>\ndata: <json>\n\n`.
    - Al terminar `evaluate()`: `event: done\ndata: <PipelineResult>\n\n`.
    - Si `evaluate()` levanta: `event: error\ndata: {...}\n\n` y cerrá
      (el `AuditEvent` de error ya lo emitió el orchestrator, INV-4).

AD-6 / regla dura 2: el LLM NO fija urgencia vinculante; el hook SSE es
puramente observacional.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from clinibrium.config import get_settings
from clinibrium.contracts.features import CaseFeatures
from clinibrium.contracts.results import PipelineResult
from clinibrium.orchestrator import evaluate as orchestrator_evaluate

logger = logging.getLogger(__name__)

router = APIRouter()


def _sse(event: str, data: Any) -> bytes:
    """Serializa un evento SSE. `data` debe ser JSON-serializable."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n".encode("utf-8")


def _serialize_result(result: PipelineResult) -> dict[str, Any]:
    """Serializa un `PipelineResult` para el evento `done`.

    `model_dump(mode="json")` resuelve enums (`Urgency`, `ForcedAction`,
    `Diagnosis`) a su `.value` (string) y `set[...]` a `list`. `default=str`
    es un seguro por si cuela un `datetime` u otro tipo no nativo.
    """
    return result.model_dump(mode="json", exclude_none=False)


async def _stream_pipeline(features: CaseFeatures) -> AsyncIterator[bytes]:
    """Generador SSE: emite un evento por cada stage del pipeline.

    Crea una `asyncio.Queue`, lanza `evaluate()` como task de fondo con
    `on_stage` que empuja eventos a la cola, y drena la cola hasta recibir
    el sentinel `None` (task terminada).
    """
    recording_mode = get_settings().RECORDING_MODE
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def on_stage(stage: str, payload: Any) -> None:
        # El hook es awaited por el orchestrator — al suspenderse en
        # `queue.put(...)` no bloquea a nadie más que a este generador.
        await queue.put({"event": stage, "data": payload})

    async def run() -> None:
        try:
            result = await orchestrator_evaluate(
                features,
                recording_mode=recording_mode,
                on_stage=on_stage,
            )
            await queue.put({"event": "done", "data": _serialize_result(result)})
        except Exception as exc:  # noqa: BLE001 — el orchestrator ya emitió AuditEvent de error
            logger.exception("SSE: orchestrator.evaluate falló — emitiendo event: error")
            await queue.put(
                {
                    "event": "error",
                    "data": {
                        "error": type(exc).__name__,
                        "message": str(exc),
                    },
                }
            )
        finally:
            # Sentinel: garantiza que el generador cierra aunque la cola
            # quede vacía por error de programación.
            await queue.put(None)

    task = asyncio.create_task(run())

    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield _sse(item["event"], item["data"])
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass


@router.post("/api/evaluate")
async def evaluate_endpoint(features: CaseFeatures) -> StreamingResponse:
    """Evalúa un caso clínico y streamea el pipeline por SSE.

    Body: `CaseFeatures` (Pydantic valida → 422 si inválido o con campos extra).
    Response: `text/event-stream` con eventos `redflag`, `differential`,
    `ml`, `reasoning`, `rails` y `done` (o `error` ante fallo inesperado).

    `recording_mode` NUNCA se lee del body — se toma de `Settings`
    (AD-6 / regla dura 2).
    """
    return StreamingResponse(
        _stream_pipeline(features),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # desactiva buffering en proxies (nginx)
        },
    )
