"""`POST /api/evaluate` endpoint — Server-Sent Events for the VertigoDx pipeline.

WIRED on top of `orchestrator.evaluate` (map: api → orchestrator, contracts).
Importing engines / reasoner / rails / grounding directly is FORBIDDEN:
everything goes through the orchestrator.

Design (v7.3 §10 demo):
    - Body: `CaseFeatures` (Pydantic validates; invalid → 422).
    - `recording_mode` is read from `Settings` (AD-6 — server-side only).
    - An `asyncio.Queue` bridges the orchestrator's `on_stage` with the SSE
      generator. Each event is serialized as `event: <stage>\ndata: <json>\n\n`.
    - When `evaluate()` finishes: `event: done\ndata: <PipelineResult>\n\n`.
    - If `evaluate()` raises: `event: error\ndata: {...}\n\n` and close
      (the error `AuditEvent` was already emitted by the orchestrator, INV-4).

AD-6 / hard rule 2: the LLM NEVER sets binding urgency; the SSE hook is
purely observational.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from clinibrium.config import get_settings
from clinibrium.contracts.features import CaseFeatures
from clinibrium.contracts.results import PipelineResult
from clinibrium.fhir import bundle_sha256, to_bundle
from clinibrium.orchestrator import evaluate as orchestrator_evaluate

logger = logging.getLogger(__name__)

router = APIRouter()


def _sse(event: str, data: Any) -> bytes:
    """Serializes an SSE event. `data` must be JSON-serializable."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n".encode("utf-8")


def _serialize_result(result: PipelineResult) -> dict[str, Any]:
    """Serializes a `PipelineResult` for the `done` event.

    `model_dump(mode="json")` resolves enums (`Urgency`, `ForcedAction`,
    `Diagnosis`) to their `.value` (string) and `set[...]` to `list`.
    `default=str` is a safety net in case a `datetime` or other non-native
    type slips through.
    """
    return result.model_dump(mode="json", exclude_none=False)


async def _stream_pipeline(
    features: CaseFeatures,
    *,
    kill_reasoner: bool = False,
) -> AsyncIterator[bytes]:
    """SSE generator: emits one event per pipeline stage.

    Creates an `asyncio.Queue`, launches `evaluate()` as a background task
    whose `on_stage` pushes events onto the queue, and drains the queue until
    the `None` sentinel is received (task finished).
    """
    recording_mode = get_settings().RECORDING_MODE
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def on_stage(stage: str, payload: Any) -> None:
        await queue.put({"event": stage, "data": payload})

    async def run() -> None:
        try:
            result = await orchestrator_evaluate(
                features,
                recording_mode=recording_mode,
                on_stage=on_stage,
                kill_reasoner=kill_reasoner,
            )
            done_data = _serialize_result(result)
            if result.audit_event is not None:
                fhir_bundle = to_bundle(
                    result, features, result.audit_event
                )
                done_data["fhir_bundle"] = fhir_bundle
                done_data["bundle_sha256"] = bundle_sha256(fhir_bundle)
            await queue.put({"event": "done", "data": done_data})
        except Exception as exc:  # noqa: BLE001 — the orchestrator already emitted the error AuditEvent
            logger.exception("SSE: orchestrator.evaluate failed — emitting event: error")
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
            # Sentinel: guarantees the generator closes even if the queue
            # is left empty due to a programming error.
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
async def evaluate_endpoint(
    features: CaseFeatures,
    debug_kill_reasoner: bool = Query(False),
) -> StreamingResponse:
    """Evaluates a clinical case and streams the pipeline via SSE.

    Body: `CaseFeatures` (Pydantic validates → 422 if invalid or with extra fields).
    Response: `text/event-stream` with events `redflag`, `differential`,
    `ml`, `reasoning`, `rails` and `done` (or `error` on unexpected failure).

    Query params:
        debug_kill_reasoner: if True, forces reasoning=None (intentional INV-8,
            controlled degradation). Does NOT affect urgency or red flags.

    `recording_mode` is NEVER read from the body — it comes from `Settings`
    (AD-6 / hard rule 2).
    """
    return StreamingResponse(
        _stream_pipeline(features, kill_reasoner=debug_kill_reasoner),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disables buffering in proxies (nginx)
        },
    )
