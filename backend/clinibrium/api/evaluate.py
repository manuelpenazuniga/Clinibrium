"""`POST /api/evaluate` endpoint — Server-Sent Events for the Clinibrium pipeline.

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
from typing import Any, AsyncIterator, Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from clinibrium.config import get_settings
from clinibrium.contracts.features import CaseFeatures
from clinibrium.contracts.results import PipelineResult
from clinibrium.fhir import bundle_sha256, to_bundle
from clinibrium.i18n import localize_redflag_label
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


def _localize_done(done_data: dict[str, Any], lang: str) -> None:
    """Localize the clinician-facing labels in the serialized `done` payload.

    PRESENTATION ONLY (in place, on the serialized dict — never the model):
    swaps each red-flag hit `label` for its English rendering, keyed by the
    STABLE hit `id`. Spanish (default) is a no-op, so the recorded demo is
    byte-identical. The enum values (`urgency`, `forced_actions`, `id`,
    `severity`) are untouched, and the FHIR bundle is built from the ORIGINAL
    model (Spanish), so this does not touch the clinical artifact.
    """
    if lang != "en":
        return
    red_flag = done_data.get("red_flag")
    if not isinstance(red_flag, dict):
        return
    for hit in red_flag.get("hits", []):
        if isinstance(hit, dict) and "id" in hit and "label" in hit:
            hit["label"] = localize_redflag_label(hit["id"], hit["label"], "en")


async def _stream_pipeline(
    features: CaseFeatures,
    *,
    kill_reasoner: bool = False,
    lang: str = "es",
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
                lang=lang,
            )
            done_data = _serialize_result(result)
            if result.audit_event is not None:
                # FHIR is built from the ORIGINAL model BEFORE the
                # presentation-only label localization below. Deterministic
                # content (labels, summary, enums) is canonical Spanish; the
                # reasoner prose keeps the language it was requested in and
                # the ClinicalImpression is tagged `language: "en"` in that
                # case (AD-19 precision — the es bundle is byte-identical).
                fhir_bundle = to_bundle(
                    result, features, result.audit_event
                )
                done_data["fhir_bundle"] = fhir_bundle
                done_data["bundle_sha256"] = bundle_sha256(fhir_bundle)
            _localize_done(done_data, lang)
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
    lang: Literal["es", "en"] = Query("es"),
) -> StreamingResponse:
    """Evaluates a clinical case and streams the pipeline via SSE.

    Body: `CaseFeatures` (Pydantic validates → 422 if invalid or with extra fields).
    Response: `text/event-stream` with events `redflag`, `differential`,
    `ml`, `reasoning`, `rails` and `done` (or `error` on unexpected failure).

    Query params:
        debug_kill_reasoner: if True, forces reasoning=None (intentional INV-8,
            controlled degradation). Does NOT affect urgency or red flags.
        lang: UI language for clinician-facing PRESENTATION only ("es" default |
            "en"). It parameterizes the reasoner's output language and swaps
            red-flag hit labels at serialization; it NEVER reaches the ML engine,
            never enters `CaseFeatures`/`NETWORK_SAFE_FIELDS`, and never changes
            any safety decision, enum value or deterministic FHIR content. The
            reasoner prose embedded in the FHIR ClinicalImpression follows `lang`
            (the resource is tagged with FHIR `language` when English), so the
            bundle hash differs between languages when the reasoner answered.

    `recording_mode` is NEVER read from the body — it comes from `Settings`
    (AD-6 / hard rule 2). `lang` follows the same "never from the body" pattern
    but comes from the query string, not Settings.

    P1.4: `debug_kill_reasoner` is a demo/debug backdoor. It is only honored when
    `DEMO_MODE` or `RECORDING_MODE` is set server-side; otherwise it returns 403
    (the backdoor is not exposed in a normal/public deployment).
    """
    if debug_kill_reasoner:
        settings = get_settings()
        if not (settings.DEMO_MODE or settings.RECORDING_MODE):
            raise HTTPException(
                status_code=403,
                detail=(
                    "debug_kill_reasoner is only available in demo/recording mode "
                    "(set DEMO_MODE=true or RECORDING_MODE=true)"
                ),
            )
    return StreamingResponse(
        _stream_pipeline(features, kill_reasoner=debug_kill_reasoner, lang=lang),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disables buffering in proxies (nginx)
        },
    )
