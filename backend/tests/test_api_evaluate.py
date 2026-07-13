"""Tests for the SSE endpoint `POST /api/evaluate` — T9b (streaming pipeline).

Covers the acceptance criteria of task T9b:
  - 200 text/event-stream with redflag, differential, ml, reasoning, rails, done events
  - 422 on invalid body (extra field — including `recording_mode` — or bad type)
  - Case with active red flag → urgency=inmediata propagated to the `rails` and `done` events
  - reasoner/ml mocked (zero real network calls in the gate)
  - `on_stage` does NOT break INV-4: hook that raises → evaluate completes + 1 AuditEvent
  - api respects the module map: only orchestrator + contracts (no
    engines/reasoner/rails/grounding/ml_client)
  - `recording_mode` NEVER comes from the body (verifiable: CaseFeatures schema without
    that field + handler signature)
"""
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from clinibrium.api import create_app
from clinibrium.audit.engine import build_audit_event
from clinibrium.config import get_settings
from clinibrium.contracts import (
    CaseFeatures,
    DixHallpikeResult,
    FocalSign,
    HeadImpulse,
    NystagmusDirection,
    Onset,
    SymptomDuration,
    TimingPattern,
    Trigger,
    Urgency,
    VascularRiskFactor,
)
from clinibrium.grounding import InlineGrounding
from clinibrium.orchestrator import evaluate as orchestrator_evaluate

# =============================================================================
# Helpers
# =============================================================================


def _bppv_benign() -> dict:
    """Typical posterior BPPV: positional, torsional, Dix-Hallpike (+).

    Same features as `test_orchestrator._bppv_benign`, serialized as a
    dict (what arrives over HTTP).
    """
    return {
        "duration": SymptomDuration.under_1min.value,
        "onset": Onset.sudden.value,
        "trigger": Trigger.positional_head.value,
        "timing_pattern": TimingPattern.episodic_triggered.value,
        "nystagmus_direction": NystagmusDirection.mixed.value,
        "nystagmus_latency_s": 2.0,
        "nystagmus_duration_s": 20.0,
        "nystagmus_fatigable": True,
        "nystagmus_suppressed_by_fixation": True,
        "dix_hallpike": DixHallpikeResult.right_positive.value,
        "torsion_confirmed_by_clinician": True,
        "episode_count": 3,
        "episode_duration": SymptomDuration.under_1min.value,
    }


def _red_flag_case() -> dict:
    """Central AVS: abnormal HINTS + focal signs + age > 60 + vascular risk."""
    return {
        "timing_pattern": TimingPattern.acute_continuous.value,
        "head_impulse": HeadImpulse.normal.value,
        "skew_deviation": True,
        "nystagmus_direction": NystagmusDirection.vertical_pure.value,
        "focal_signs": [FocalSign.dysarthria.value],
        "age_years": 65,
        "vascular_risk_factors": [VascularRiskFactor.hypertension.value],
    }


def _mock_ml_and_reasoner(monkeypatch) -> None:
    """Mock ml + reasoner so the gate makes zero real network calls."""
    monkeypatch.setattr("clinibrium.ml_client.predict", AsyncMock(return_value=None))
    monkeypatch.setattr("clinibrium.reasoner.reason", AsyncMock(return_value=None))


def _capture_audit(monkeypatch) -> list:
    """Capture every call to the audit engine's `emit()`.

    Returns the list being filled — by the end of the test it holds all
    the AuditEvents the orchestrator emitted.
    """
    events: list = []

    async def _fake_emit(*args, **kwargs):
        event = build_audit_event(*args, **kwargs)
        events.append(event)
        return event

    monkeypatch.setattr("clinibrium.audit.engine.emit", _fake_emit)
    return events


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    """Split the SSE stream into ordered (event, data) pairs.

    The expected per-event format is:
        event: <name>\\ndata: <json>\\n\\n
    (json.dumps produces JSON without spaces by default — json.loads accepts it).
    """
    out: list[tuple[str, dict]] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        ev_name: str | None = None
        ev_data: str | None = None
        for line in block.splitlines():
            if line.startswith("event:"):
                ev_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                ev_data = line[len("data:") :].strip()
        if ev_name is not None and ev_data is not None:
            out.append((ev_name, json.loads(ev_data)))
    return out


async def _post_evaluate(
    app, payload: dict
) -> tuple[int, dict, list[tuple[str, dict]]]:
    """POST to /api/evaluate and return (status, headers, parsed events)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream("POST", "/api/evaluate", json=payload) as resp:
            status = resp.status_code
            ctype = resp.headers.get("content-type", "")
            text = ""
            async for chunk in resp.aiter_text():
                text += chunk
    return status, {"content-type": ctype}, _parse_sse(text)


# =============================================================================
# Happy path: benign BPPV → 200 text/event-stream + done with audit_event_id
# =============================================================================


async def test_evaluate_sse_bppv_benign_streams_all_stages_and_done(monkeypatch):
    """POST /api/evaluate with benign BPPV → 200 text/event-stream + done."""
    _mock_ml_and_reasoner(monkeypatch)
    app = create_app()

    status, headers, events = await _post_evaluate(app, _bppv_benign())

    assert status == 200
    assert headers["content-type"].startswith("text/event-stream")

    names = [n for n, _ in events]
    # The demo (v7.3 §10) shows the pipeline "thinking" as a stream,
    # in this strict order: redflag → differential → ml → reasoning → rails → done.
    assert names == [
        "redflag",
        "differential",
        "ml",
        "reasoning",
        "rails",
        "done",
    ], f"unexpected order/presence: {names}"

    # done payload: audit_event_id populated + ambulatory urgency (benign BPPV).
    done_payload = events[-1][1]
    assert done_payload["audit_event_id"] is not None
    assert done_payload["urgency"] == Urgency.ambulatoria.value
    # Auditable FHIR artifact included in the done event.
    bundle = done_payload["fhir_bundle"]
    assert bundle["resourceType"] == "Bundle"
    assert any(
        e["resource"]["resourceType"] == "AuditEvent" for e in bundle["entry"]
    )
    # bundle_sha256 present and with the correct length (tamper-evident).
    sha = done_payload["bundle_sha256"]
    assert isinstance(sha, str)
    assert len(sha) == 64
    assert all(c in "0123456789abcdef" for c in sha)

    # redflag and rails payloads: correct de-identified shape.
    redflag_payload = events[0][1]
    assert redflag_payload["red_flag_activa"] is False
    assert redflag_payload["hits_count"] == 0

    rails_payload = events[4][1]
    assert rails_payload["urgency"] == Urgency.ambulatoria.value
    assert "forced_actions" in rails_payload
    assert "applied_rails" in rails_payload


# =============================================================================
# Active red flag → urgency=inmediata propagated to the SSE
# =============================================================================


async def test_evaluate_sse_red_flag_urgency_inmediata_propagated(monkeypatch):
    """Active red flag → urgency=inmediata in the `rails` and `done` events."""
    _mock_ml_and_reasoner(monkeypatch)
    app = create_app()

    status, headers, events = await _post_evaluate(app, _red_flag_case())

    assert status == 200
    assert headers["content-type"].startswith("text/event-stream")
    names = [n for n, _ in events]
    assert "rails" in names and "done" in names

    rails_payload = events[names.index("rails")][1]
    assert rails_payload["urgency"] == Urgency.inmediata.value

    done_payload = events[names.index("done")][1]
    assert done_payload["urgency"] == Urgency.inmediata.value
    assert done_payload["red_flag"]["red_flag_activa"] is True
    assert done_payload["audit_event_id"] is not None


# =============================================================================
# 422 — invalid body (extra field / bad type)
# =============================================================================


async def test_evaluate_sse_invalid_body_extra_field_returns_422(monkeypatch):
    """Body with fields outside the allowlist (e.g. `recording_mode`) → 422.

    Demonstrates AD-6 from the HTTP side: the client canNOT sneak
    `recording_mode` or PII into the body.
    """
    _mock_ml_and_reasoner(monkeypatch)
    app = create_app()

    payload = _bppv_benign()
    payload["recording_mode"] = True  # AD-6: NEVER comes from the body
    payload["patient_name"] = "Juan Pérez"  # PII forbidden by INV-2

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/evaluate", json=payload)
    assert resp.status_code == 422


async def test_evaluate_sse_invalid_body_bad_type_returns_422(monkeypatch):
    """Body with a bad type (string instead of int) → 422."""
    _mock_ml_and_reasoner(monkeypatch)
    app = create_app()

    payload = _bppv_benign()
    payload["age_years"] = "old"  # int expected, string sent

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/evaluate", json=payload)
    assert resp.status_code == 422


# =============================================================================
# AD-6: recording_mode is NOT in the handler signature (server-side only)
# =============================================================================


def test_evaluate_endpoint_signature_has_no_recording_mode(monkeypatch):
    """The handler does NOT accept `recording_mode` from the body — it is read
    from Settings (server-side, AD-6). `debug_kill_reasoner` is a legitimate demo query param."""
    from clinibrium.api.evaluate import evaluate_endpoint

    sig = inspect.signature(evaluate_endpoint)
    assert "recording_mode" not in sig.parameters, (
        "AD-6 violated: recording_mode must NEVER be an endpoint parameter. "
        f"Parameters: {list(sig.parameters)}"
    )
    assert "features" in sig.parameters
    assert "debug_kill_reasoner" in sig.parameters


# =============================================================================
# on_stage does NOT break INV-4
# =============================================================================


async def test_on_stage_hook_raising_does_not_break_inv4(monkeypatch):
    """Hook that raises → evaluate completes and emits EXACTLY 1 AuditEvent.

    Additional guarantee on top of the existing INV-4: even if the observer
    fails, the audit trail and the pipeline result are unaffected.
    """
    _mock_ml_and_reasoner(monkeypatch)
    events = _capture_audit(monkeypatch)

    async def _raising_hook(stage: str, payload: dict) -> None:
        raise RuntimeError("hook explosion!")

    features = CaseFeatures(**_bppv_benign())
    result = await orchestrator_evaluate(
        features,
        grounding=InlineGrounding(),
        on_stage=_raising_hook,
    )

    # Pipeline completes, correct urgency, audit_event_id populated.
    assert result.audit_event_id is not None
    assert result.urgency == Urgency.ambulatoria
    # INV-4: EXACTLY 1 AuditEvent (the successful one, NOT a 2nd error
    # event produced by the hook's raise).
    assert len(events) == 1
    assert events[0].outcome == "evaluation"
    assert events[0].id == result.audit_event_id


async def test_on_stage_hook_receives_all_five_stages_in_order(monkeypatch):
    """on_stage is invoked at least 5 times, in order, for the 5 stages."""
    _mock_ml_and_reasoner(monkeypatch)

    seen: list[str] = []

    async def _hook(stage: str, payload: dict) -> None:
        seen.append(stage)

    features = CaseFeatures(**_bppv_benign())
    await orchestrator_evaluate(
        features,
        grounding=InlineGrounding(),
        on_stage=_hook,
    )

    # The spec requires these 5 stages in this strict order.
    assert seen == ["redflag", "differential", "ml", "reasoning", "rails"], (
        f"unexpected stages: {seen}"
    )


async def test_on_stage_hook_receives_serializable_payloads(monkeypatch):
    """Every stage payload is JSON-serializable (de-identified)."""
    _mock_ml_and_reasoner(monkeypatch)

    payloads: dict[str, dict] = {}

    async def _hook(stage: str, payload: dict) -> None:
        payloads[stage] = payload

    features = CaseFeatures(**_bppv_benign())
    await orchestrator_evaluate(
        features,
        grounding=InlineGrounding(),
        on_stage=_hook,
    )

    # Every payload must be JSON-serializable (no datetime, no set, no PII).
    for stage, payload in payloads.items():
        try:
            json.dumps(payload)
        except (TypeError, ValueError) as exc:
            pytest.fail(f"payload of {stage!r} not JSON-serializable: {exc} — {payload!r}")

    # Expected shape per stage.
    assert set(payloads["redflag"].keys()) == {"red_flag_activa", "hits_count"}
    assert "top_candidates" in payloads["differential"]
    assert payloads["ml"] == {"available": False}
    assert payloads["reasoning"] == {"available": False, "model_used": None}
    assert set(payloads["rails"].keys()) == {"urgency", "forced_actions", "applied_rails"}


# =============================================================================
# Module map: api does NOT import engines / reasoner / rails / grounding / ml_client directly
# =============================================================================


def _iter_clinibrium_imports(py_file: Path) -> list[tuple[int, str]]:
    """Return (line_no, module) for every import of `clinibrium.*`."""
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


def test_api_does_not_import_engines_reasoner_rails_grounding_ml_client():
    """Module map: `api → orchestrator, contracts`. Importing engines / reasoner /
    rails / grounding / ml_client directly is FORBIDDEN. Everything goes through
    the orchestrator."""
    forbidden = {
        "clinibrium.redflag_engine",
        "clinibrium.differential_engine",
        "clinibrium.reasoner",
        "clinibrium.rails",
        "clinibrium.grounding",
        "clinibrium.ml_client",
    }
    pkg_root = Path(__file__).resolve().parents[1] / "clinibrium" / "api"
    offenders: list[str] = []
    for py in sorted(pkg_root.glob("*.py")):
        for lineno, mod in _iter_clinibrium_imports(py):
            if mod in forbidden or any(
                mod == f or mod.startswith(f + ".") for f in forbidden
            ):
                offenders.append(f"api/{py.name}:{lineno} → {mod}")
    assert not offenders, (
        "api imports forbidden modules (must go through orchestrator):\n  "
        + "\n  ".join(offenders)
    )


# =============================================================================
# Settings — recording_mode is read server-side, not from the body
# =============================================================================


def test_settings_has_recording_mode_server_side(monkeypatch):
    """`Settings.RECORDING_MODE` is server-side (env / .env) — the endpoint
    reads it via `get_settings().RECORDING_MODE`, NEVER from the body."""
    assert hasattr(get_settings(), "RECORDING_MODE")
    assert isinstance(get_settings().RECORDING_MODE, bool)


# =============================================================================
# kill_reasoner — debug_kill_reasoner query param (intentional INV-8)
# =============================================================================


async def test_evaluate_sse_debug_kill_reasoner_yields_reasoning_none(monkeypatch):
    """debug_kill_reasoner=true → done event with reasoning=None, reasoner not called."""
    import clinibrium.config as _cfg

    # P1.4: the Kill-Claude backdoor requires demo/recording mode.
    monkeypatch.setattr(_cfg, "_settings", _cfg.Settings(DEMO_MODE=True))
    mock_reasoner = AsyncMock()
    monkeypatch.setattr("clinibrium.reasoner.reason", mock_reasoner)
    monkeypatch.setattr("clinibrium.ml_client.predict", AsyncMock(return_value=None))
    events_capture = _capture_audit(monkeypatch)

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream(
            "POST", "/api/evaluate?debug_kill_reasoner=true",
            json=_bppv_benign(),
        ) as resp:
            text = ""
            async for chunk in resp.aiter_text():
                text += chunk

    assert resp.status_code == 200
    parsed = _parse_sse(text)
    names = [n for n, _ in parsed]
    assert "done" in names

    done = parsed[names.index("done")][1]
    assert done["reasoning"] is None
    assert done["urgency"] == Urgency.ambulatoria.value
    assert "fhir_bundle" in done
    assert "bundle_sha256" in done
    assert len(done["bundle_sha256"]) == 64

    mock_reasoner.assert_not_called()
    assert len(events_capture) == 1
    assert events_capture[0].reasoner_status == "degraded"


async def test_evaluate_debug_kill_reasoner_forbidden_in_normal_mode(monkeypatch):
    """P1.4: without DEMO_MODE/RECORDING_MODE, the Kill-Claude backdoor → 403."""
    import clinibrium.config as _cfg

    monkeypatch.setattr(_cfg, "_settings", _cfg.Settings(DEMO_MODE=False, RECORDING_MODE=False))

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/evaluate?debug_kill_reasoner=true", json=_bppv_benign()
        )
    assert resp.status_code == 403
    # sin el flag de debug, la evaluación normal sigue funcionando
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream("POST", "/api/evaluate", json=_bppv_benign()) as r2:
            async for _ in r2.aiter_text():
                pass
        assert r2.status_code == 200


# =============================================================================
# POST /api/decision — human intervention recorded (AD-4)
# =============================================================================


async def test_decision_rejects_pii_reason(monkeypatch):
    """P1.3: a reason that looks like it carries PII (RUT/id) → 422 (fail-closed)."""
    from unittest.mock import AsyncMock

    monkeypatch.setattr(
        "clinibrium.audit.engine.persist_audit", AsyncMock(return_value=None)
    )
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for bad in ("Paciente RUT 12.345.678-5", "contacto juan@mail.com", "ficha 00123456"):
            resp = await client.post(
                "/api/decision",
                json={"audit_event_id": "evt-x", "decision": "accept", "reason": bad},
            )
            assert resp.status_code == 422, f"debió rechazar PII: {bad!r}"


async def test_decision_accept_returns_audit_event_clinician(monkeypatch):
    """POST /api/decision with accept → 200 + AuditEvent actor=clinician."""
    from unittest.mock import AsyncMock

    monkeypatch.setattr(
        "clinibrium.audit.engine.persist_audit", AsyncMock(return_value=None)
    )

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/decision",
            json={
                "audit_event_id": "evt-test-accept",
                "decision": "accept",
                "reason": "BPPV classic, no red flags",
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["actor"] == "clinician"
    assert data["event_type"] == "clinician_decision"
    assert data["outcome"] == "accept"
    assert "accept" in data["outcome_summary"]
    assert "evt-test-accept" in data["outcome_summary"]
    assert "BPPV classic" in data["outcome_summary"]


async def test_decision_reject_returns_audit_event_clinician(monkeypatch):
    """decision=reject → actor=clinician, outcome=reject."""
    from unittest.mock import AsyncMock

    monkeypatch.setattr(
        "clinibrium.audit.engine.persist_audit", AsyncMock(return_value=None)
    )

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/decision",
            json={
                "audit_event_id": "evt-test-reject",
                "decision": "reject",
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["actor"] == "clinician"
    assert data["outcome"] == "reject"
    assert "reject" in data["outcome_summary"]


async def test_decision_invalid_returns_422(monkeypatch):
    """decision=maybe → 422."""
    from unittest.mock import AsyncMock

    monkeypatch.setattr(
        "clinibrium.audit.engine.persist_audit", AsyncMock(return_value=None)
    )

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/decision",
            json={
                "audit_event_id": "evt-invalid",
                "decision": "maybe",
            },
        )

    assert resp.status_code == 422


async def test_decision_emits_exactly_one_audit_event(monkeypatch):
    """Each POST /api/decision → 1 clinician_decision AuditEvent."""
    from unittest.mock import AsyncMock

    persist_mock = AsyncMock(return_value=None)
    monkeypatch.setattr("clinibrium.audit.engine.persist_audit", persist_mock)

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/decision",
            json={
                "audit_event_id": "evt-count",
                "decision": "accept",
                "reason": "test single event",
            },
        )

    assert resp.status_code == 200
    persist_mock.assert_called_once()
    called_event = persist_mock.call_args[0][0]
    assert called_event.event_type == "clinician_decision"
    assert called_event.actor.value == "clinician"
    assert called_event.outcome == "accept"
