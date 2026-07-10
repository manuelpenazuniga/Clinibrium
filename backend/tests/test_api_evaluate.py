"""Tests del endpoint SSE `POST /api/evaluate` — T9b (streaming pipeline).

Cubre los criterios de aceptación de la tarea T9b:
  - 200 text/event-stream con eventos redflag, differential, ml, reasoning, rails, done
  - 422 ante body inválido (campo extra — incluido `recording_mode` — o tipo malo)
  - Caso con red flag activa → urgency=inmediata propagada al evento `rails` y `done`
  - reasoner/ml mockeados (cero llamadas reales de red en el gate)
  - `on_stage` NO rompe INV-4: hook que raise → evaluate completa + 1 AuditEvent
  - api respeta el mapa: solo orchestrator + contracts (sin engines/reasoner/rails/grounding/ml_client)
  - `recording_mode` NUNCA viene del body (verificable: schema CaseFeatures sin
    ese campo + firma del handler)
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
    """BPPV posterior típico: positional, torsional, Dix-Hallpike (+).

    Mismas features que `test_orchestrator._bppv_benign`, serializadas como
    dict (lo que llega por HTTP).
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
    """AVS central: HINTS anormal + focal signs + edad > 60 + vascular risk."""
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
    """Mockea ml + reasoner para cero llamadas reales de red en el gate."""
    monkeypatch.setattr("clinibrium.ml_client.predict", AsyncMock(return_value=None))
    monkeypatch.setattr("clinibrium.reasoner.reason", AsyncMock(return_value=None))


def _capture_audit(monkeypatch) -> list:
    """Captura cada llamada a `emit()` del audit engine.

    Devuelve la lista que se va llenando — al final del test tiene todos
    los AuditEvents que emitió el orchestrator.
    """
    events: list = []

    async def _fake_emit(*args, **kwargs):
        event = build_audit_event(*args, **kwargs)
        events.append(event)
        return event

    monkeypatch.setattr("clinibrium.audit.engine.emit", _fake_emit)
    return events


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    """Divide el stream SSE en pares ordenados (event, data).

    El formato esperado por evento es:
        event: <name>\\ndata: <json>\\n\\n
    (json.dumps produce JSON sin espacios por default — json.loads lo acepta).
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
    """POST a /api/evaluate y devuelve (status, headers, eventos parseados)."""
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
# Happy path: BPPV benigno → 200 text/event-stream + done con audit_event_id
# =============================================================================


async def test_evaluate_sse_bppv_benign_streams_all_stages_and_done(monkeypatch):
    """POST /api/evaluate con BPPV benigno → 200 text/event-stream + done."""
    _mock_ml_and_reasoner(monkeypatch)
    app = create_app()

    status, headers, events = await _post_evaluate(app, _bppv_benign())

    assert status == 200
    assert headers["content-type"].startswith("text/event-stream")

    names = [n for n, _ in events]
    # El demo (v7.3 §10) muestra el pipeline "pensando" en streaming
    # en este orden estricto: redflag → differential → ml → reasoning → rails → done.
    assert names == [
        "redflag",
        "differential",
        "ml",
        "reasoning",
        "rails",
        "done",
    ], f"orden/presencia inesperado: {names}"

    # done payload: audit_event_id poblado + urgencia ambulatoria (BPPV benigno).
    done_payload = events[-1][1]
    assert done_payload["audit_event_id"] is not None
    assert done_payload["urgency"] == Urgency.ambulatoria.value
    # Artefacto auditable FHIR incluido en el done event.
    bundle = done_payload["fhir_bundle"]
    assert bundle["resourceType"] == "Bundle"
    assert any(
        e["resource"]["resourceType"] == "AuditEvent" for e in bundle["entry"]
    )
    # bundle_sha256 presente y con longitud correcta (tamper-evident).
    sha = done_payload["bundle_sha256"]
    assert isinstance(sha, str)
    assert len(sha) == 64
    assert all(c in "0123456789abcdef" for c in sha)

    # Payload de redflag y rails: shape desidentificado correcto.
    redflag_payload = events[0][1]
    assert redflag_payload["red_flag_activa"] is False
    assert redflag_payload["hits_count"] == 0

    rails_payload = events[4][1]
    assert rails_payload["urgency"] == Urgency.ambulatoria.value
    assert "forced_actions" in rails_payload
    assert "applied_rails" in rails_payload


# =============================================================================
# Red flag activa → urgencia=inmediata propagada al SSE
# =============================================================================


async def test_evaluate_sse_red_flag_urgency_inmediata_propagated(monkeypatch):
    """Red flag activa → urgency=inmediata en el evento `rails` y `done`."""
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
# 422 — body inválido (campo extra / tipo malo)
# =============================================================================


async def test_evaluate_sse_invalid_body_extra_field_returns_422(monkeypatch):
    """Body con campos fuera del allowlist (p.ej. `recording_mode`) → 422.

    Demuestra AD-6 desde el lado HTTP: el cliente NO puede meter
    `recording_mode` ni PII en el body.
    """
    _mock_ml_and_reasoner(monkeypatch)
    app = create_app()

    payload = _bppv_benign()
    payload["recording_mode"] = True  # AD-6: NUNCA viene del body
    payload["patient_name"] = "Juan Pérez"  # PII prohibida por INV-2

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/evaluate", json=payload)
    assert resp.status_code == 422


async def test_evaluate_sse_invalid_body_bad_type_returns_422(monkeypatch):
    """Body con tipo malo (string en vez de int) → 422."""
    _mock_ml_and_reasoner(monkeypatch)
    app = create_app()

    payload = _bppv_benign()
    payload["age_years"] = "old"  # int esperado, string enviado

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/evaluate", json=payload)
    assert resp.status_code == 422


# =============================================================================
# AD-6: recording_mode NO está en la firma del handler (server-side only)
# =============================================================================


def test_evaluate_endpoint_signature_has_no_recording_mode(monkeypatch):
    """El handler NO acepta `recording_mode` del body — se lee de Settings
    (server-side, AD-6). `debug_kill_reasoner` es query param legítimo de demo."""
    from clinibrium.api.evaluate import evaluate_endpoint

    sig = inspect.signature(evaluate_endpoint)
    assert "recording_mode" not in sig.parameters, (
        "AD-6 violado: recording_mode NUNCA debe ser parámetro del endpoint. "
        f"Parámetros: {list(sig.parameters)}"
    )
    assert "features" in sig.parameters
    assert "debug_kill_reasoner" in sig.parameters


# =============================================================================
# on_stage NO rompe INV-4
# =============================================================================


async def test_on_stage_hook_raising_does_not_break_inv4(monkeypatch):
    """Hook que raise → evaluate completa y emite EXACTAMENTE 1 AuditEvent.

    Garantía adicional al INV-4 existente: aunque el observador falle,
    la auditoría y el resultado del pipeline no se ven afectados.
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

    # Pipeline completa, urgencia correcta, audit_event_id poblado.
    assert result.audit_event_id is not None
    assert result.urgency == Urgency.ambulatoria
    # INV-4: EXACTAMENTE 1 AuditEvent (el exitoso, NO un 2º de error
    # producto del raise del hook).
    assert len(events) == 1
    assert events[0].outcome == "evaluation"
    assert events[0].id == result.audit_event_id


async def test_on_stage_hook_receives_all_five_stages_in_order(monkeypatch):
    """on_stage es invocado al menos 5 veces, en orden, para los 5 stages."""
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

    # El spec exige estos 5 stages en este orden estricto.
    assert seen == ["redflag", "differential", "ml", "reasoning", "rails"], (
        f"stages inesperados: {seen}"
    )


async def test_on_stage_hook_receives_serializable_payloads(monkeypatch):
    """Los payloads de cada stage son JSON-serializables (desidentificados)."""
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

    # Cada payload debe ser JSON-serializable (no datetime, no set, no PII).
    for stage, payload in payloads.items():
        try:
            json.dumps(payload)
        except (TypeError, ValueError) as exc:
            pytest.fail(f"payload de {stage!r} no JSON-serializable: {exc} — {payload!r}")

    # Shape esperado por stage.
    assert set(payloads["redflag"].keys()) == {"red_flag_activa", "hits_count"}
    assert "top_candidates" in payloads["differential"]
    assert payloads["ml"] == {"available": False}
    assert payloads["reasoning"] == {"available": False, "model_used": None}
    assert set(payloads["rails"].keys()) == {"urgency", "forced_actions", "applied_rails"}


# =============================================================================
# Mapa: api NO importa engines / reasoner / rails / grounding / ml_client directo
# =============================================================================


def _iter_clinibrium_imports(py_file: Path) -> list[tuple[int, str]]:
    """Devuelve (line_no, module) de cada import de `clinibrium.*`."""
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
    """Mapa: `api → orchestrator, contracts`. PROHIBIDO importar directo
    engines / reasoner / rails / grounding / ml_client. Todo pasa por
    orchestrator."""
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
        "api importa módulos prohibidos (debe pasar por orchestrator):\n  "
        + "\n  ".join(offenders)
    )


# =============================================================================
# Settings — recording_mode se lee del server, no del body
# =============================================================================


def test_settings_has_recording_mode_server_side(monkeypatch):
    """`Settings.RECORDING_MODE` es server-side (env / .env) — el endpoint
    lo lee con `get_settings().RECORDING_MODE`, NUNCA del body."""
    assert hasattr(get_settings(), "RECORDING_MODE")
    assert isinstance(get_settings().RECORDING_MODE, bool)


# =============================================================================
# kill_reasoner — debug_kill_reasoner query param (INV-8 intencional)
# =============================================================================


async def test_evaluate_sse_debug_kill_reasoner_yields_reasoning_none(monkeypatch):
    """debug_kill_reasoner=true → done event con reasoning=None, reasoner no llamado."""
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


# =============================================================================
# POST /api/decision — intervención humana registrada (AD-4)
# =============================================================================


async def test_decision_accept_returns_audit_event_clinician(monkeypatch):
    """POST /api/decision con accept → 200 + AuditEvent actor=clinician."""
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
    """Cada POST /api/decision → 1 AuditEvent clinician_decision."""
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
