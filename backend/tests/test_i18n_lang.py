"""Bilingual UI (`lang`) presentation-boundary tests.

These guard the safety contract of the language toggle:
  - Spanish (default) output is byte-identical to the pre-toggle behavior.
  - English localizes ONLY presentation (labels/prose), never enums, urgency,
    forced actions, red-flag decisions, or the `/predict` payload.
  - `lang` never reaches the ML engine and never enters CaseFeatures.
  - Every canonical id/key has an English translation (completeness).
  - Invalid `lang` is rejected (422), not silently accepted.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from httpx import ASGITransport, AsyncClient

from clinibrium.api import create_app
from clinibrium.audit.engine import build_audit_event
from clinibrium.contracts import (
    AuditEvent,
    FocalSign,
    HeadImpulse,
    NystagmusDirection,
    TimingPattern,
    Urgency,
    VascularRiskFactor,
)
from clinibrium.storage.persist import _persist_jsonl, _persist_postgres
from clinibrium.counterfactual.engine import _PERTURBATIONS
from clinibrium.i18n import (
    COUNTERFACTUAL_LABELS_EN,
    REDFLAG_LABELS_EN,
    localize_counterfactual_change,
    localize_redflag_label,
)
from clinibrium.reasoner.engine import _build_system_prompt
from clinibrium.redflag_engine.rules import RULES


# =============================================================================
# Helpers
# =============================================================================
def _bppv_benign() -> dict:
    return {
        "duration": "under_1min",
        "onset": "sudden",
        "trigger": "positional_head",
        "timing_pattern": "episodic_triggered",
        "nystagmus_direction": "mixed",
        "nystagmus_latency_s": 2.0,
        "nystagmus_duration_s": 20.0,
        "nystagmus_fatigable": True,
        "nystagmus_suppressed_by_fixation": True,
        "dix_hallpike": "right_positive",
        "torsion_confirmed_by_clinician": True,
        "episode_count": 3,
        "episode_duration": "under_1min",
    }


def _red_flag_case() -> dict:
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
    monkeypatch.setattr("clinibrium.ml_client.predict", AsyncMock(return_value=None))
    monkeypatch.setattr("clinibrium.reasoner.reason", AsyncMock(return_value=None))


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        ev_name = ev_data = None
        for line in block.splitlines():
            if line.startswith("event:"):
                ev_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                ev_data = line[len("data:") :].strip()
        if ev_name is not None and ev_data is not None:
            out.append((ev_name, json.loads(ev_data)))
    return out


async def _post_evaluate(app, payload: dict, lang: str | None = None):
    url = "/api/evaluate" if lang is None else f"/api/evaluate?lang={lang}"
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream("POST", url, json=payload) as resp:
            status = resp.status_code
            text = ""
            async for chunk in resp.aiter_text():
                text += chunk
    return status, _parse_sse(text)


def _done(events) -> dict:
    return next(data for name, data in events if name == "done")


# =============================================================================
# Completeness: every canonical id/key has an English translation
# =============================================================================
def test_every_redflag_rule_has_english_label():
    missing = [r.id for r in RULES if r.id not in REDFLAG_LABELS_EN]
    assert not missing, f"red-flag rules without EN label: {missing}"


def test_every_counterfactual_key_has_english_label():
    missing = [p.key for p in _PERTURBATIONS if p.key not in COUNTERFACTUAL_LABELS_EN]
    assert not missing, f"counterfactual perturbations without EN label: {missing}"


def test_counterfactual_keys_are_unique():
    keys = [p.key for p in _PERTURBATIONS]
    assert len(keys) == len(set(keys))


# =============================================================================
# Localize helpers: es no-op, en maps, unknown → fallback
# =============================================================================
def test_localize_redflag_es_is_noop():
    assert localize_redflag_label("A1", "Etiqueta ES", "es") == "Etiqueta ES"


def test_localize_redflag_en_maps_and_falls_back():
    assert localize_redflag_label("A5", "Signos neurológicos focales", "en") == (
        "Focal neurological signs"
    )
    # Unknown id → falls back to the canonical label (never blank/crash).
    assert localize_redflag_label("ZZ", "Canon", "en") == "Canon"


def test_localize_counterfactual_es_noop_en_maps():
    assert localize_counterfactual_change("neck_stiffness", "Rigidez de nuca", "es") == (
        "Rigidez de nuca"
    )
    assert localize_counterfactual_change("neck_stiffness", "Rigidez de nuca", "en") == (
        "Neck stiffness"
    )


# =============================================================================
# Reasoner prompt: es byte-identical, en appends an output-language directive
# =============================================================================
def test_system_prompt_es_is_byte_identical_to_default():
    assert _build_system_prompt("es") == _build_system_prompt()


def test_system_prompt_en_appends_directive_without_restructuring():
    es = _build_system_prompt("es")
    en = _build_system_prompt("en")
    assert en != es
    assert en.startswith(es)  # same prompt, directive only appended
    assert "ENGLISH" in en


# =============================================================================
# /api/evaluate: en localizes labels but NOT enums/urgency/red flags
# =============================================================================
async def test_evaluate_en_localizes_redflag_labels_only(monkeypatch):
    _mock_ml_and_reasoner(monkeypatch)
    app = create_app()

    status_es, events_es = await _post_evaluate(app, _red_flag_case(), lang="es")
    status_en, events_en = await _post_evaluate(app, _red_flag_case(), lang="en")
    assert status_es == 200 and status_en == 200

    done_es, done_en = _done(events_es), _done(events_en)

    # Safety-critical fields are byte-identical across languages.
    assert done_es["urgency"] == done_en["urgency"]
    assert done_es["forced_actions"] == done_en["forced_actions"]
    assert done_es["applied_rails"] == done_en["applied_rails"]
    assert done_es["red_flag"]["red_flag_activa"] == done_en["red_flag"]["red_flag_activa"]

    hits_es = {h["id"]: h["label"] for h in done_es["red_flag"]["hits"]}
    hits_en = {h["id"]: h["label"] for h in done_en["red_flag"]["hits"]}
    assert set(hits_es) == set(hits_en)  # same rule ids fired
    assert hits_es, "expected at least one red-flag hit for the red-flag case"

    # es labels are the canonical Spanish; en labels are the mapped English.
    for rid, es_label in hits_es.items():
        assert es_label == next(r.label for r in RULES if r.id == rid)
        assert hits_en[rid] == REDFLAG_LABELS_EN[rid]
        assert hits_en[rid] != es_label


def _lang_projection(done: dict) -> dict:
    """Language-relevant + safety fields, minus volatile audit id/timestamp."""
    return {
        "urgency": done["urgency"],
        "forced_actions": done["forced_actions"],
        "applied_rails": done["applied_rails"],
        "hits": [(h["id"], h["label"]) for h in done["red_flag"]["hits"]],
    }


async def test_evaluate_default_lang_is_spanish(monkeypatch):
    _mock_ml_and_reasoner(monkeypatch)
    app = create_app()
    _, events_default = await _post_evaluate(app, _red_flag_case())
    _, events_es = await _post_evaluate(app, _red_flag_case(), lang="es")
    assert _lang_projection(_done(events_default)) == _lang_projection(_done(events_es))


async def test_evaluate_invalid_lang_returns_422(monkeypatch):
    _mock_ml_and_reasoner(monkeypatch)
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/evaluate?lang=fr", json=_bppv_benign())
    assert resp.status_code == 422


async def test_predict_payload_is_identical_regardless_of_lang(monkeypatch):
    """INV-2 / contract: `lang` NEVER perturbs the payload sent to /predict."""
    predict_mock = AsyncMock(return_value=None)
    monkeypatch.setattr("clinibrium.ml_client.predict", predict_mock)
    monkeypatch.setattr("clinibrium.reasoner.reason", AsyncMock(return_value=None))
    app = create_app()

    await _post_evaluate(app, _bppv_benign(), lang="es")
    await _post_evaluate(app, _bppv_benign(), lang="en")

    assert predict_mock.await_count == 2
    features_es = predict_mock.await_args_list[0].args[0]
    features_en = predict_mock.await_args_list[1].args[0]
    # The CaseFeatures object handed to the ML client is byte-identical, and it
    # carries no `lang` field at all (would 422 on CaseFeatures otherwise).
    assert features_es.model_dump(mode="json") == features_en.model_dump(mode="json")
    assert "lang" not in features_es.model_dump(mode="json")


async def test_audit_records_output_lang(monkeypatch):
    _mock_ml_and_reasoner(monkeypatch)
    captured: list = []

    async def _fake_emit(*args, **kwargs):
        event = build_audit_event(*args, **kwargs)
        captured.append(event)
        return event

    monkeypatch.setattr("clinibrium.audit.engine.emit", _fake_emit)
    app = create_app()
    await _post_evaluate(app, _bppv_benign(), lang="en")
    assert captured and captured[-1].output_lang == "en"


# =============================================================================
# /api/what-would-change: en localizes `change`, es unchanged
# =============================================================================
async def _post_wwc(app, payload: dict, lang: str | None = None):
    url = "/api/what-would-change" if lang is None else f"/api/what-would-change?lang={lang}"
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(url, json=payload)
    return resp.status_code, resp.json()


async def test_what_would_change_es_default_is_spanish(monkeypatch):
    app = create_app()
    status, data = await _post_wwc(app, _bppv_benign())
    assert status == 200
    cfs = data["counterfactuals"]
    assert cfs, "expected counterfactuals for a benign case"
    # Every cf carries a stable key and its canonical Spanish change text.
    for cf in cfs:
        assert cf["change_key"]
    diplopia = next(c for c in cfs if c["change_key"] == "focal_signs.diplopia")
    assert diplopia["change"] == "Nuevo signo focal: diplopía"


async def test_what_would_change_en_localizes_change(monkeypatch):
    app = create_app()
    status, data = await _post_wwc(app, _bppv_benign(), lang="en")
    assert status == 200
    diplopia = next(
        c for c in data["counterfactuals"] if c["change_key"] == "focal_signs.diplopia"
    )
    assert diplopia["change"] == "New focal sign: diplopia"
    # Enum fields untouched by localization.
    assert diplopia["new_urgency"] in {"inmediata", "prioritaria", "ambulatoria"}


async def test_what_would_change_invalid_lang_returns_422(monkeypatch):
    transport = ASGITransport(app=create_app())
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/what-would-change?lang=de", json=_bppv_benign())
    assert resp.status_code == 422


# =============================================================================
# Persistence: output_lang survives the REAL Postgres and JSONL paths
# (codex-audit-4 Alta 2: inspecting the in-memory event is not enough —
# these exercise `_persist_postgres` / `_persist_jsonl` themselves)
# =============================================================================
def _audit_event_en() -> AuditEvent:
    return AuditEvent(
        id="evt-i18n-1",
        occurred_at=datetime(2026, 7, 13, 12, 0, 0, tzinfo=timezone.utc),
        event_type="pipeline_evaluation",
        input_features_hash="sha256:" + "a" * 64,
        urgency=Urgency.ambulatoria,
        red_flag_activa=False,
        outcome_summary="i18n persistence test",
        output_lang="en",
    )


async def test_postgres_persistence_carries_output_lang(monkeypatch):
    executed: list[tuple[str, tuple]] = []

    class _FakeConn:
        async def execute(self, query: str, *args):
            executed.append((query, args))

        async def close(self) -> None:
            pass

    async def _fake_connect(url: str):
        return _FakeConn()

    monkeypatch.setattr("clinibrium.storage.persist.asyncpg.connect", _fake_connect)
    await _persist_postgres(_audit_event_en(), "postgresql://fake")

    ddl = next(q for q, _ in executed if "CREATE TABLE" in q)
    assert "output_lang" in ddl
    # Tables created BEFORE the column existed get the additive migration
    # (CREATE TABLE IF NOT EXISTS never alters an existing table).
    assert any(
        "ALTER TABLE audit_events ADD COLUMN IF NOT EXISTS output_lang" in q
        for q, _ in executed
    )

    insert_q, insert_args = next((q, a) for q, a in executed if "INSERT INTO" in q)
    assert "output_lang" in insert_q
    # Column list, $N placeholders and the row tuple stay aligned.
    cols = insert_q.split("(", 1)[1].split(")")[0].split(",")
    placeholders = re.findall(r"\$\d+", insert_q)
    assert len(cols) == len(placeholders) == len(insert_args)
    assert insert_args[-1] == "en"


def test_jsonl_persistence_carries_output_lang(tmp_path):
    path = tmp_path / "audit.jsonl"
    _persist_jsonl(_audit_event_en(), str(path))
    record = json.loads(path.read_text().strip())
    assert record["output_lang"] == "en"
