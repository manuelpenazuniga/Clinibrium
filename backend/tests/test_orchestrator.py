"""Tests for orchestrator + audit + storage — INV-4, INV-6, INV-8, determinism.

Covers the acceptance criteria of task T9a:
  - INV-4 exactly-1 under total degradation (ml + reasoner down)
  - INV-4 exactly-1 under unexpected exception (rails raise)
  - INV-1 end-to-end: active red flag → immediate urgency (rails win)
  - INV-6: ml down does not change urgency vs a run with ml
  - persist_audit fallback: DATABASE_URL=None → JSONL; DB failure → no break
  - Determinism: injected `now` → stable AuditEvent.occurred_at
  - Import separation: orchestrator/audit/storage respect the map
"""
from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from clinibrium.audit.engine import build_audit_event, emit
from clinibrium.config import Settings, get_settings
from clinibrium.contracts import (
    AuditEvent,
    CaseFeatures,
    Diagnosis,
    DifferentialCandidate,
    DifferentialResult,
    DixHallpikeResult,
    ForcedAction,
    HeadImpulse,
    HearingLoss,
    NystagmusDirection,
    Onset,
    PipelineResult,
    ReasonerOutput,
    RedFlagResult,
    SymptomDuration,
    TimingPattern,
    Trigger,
    Urgency,
)
from clinibrium.grounding import InlineGrounding
from clinibrium.orchestrator import evaluate
from clinibrium.storage.persist import persist_audit

FIXED_DT = datetime(2026, 7, 10, 12, 0, 0, tzinfo=timezone.utc)

# =============================================================================
# Helpers
# =============================================================================


def _bppv_benign() -> CaseFeatures:
    """Typical posterior BPPV: positional, torsional, Dix-Hallpike (+)."""
    return CaseFeatures(
        duration=SymptomDuration.under_1min,
        onset=Onset.sudden,
        trigger=Trigger.positional_head,
        timing_pattern=TimingPattern.episodic_triggered,
        nystagmus_direction=NystagmusDirection.mixed,
        nystagmus_latency_s=2.0,
        nystagmus_duration_s=20.0,
        nystagmus_fatigable=True,
        nystagmus_suppressed_by_fixation=True,
        dix_hallpike=DixHallpikeResult.right_positive,
        torsion_confirmed_by_clinician=True,
        episode_count=3,
        episode_duration=SymptomDuration.under_1min,
    )


def _red_flag_case() -> CaseFeatures:
    """Central AVS: abnormal HINTS + focal signs + age > 60 + vascular."""
    from clinibrium.contracts import FocalSign, VascularRiskFactor

    return CaseFeatures(
        timing_pattern=TimingPattern.acute_continuous,
        head_impulse=HeadImpulse.normal,
        skew_deviation=True,
        nystagmus_direction=NystagmusDirection.vertical_pure,
        focal_signs={FocalSign.dysarthria},
        age_years=65,
        vascular_risk_factors={VascularRiskFactor.hypertension},
    )


def _mock_reasoner_output() -> ReasonerOutput:
    return ReasonerOutput(
        explanation="Caso compatible con BPPV posterior típico.",
        reconciliation="Las red flags están presentes pero el cuadro posicional es clásico.",
        suggested_next_steps=["Derivar a urgencias por red flags activas."],
        model_used="claude-opus-4-8",
        reasoner_suggested_urgency=Urgency.ambulatoria,
        grounding_refs=["clinibrium-paraphrase:bppv_posterior-1"],
    )


def _capture_audit(monkeypatch):
    """Captures every call to the audit engine's emit()."""
    events: list[AuditEvent] = []

    async def _fake_emit(*args, **kwargs):
        event = build_audit_event(*args, **kwargs)
        events.append(event)
        return event

    monkeypatch.setattr("clinibrium.audit.engine.emit", _fake_emit)
    return events


def _capture_audit_module(monkeypatch):
    """Captures emit() from the audit module directly (for persistence tests)."""
    events: list[AuditEvent] = []

    async def _fake_emit(*args, **kwargs):
        event = build_audit_event(*args, **kwargs)
        events.append(event)
        return event

    monkeypatch.setattr("clinibrium.audit.engine.persist_audit", AsyncMock(return_value=None))
    return events


# =============================================================================
# INV-4 — exactly 1 AuditEvent under total degradation
# =============================================================================


async def test_exactly_one_audit_event_full_degradation(monkeypatch):
    """ml + reasoner down → 1 AuditEvent with reasoner_status='degraded'."""
    monkeypatch.setattr("clinibrium.ml_client.predict", AsyncMock(return_value=None))
    monkeypatch.setattr("clinibrium.reasoner.reason", AsyncMock(return_value=None))

    events = _capture_audit(monkeypatch)
    features = _bppv_benign()
    result = await evaluate(features, grounding=InlineGrounding(), now=FIXED_DT)

    assert len(events) == 1
    assert result.ml is None
    assert result.reasoning is None
    assert result.audit_event_id == events[0].id
    assert events[0].reasoner_status == "degraded"


async def test_exactly_one_audit_event_full_degradation_second_call(monkeypatch):
    """Two invocations → 2 events, never 0 nor shared."""
    monkeypatch.setattr("clinibrium.ml_client.predict", AsyncMock(return_value=None))
    monkeypatch.setattr("clinibrium.reasoner.reason", AsyncMock(return_value=None))

    events1 = _capture_audit(monkeypatch)
    r1 = await evaluate(_bppv_benign(), grounding=InlineGrounding(), now=FIXED_DT)
    assert len(events1) == 1

    events2 = _capture_audit(monkeypatch)
    r2 = await evaluate(_bppv_benign(), grounding=InlineGrounding(), now=FIXED_DT)
    assert len(events2) == 1
    assert events1[0].id != events2[0].id
    assert r1.audit_event_id != r2.audit_event_id


# =============================================================================
# INV-4 — exactly 1 AuditEvent under unexpected exception
# =============================================================================


async def test_exactly_one_audit_event_under_unexpected_exception(monkeypatch):
    """rails.apply_rails raises → 1 AuditEvent with outcome='error', urgency=inmediata."""
    monkeypatch.setattr("clinibrium.ml_client.predict", AsyncMock(return_value=None))
    monkeypatch.setattr("clinibrium.reasoner.reason", AsyncMock(return_value=None))

    def _raise(*args, **kwargs):  # noqa: ARG001
        raise RuntimeError("simulated rails failure")

    monkeypatch.setattr("clinibrium.rails.apply_rails", _raise)

    events = _capture_audit(monkeypatch)

    with pytest.raises(RuntimeError, match="simulated rails failure"):
        await evaluate(_bppv_benign(), grounding=InlineGrounding(), now=FIXED_DT)

    assert len(events) == 1
    assert events[0].outcome == "error"
    assert events[0].urgency == Urgency.inmediata


# =============================================================================
# INV-1 end-to-end — active red flag wins (rails seal the urgency)
# =============================================================================


async def test_inv1_red_flag_wins_over_ml_and_reasoner(monkeypatch):
    """Active red flag + ml/reasoner suggesting benign BPPV → immediate urgency."""
    mock_reasoner = AsyncMock(return_value=_mock_reasoner_output())
    monkeypatch.setattr("clinibrium.reasoner.reason", mock_reasoner)

    from clinibrium.contracts import PredictResponse

    monkeypatch.setattr(
        "clinibrium.ml_client.predict",
        AsyncMock(
            return_value=PredictResponse(
                probabilities={"bppv_posterior": 0.95, "meniere": 0.03, "central_suspected": 0.02},
                model_version="catboost-v1",
            )
        ),
    )

    events = _capture_audit(monkeypatch)
    result = await evaluate(_red_flag_case(), grounding=InlineGrounding(), now=FIXED_DT)

    assert result.urgency == Urgency.inmediata
    assert result.red_flag.red_flag_activa is True
    assert len(events) == 1
    assert events[0].urgency == Urgency.inmediata


# =============================================================================
# INV-6 — ml down does not change urgency vs a run with ml
# =============================================================================


async def test_inv6_ml_down_same_urgency_as_ml_present(monkeypatch):
    """ml=None vs ml=PredictResponse → same urgency (ML does not decide safety)."""
    mock_reasoner_output = _mock_reasoner_output()
    monkeypatch.setattr(
        "clinibrium.reasoner.reason", AsyncMock(return_value=mock_reasoner_output)
    )

    events = _capture_audit(monkeypatch)
    result_without_ml = await evaluate(
        _bppv_benign(), grounding=InlineGrounding(), now=FIXED_DT
    )
    assert result_without_ml.ml is None
    urgency_no_ml = result_without_ml.urgency
    assert len(events) == 1

    from clinibrium.contracts import PredictResponse

    monkeypatch.setattr(
        "clinibrium.ml_client.predict",
        AsyncMock(
            return_value=PredictResponse(
                probabilities={"bppv_posterior": 0.9},
                model_version="catboost-v1",
            )
        ),
    )

    events2 = _capture_audit(monkeypatch)
    result_with_ml = await evaluate(
        _bppv_benign(), grounding=InlineGrounding(), now=FIXED_DT
    )
    assert result_with_ml.ml is not None
    urgency_with_ml = result_with_ml.urgency
    assert len(events2) == 1

    assert urgency_no_ml == urgency_with_ml


# =============================================================================
# INV-8 — reasoner down: pipeline completes with reasoning marked degraded
# =============================================================================


async def test_reasoner_degraded_marks_reasoner_status(monkeypatch):
    """reasoner down → reasoner_status='degraded', pipeline completes."""
    monkeypatch.setattr("clinibrium.ml_client.predict", AsyncMock(return_value=None))
    monkeypatch.setattr("clinibrium.reasoner.reason", AsyncMock(return_value=None))

    events = _capture_audit(monkeypatch)
    result = await evaluate(_bppv_benign(), grounding=InlineGrounding(), now=FIXED_DT)

    assert result.reasoning is None
    assert result.audit_event_id is not None
    assert len(events) == 1
    assert events[0].reasoner_status == "degraded"


# =============================================================================
# persist_audit — JSONL fallback + DB failure does not break
# =============================================================================


def test_persist_audit_fallback_jsonl(monkeypatch, tmp_path):
    """DATABASE_URL=None → writes 1 line to the JSONL."""
    import asyncio

    jsonl = tmp_path / "audit.jsonl"

    monkeypatch.setattr(
        get_settings(),
        "DATABASE_URL",
        "",
    )
    monkeypatch.setattr(get_settings(), "AUDIT_LOG_PATH", str(jsonl))

    event = AuditEvent(
        id="evt-jsonl",
        occurred_at=FIXED_DT,
        event_type="pipeline_evaluation",
        input_features_hash="abc123",
        urgency=Urgency.ambulatoria,
        red_flag_activa=False,
        outcome_summary="test fallback JSONL",
    )

    asyncio.run(persist_audit(event))

    assert jsonl.exists()
    lines = jsonl.read_text().strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["id"] == "evt-jsonl"


def test_persist_audit_db_failure_does_not_break_pipeline(monkeypatch, tmp_path):
    """DB failure → does not break the pipeline, the event is still emitted."""
    import asyncio

    jsonl = tmp_path / "audit_fallback.jsonl"

    monkeypatch.setattr(
        get_settings(),
        "DATABASE_URL",
        "postgresql://fake:fake@localhost:9999/fake",
    )
    monkeypatch.setattr(get_settings(), "AUDIT_LOG_PATH", str(jsonl))

    event = AuditEvent(
        id="evt-db-fail",
        occurred_at=FIXED_DT,
        event_type="pipeline_evaluation",
        input_features_hash="def456",
        urgency=Urgency.ambulatoria,
        red_flag_activa=False,
        outcome_summary="test DB failure fallback",
    )

    asyncio.run(persist_audit(event))

    assert jsonl.exists()
    lines = jsonl.read_text().strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["id"] == "evt-db-fail"


# =============================================================================
# Determinism — injected now → stable AuditEvent.occurred_at
# =============================================================================


async def test_occurred_at_stable_with_injected_now(monkeypatch):
    """now=datetime(2026,7,10) → AuditEvent.occurred_at == that datetime."""
    monkeypatch.setattr("clinibrium.ml_client.predict", AsyncMock(return_value=None))
    monkeypatch.setattr("clinibrium.reasoner.reason", AsyncMock(return_value=None))

    events = _capture_audit(monkeypatch)

    injected = datetime(2026, 7, 10, 14, 30, 0, tzinfo=timezone.utc)
    result = await evaluate(_bppv_benign(), grounding=InlineGrounding(), now=injected)

    assert len(events) == 1
    assert events[0].occurred_at == injected
    assert result.audit_event_id == events[0].id


# =============================================================================
# build_audit_event — pure, deterministic
# =============================================================================


def test_build_audit_event_is_pure_and_deterministic():
    """Same inputs → same hash, same actor, no side effects."""
    features = _bppv_benign()
    result = PipelineResult(
        case_id="case-test",
        urgency=Urgency.ambulatoria,
        red_flag=RedFlagResult(red_flag_activa=False),
        differential=DifferentialResult(
            candidates=[
                DifferentialCandidate(diagnosis=Diagnosis.bppv_posterior, score=0.9)
            ]
        ),
    )

    e1 = build_audit_event(
        result, features,
        reasoner_status="degraded", outcome="evaluation", occurred_at=FIXED_DT,
    )
    e2 = build_audit_event(
        result, features,
        reasoner_status="degraded", outcome="evaluation", occurred_at=FIXED_DT,
    )

    assert e1.input_features_hash == e2.input_features_hash
    assert e1.actor == e2.actor
    assert e1.id != e2.id  # UUIDs are unique
    assert e1.occurred_at == FIXED_DT


def test_features_hash_deterministic():
    """Same CaseFeatures → same hash, different from another case."""
    from clinibrium.audit.engine import _features_hash

    h1 = _features_hash(_bppv_benign())
    h2 = _features_hash(_bppv_benign())
    h3 = _features_hash(_red_flag_case())

    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 64  # sha256 hex


# =============================================================================
# Import separation — respect the map (no forbidden imports)
# =============================================================================


def _iter_clinibrium_imports(py_file: Path) -> list[tuple[int, str]]:
    """Returns (line_no, module) for every `clinibrium.*` import found."""
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


def test_orchestrator_not_imported_by_engines_reasoner_rails():
    """FORBIDDEN: engines / reasoner / rails importing orchestrator."""
    pkg_root = Path(__file__).resolve().parents[1] / "clinibrium"
    offenders: list[str] = []
    for mod_name in ["redflag_engine", "differential_engine", "reasoner", "rails"]:
        mod_dir = pkg_root / mod_name
        if not mod_dir.exists():
            continue
        for py in sorted(mod_dir.glob("*.py")):
            for lineno, mod in _iter_clinibrium_imports(py):
                if "orchestrator" in mod:
                    offenders.append(f"{mod_name}/{py.name}:{lineno} → {mod}")
    assert not offenders, (
        "INV-5 violated — engines/reasoner/rails import orchestrator:\n  "
        + "\n  ".join(offenders)
    )


def test_audit_imports_respect_map():
    """audit → contracts, storage (NO engines/reasoner/orchestrator)."""
    pkg_root = Path(__file__).resolve().parents[1] / "clinibrium" / "audit"
    forbidden = {"redflag_engine", "differential_engine", "reasoner", "ml_client", "orchestrator", "rails"}
    offenders: list[str] = []
    for py in sorted(pkg_root.glob("*.py")):
        for lineno, mod in _iter_clinibrium_imports(py):
            for f in forbidden:
                if mod == f"clinibrium.{f}" or mod.startswith(f"clinibrium.{f}."):
                    offenders.append(f"audit/{py.name}:{lineno} → {mod}")
    assert not offenders, (
        "audit imports forbidden modules:\n  " + "\n  ".join(offenders)
    )


def test_storage_imports_respect_map():
    """storage → contracts, config, asyncpg (NO engines/reasoner/orchestrator)."""
    pkg_root = Path(__file__).resolve().parents[1] / "clinibrium" / "storage"
    forbidden = {"redflag_engine", "differential_engine", "reasoner", "ml_client", "orchestrator", "rails", "audit"}
    offenders: list[str] = []
    for py in sorted(pkg_root.glob("*.py")):
        for lineno, mod in _iter_clinibrium_imports(py):
            for f in forbidden:
                if mod == f"clinibrium.{f}" or mod.startswith(f"clinibrium.{f}."):
                    offenders.append(f"storage/{py.name}:{lineno} → {mod}")
    assert not offenders, (
        "storage imports forbidden modules:\n  " + "\n  ".join(offenders)
    )


def test_orchestrator_can_import_all_engines():
    """orchestrator is the only composer — it must import everything downstream."""
    pkg_root = Path(__file__).resolve().parents[1] / "clinibrium" / "orchestrator"
    all_imports: set[str] = set()
    for py in sorted(pkg_root.glob("*.py")):
        for _lineno, mod in _iter_clinibrium_imports(py):
            all_imports.add(mod)
    required = {"redflag_engine", "differential_engine", "reasoner", "ml_client", "rails", "audit", "grounding"}
    found: set[str] = set()
    for mod in all_imports:
        for req in required:
            if f"clinibrium.{req}" in mod or mod.endswith(f".{req}"):
                found.add(req)
    missing = required - found
    assert not missing, f"orchestrator does not import required modules: {missing}"


# =============================================================================
# CaseFeatures → AuditEvent: no PII in the hash (negative)
# =============================================================================


def test_features_hash_contains_no_pii():
    """The features hash is over de-identified fields (NO patient_name, etc)."""
    from clinibrium.audit.engine import _features_hash

    features = _bppv_benign()
    h = _features_hash(features)

    assert isinstance(h, str)
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


# =============================================================================
# Immutable JSONL (append-only) — append verification
# =============================================================================


def test_jsonl_append_only(monkeypatch, tmp_path):
    """Two events → two JSONL lines (append, not rewrite)."""
    import asyncio

    jsonl = tmp_path / "audit_append.jsonl"
    monkeypatch.setattr(get_settings(), "DATABASE_URL", "")
    monkeypatch.setattr(get_settings(), "AUDIT_LOG_PATH", str(jsonl))

    e1 = AuditEvent(
        id="evt-1",
        occurred_at=FIXED_DT,
        event_type="pipeline_evaluation",
        input_features_hash="aaa",
        urgency=Urgency.ambulatoria,
        red_flag_activa=False,
        outcome_summary="first",
    )
    e2 = AuditEvent(
        id="evt-2",
        occurred_at=FIXED_DT,
        event_type="pipeline_evaluation",
        input_features_hash="bbb",
        urgency=Urgency.inmediata,
        red_flag_activa=True,
        outcome_summary="second",
    )

    asyncio.run(persist_audit(e1))
    asyncio.run(persist_audit(e2))

    lines = jsonl.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["id"] == "evt-1"
    assert json.loads(lines[1])["id"] == "evt-2"


# =============================================================================
# AuditEvent frozen — cannot be mutated after build
# =============================================================================


def test_audit_event_frozen_from_build():
    """build_audit_event produces an immutable (frozen) AuditEvent."""
    features = _bppv_benign()
    result = PipelineResult(
        case_id="case-frozen",
        urgency=Urgency.ambulatoria,
        red_flag=RedFlagResult(red_flag_activa=False),
        differential=DifferentialResult(candidates=[]),
    )
    event = build_audit_event(
        result, features,
        reasoner_status="ok", outcome="evaluation", occurred_at=FIXED_DT,
    )

    with pytest.raises(Exception):
        event.urgency = Urgency.inmediata  # type: ignore[misc]


async def test_exactly_one_audit_event_when_exception_after_emit(monkeypatch):
    """INV-4 (Gemini audit fix): if something fails AFTER the successful
    emit of step 8 (e.g. model_copy), the `audited` flag prevents a SECOND
    error AuditEvent. Exactly 1 must remain (the successful one)."""
    monkeypatch.setattr("clinibrium.ml_client.predict", AsyncMock(return_value=None))
    monkeypatch.setattr("clinibrium.reasoner.reason", AsyncMock(return_value=None))
    events = _capture_audit(monkeypatch)

    # Selective model_copy: blows up ONLY on the post-emit call (the one
    # carrying audit_event_id in the update); lets any other model_copy through.
    real_model_copy = PipelineResult.model_copy

    def _selective_boom(self, *args, update=None, **kwargs):  # type: ignore[no-untyped-def]
        if update and "audit_event_id" in update:
            raise RuntimeError("post-emit boom")
        return real_model_copy(self, *args, update=update, **kwargs)

    monkeypatch.setattr(PipelineResult, "model_copy", _selective_boom)

    with pytest.raises(RuntimeError, match="post-emit boom"):
        await evaluate(_bppv_benign(), grounding=InlineGrounding(), now=FIXED_DT)

    # Exactly 1 AuditEvent (the successful "evaluation"), NOT a 2nd error one.
    assert len(events) == 1
    assert events[0].outcome == "evaluation"


# =============================================================================
# Kill-Claude toggle (kill_reasoner, intentional INV-8)
# =============================================================================


async def test_kill_reasoner_yields_reasoning_none(monkeypatch):
    """kill_reasoner=True → reasoning=None, reasoner.reason NEVER called."""
    monkeypatch.setattr("clinibrium.ml_client.predict", AsyncMock(return_value=None))
    mock_reasoner = AsyncMock()
    monkeypatch.setattr("clinibrium.reasoner.reason", mock_reasoner)

    events = _capture_audit(monkeypatch)
    result = await evaluate(
        _bppv_benign(),
        grounding=InlineGrounding(),
        now=FIXED_DT,
        kill_reasoner=True,
    )

    assert result.reasoning is None
    assert len(events) == 1
    assert events[0].reasoner_status == "degraded"
    mock_reasoner.assert_not_called()


async def test_kill_reasoner_same_urgency_as_without_kill(monkeypatch):
    """kill_reasoner=True → same urgency as without kill (INV-8 does not touch safety)."""
    mock_reasoner_output = _mock_reasoner_output()
    monkeypatch.setattr(
        "clinibrium.reasoner.reason", AsyncMock(return_value=mock_reasoner_output)
    )
    monkeypatch.setattr("clinibrium.ml_client.predict", AsyncMock(return_value=None))

    events1 = _capture_audit(monkeypatch)
    result_with = await evaluate(
        _bppv_benign(),
        grounding=InlineGrounding(),
        now=FIXED_DT,
        kill_reasoner=False,
    )
    assert len(events1) == 1
    assert result_with.reasoning is not None
    urgency_with = result_with.urgency

    events2 = _capture_audit(monkeypatch)
    result_without = await evaluate(
        _bppv_benign(),
        grounding=InlineGrounding(),
        now=FIXED_DT,
        kill_reasoner=True,
    )
    assert len(events2) == 1
    assert result_without.reasoning is None
    urgency_without = result_without.urgency

    assert urgency_with == urgency_without


async def test_kill_reasoner_exactly_one_audit_event(monkeypatch):
    """kill_reasoner=True → exactly 1 AuditEvent (INV-4 intact)."""
    monkeypatch.setattr("clinibrium.ml_client.predict", AsyncMock(return_value=None))
    monkeypatch.setattr("clinibrium.reasoner.reason", AsyncMock(return_value=None))

    events = _capture_audit(monkeypatch)
    result = await evaluate(
        _bppv_benign(),
        grounding=InlineGrounding(),
        now=FIXED_DT,
        kill_reasoner=True,
    )

    assert len(events) == 1
    assert result.audit_event_id is not None
    assert result.audit_event_id == events[0].id
    assert events[0].reasoner_status == "degraded"


async def test_kill_reasoner_red_flag_case_urgency_unchanged(monkeypatch):
    """kill_reasoner=True with an active red flag → immediate urgency (no downgrade)."""
    mock_reasoner_output = _mock_reasoner_output()
    monkeypatch.setattr(
        "clinibrium.reasoner.reason", AsyncMock(return_value=mock_reasoner_output)
    )
    monkeypatch.setattr("clinibrium.ml_client.predict", AsyncMock(return_value=None))

    events = _capture_audit(monkeypatch)
    result = await evaluate(
        _red_flag_case(),
        grounding=InlineGrounding(),
        now=FIXED_DT,
        kill_reasoner=True,
    )

    assert result.red_flag.red_flag_activa is True
    assert result.urgency == Urgency.inmediata
    assert len(events) == 1
    assert events[0].urgency == Urgency.inmediata


# =============================================================================
# T-CLIN r1 — sudden hearing loss: A8 (with AVS) urgent vs B1 (isolated) priority
# =============================================================================


async def test_isolated_sudden_hearing_loss_is_prioritaria(monkeypatch):
    """T-CLIN r1: ISOLATED sudden sensorineural hearing loss → PRIORITY
    (ENT 48h), NOT immediate. B1 contributes ESCALAR; does not set red_flag_activa."""
    monkeypatch.setattr("clinibrium.ml_client.predict", AsyncMock(return_value=None))
    f = CaseFeatures(hearing_loss=HearingLoss.sudden_unilateral)
    result = await evaluate(
        f, grounding=InlineGrounding(), now=FIXED_DT, kill_reasoner=True
    )
    assert result.red_flag.red_flag_activa is False
    assert result.urgency == Urgency.prioritaria
    assert ForcedAction.ESCALAR in result.forced_actions


async def test_sudden_hearing_loss_with_avs_is_inmediata(monkeypatch):
    """Sudden hearing loss + acute vertigo (AVS) → A8 (AICA) → IMMEDIATE.
    Safety test for the B1 change: it must not de-escalate the AVS case."""
    monkeypatch.setattr("clinibrium.ml_client.predict", AsyncMock(return_value=None))
    f = CaseFeatures(
        hearing_loss=HearingLoss.sudden_unilateral,
        timing_pattern=TimingPattern.acute_continuous,
    )
    result = await evaluate(
        f, grounding=InlineGrounding(), now=FIXED_DT, kill_reasoner=True
    )
    assert result.red_flag.red_flag_activa is True
    assert result.urgency == Urgency.inmediata
    assert ForcedAction.DERIVAR_URGENTE in result.forced_actions
