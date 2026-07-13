"""Tests for the `reasoner` module (T7).

Covers:
  - INV-2: privacy validator (fail-closed)
  - INV-8: degradation to None on API failure
  - Happy path: reason() with a mocked SDK
  - pick_model: Opus vs Haiku selection
  - Import separation: reasoner does not import forbidden modules
"""
from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from clinibrium.contracts import (
    NETWORK_SAFE_FIELDS,
    CaseFeatures,
    Diagnosis,
    DifferentialCandidate,
    DifferentialResult,
    PredictResponse,
    ReasonerOutput,
    RedFlagResult,
    Urgency,
)
from clinibrium.grounding import GroundingChunk
from clinibrium.reasoner import (
    HAIKU,
    OPUS,
    PrivacyViolation,
    build_network_payload,
    pick_model,
    reason,
)
from clinibrium.reasoner.engine import _LLMReasoning

# =========================================================================
# INV-2 — privacy validator
# =========================================================================


def test_privacy_rejects_field_outside_allowlist() -> None:
    """Exercises the REAL build_network_payload (not a duplicated helper): a
    features whose model_dump leaks a PII key must RAISE."""

    class _LeakyFeatures:
        """Stub that injects a key outside the allowlist into the dump."""

        def model_dump(self, mode: str = "json") -> dict:
            payload = CaseFeatures().model_dump(mode="json")
            payload["patient_name"] = "Juan Pérez"
            return payload

    with pytest.raises(PrivacyViolation):
        build_network_payload(_LeakyFeatures())  # type: ignore[arg-type]


def test_privacy_allows_only_safe_fields() -> None:
    safe = build_network_payload(CaseFeatures())
    assert set(safe.keys()) <= NETWORK_SAFE_FIELDS


def test_build_network_payload_passes_for_valid_features() -> None:
    """build_network_payload does not reject a clean CaseFeatures."""
    features = CaseFeatures(nystagmus_direction="torsional_pure")  # type: ignore[arg-type]
    payload = build_network_payload(features)
    assert payload["nystagmus_direction"] == "torsional_pure"


# =========================================================================
# pick_model
# =========================================================================


def test_pick_model_red_flag_activa_returns_opus() -> None:
    rf = RedFlagResult(red_flag_activa=True)
    assert pick_model(rf) == OPUS


def test_pick_model_ambulatory_returns_haiku() -> None:
    rf = RedFlagResult(red_flag_activa=False)
    assert pick_model(rf) == HAIKU


def test_pick_model_recording_mode_returns_opus() -> None:
    rf = RedFlagResult(red_flag_activa=False)
    assert pick_model(rf, recording_mode=True) == OPUS


def test_pick_model_strings_are_exact_pinned_ids() -> None:
    assert OPUS == "claude-opus-4-8"
    assert HAIKU == "claude-haiku-4-5-20251001"


# =========================================================================
# INV-2 choke point — engine builds prompt from build_network_payload
# =========================================================================


def _sample_grounding_chunks() -> list[GroundingChunk]:
    return [
        GroundingChunk(
            text="El VPPB del canal posterior cursa con nistagmo torsional-upbeating...",
            diagnosis=Diagnosis.bppv_posterior,
            source_id="clinibrium-paraphrase:bppv_posterior-1",
        ),
        GroundingChunk(
            text="La neuritis vestibular presenta nistagmo horizonto-rotatorio espontáneo...",
            diagnosis=Diagnosis.vestibular_neuritis,
            source_id="clinibrium-paraphrase:vestibular_neuritis-1",
        ),
    ]


def _sample_differential() -> DifferentialResult:
    return DifferentialResult(
        candidates=[
            DifferentialCandidate(
                diagnosis=Diagnosis.bppv_posterior, score=0.85, rule_ids=["R1"]
            ),
            DifferentialCandidate(
                diagnosis=Diagnosis.vestibular_neuritis, score=0.40, rule_ids=["R3"]
            ),
        ]
    )


def _sample_features() -> CaseFeatures:
    from clinibrium.contracts.enums import (
        NystagmusDirection,
        Onset,
        SymptomDuration,
        TimingPattern,
        Trigger,
    )

    return CaseFeatures(
        duration=SymptomDuration.seconds,
        onset=Onset.sudden,
        trigger=Trigger.positional_head,
        timing_pattern=TimingPattern.episodic_triggered,
        nystagmus_direction=NystagmusDirection.torsional_pure,
    )


async def test_reason_calls_build_network_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """INV-2: the engine must build the prompt ONLY from build_network_payload."""
    called_with: list[CaseFeatures] = []

    def _spy(features: CaseFeatures) -> dict:
        called_with.append(features)
        return {"test": "ok"}

    monkeypatch.setattr("clinibrium.reasoner.engine.build_network_payload", _spy)

    rf = RedFlagResult(red_flag_activa=False)
    features = _sample_features()
    differential = _sample_differential()
    chunks = _sample_grounding_chunks()

    mock_client = MagicMock()
    mock_parse = AsyncMock()
    mock_parse.return_value = MagicMock(
        parsed_output=_LLMReasoning(explanation="x", reconciliation="y")
    )
    mock_client.messages.parse = mock_parse

    await reason(
        features, rf, differential, None, chunks,
        client=mock_client, timeout_s=2.0,
    )

    assert len(called_with) == 1
    assert called_with[0] is features


# =========================================================================
# INV-8 — degradation
# =========================================================================


async def test_reason_returns_none_on_api_status_error_500() -> None:
    import anthropic
    import httpx

    rf = RedFlagResult(red_flag_activa=False)
    features = _sample_features()
    differential = _sample_differential()
    chunks = _sample_grounding_chunks()

    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx.Response(
        500, request=req,
        content=b'{"error":{"type":"server_error","message":"boom"}}',
    )
    error = anthropic.APIStatusError("Server error", response=resp, body=None)

    mock_client = MagicMock()
    mock_client.messages.parse = AsyncMock(side_effect=error)

    out = await reason(
        features, rf, differential, None, chunks,
        client=mock_client, timeout_s=2.0,
    )
    assert out is None


async def test_reason_returns_none_on_api_connection_error() -> None:
    import httpx
    from anthropic import APIConnectionError

    rf = RedFlagResult(red_flag_activa=False)
    features = _sample_features()
    differential = _sample_differential()
    chunks = _sample_grounding_chunks()

    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    error = APIConnectionError(message="Connection error.", request=req)  # type: ignore[call-arg]

    mock_client = MagicMock()
    mock_client.messages.parse = AsyncMock(side_effect=error)

    out = await reason(
        features, rf, differential, None, chunks,
        client=mock_client, timeout_s=2.0,
    )
    assert out is None


async def test_reason_returns_none_on_rate_limit_exhausted() -> None:
    import httpx
    from anthropic import RateLimitError

    rf = RedFlagResult(red_flag_activa=False)
    features = _sample_features()
    differential = _sample_differential()
    chunks = _sample_grounding_chunks()

    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    content = b'{"error":{"type":"rate_limit_error","message":"slow down"}}'
    resp = httpx.Response(429, request=req, content=content)
    error = RateLimitError("Rate limited", response=resp, body=None)

    mock_client = MagicMock()
    mock_client.messages.parse = AsyncMock(side_effect=error)

    out = await reason(
        features, rf, differential, None, chunks,
        client=mock_client, timeout_s=2.0,
    )
    assert out is None


async def test_reason_returns_none_on_timeout() -> None:
    rf = RedFlagResult(red_flag_activa=False)
    features = _sample_features()
    differential = _sample_differential()
    chunks = _sample_grounding_chunks()

    mock_client = MagicMock()
    mock_client.messages.parse = AsyncMock(side_effect=TimeoutError())

    out = await reason(
        features, rf, differential, None, chunks,
        client=mock_client, timeout_s=0.01,
    )
    assert out is None


# =========================================================================
# Happy path
# =========================================================================


async def test_reason_happy_path_returns_reasoner_output() -> None:
    rf = RedFlagResult(red_flag_activa=False)
    features = _sample_features()
    differential = _sample_differential()
    chunks = _sample_grounding_chunks()

    llm_out = _LLMReasoning(
        explanation="El cuadro sugiere VPPB del canal posterior.",
        reconciliation="Las features son compatibles con los criterios de VPPB.",
        suggested_next_steps=["Realizar maniobra de Epley", "Reevaluar en 48 h"],
        reasoner_suggested_urgency=Urgency.ambulatoria,
    )

    mock_client = MagicMock()
    mock_client.messages.parse = AsyncMock(
        return_value=MagicMock(parsed_output=llm_out)
    )

    out = await reason(
        features, rf, differential, None, chunks,
        client=mock_client, timeout_s=2.0,
    )

    assert out is not None
    assert out.explanation == "El cuadro sugiere VPPB del canal posterior."
    assert out.reconciliation == "Las features son compatibles con los criterios de VPPB."
    assert out.suggested_next_steps == ["Realizar maniobra de Epley", "Reevaluar en 48 h"]
    assert out.model_used == HAIKU
    assert out.reasoner_suggested_urgency == Urgency.ambulatoria
    assert out.grounding_refs == [c.source_id for c in chunks]
    # Haiku 4.5 does NOT use `thinking`: the parameter must be OMITTED (not None).
    # Passing an explicit `thinking=None` is rejected by the API with 400
    # ("thinking: Input should be an object").
    assert "thinking" not in mock_client.messages.parse.call_args.kwargs


async def test_reason_happy_path_uses_opus_with_red_flag() -> None:
    rf = RedFlagResult(red_flag_activa=True)
    features = _sample_features()
    differential = _sample_differential()
    chunks = _sample_grounding_chunks()

    llm_out = _LLMReasoning(explanation="Urgente.", reconciliation="Red flag activa.")

    mock_client = MagicMock()
    mock_client.messages.parse = AsyncMock(
        return_value=MagicMock(parsed_output=llm_out)
    )

    out = await reason(
        features, rf, differential, None, chunks,
        client=mock_client, timeout_s=2.0,
    )

    assert out is not None
    assert out.model_used == OPUS
    # Opus 4.8 DOES use adaptive thinking ({"type": "adaptive"} is mandatory;
    # `budget_tokens` is deprecated and would be rejected with 400).
    assert mock_client.messages.parse.call_args.kwargs.get("thinking") == {
        "type": "adaptive"
    }


async def test_reason_includes_ml_when_provided() -> None:
    rf = RedFlagResult(red_flag_activa=False)
    features = _sample_features()
    differential = _sample_differential()
    chunks = _sample_grounding_chunks()
    ml = PredictResponse(
        probabilities={"bppv_posterior": 0.88, "vestibular_neuritis": 0.07},
        model_version="catboost-v0.1",
    )

    llm_out = _LLMReasoning(explanation="ML respalda VPPB.", reconciliation="Alta concordancia.")

    mock_client = MagicMock()
    mock_client.messages.parse = AsyncMock(
        return_value=MagicMock(parsed_output=llm_out)
    )

    out = await reason(
        features, rf, differential, ml, chunks,
        client=mock_client, timeout_s=2.0,
    )

    assert out is not None
    assert out.explanation == "ML respalda VPPB."


async def test_reason_handler_no_parse_output_raises_and_returns_none() -> None:
    rf = RedFlagResult(red_flag_activa=False)
    features = _sample_features()
    differential = _sample_differential()
    chunks = _sample_grounding_chunks()

    mock_client = MagicMock()
    mock_client.messages.parse = AsyncMock(
        return_value=MagicMock(parsed_output=None)
    )

    out = await reason(
        features, rf, differential, None, chunks,
        client=mock_client, timeout_s=2.0,
    )
    assert out is None


async def test_reason_does_not_convert_suggested_urgency_to_binding() -> None:
    """AD-4/INV-3: reasoner_suggested_urgency is only a suggestion.
    The test verifies that reason() extracts the LLM's value into the
    reasoner_suggested_urgency field, NOT into any binding urgency."""
    rf = RedFlagResult(red_flag_activa=False)
    features = _sample_features()
    differential = _sample_differential()
    chunks = _sample_grounding_chunks()

    llm_out = _LLMReasoning(
        explanation="...",
        reconciliation="...",
        reasoner_suggested_urgency=Urgency.inmediata,
    )

    mock_client = MagicMock()
    mock_client.messages.parse = AsyncMock(
        return_value=MagicMock(parsed_output=llm_out)
    )

    out = await reason(
        features, rf, differential, None, chunks,
        client=mock_client, timeout_s=2.0,
    )

    assert out is not None
    # The reasoner reports the suggestion but does NOT set a binding urgency
    assert out.reasoner_suggested_urgency == Urgency.inmediata
    # reason() has NO "urgency" field — that belongs to rails
    assert not hasattr(out, "urgency")


async def test_reason_returns_none_on_unexpected_exception() -> None:
    rf = RedFlagResult(red_flag_activa=False)
    features = _sample_features()
    differential = _sample_differential()
    chunks = _sample_grounding_chunks()

    mock_client = MagicMock()
    mock_client.messages.parse = AsyncMock(side_effect=ValueError("unexpected"))

    out = await reason(
        features, rf, differential, None, chunks,
        client=mock_client, timeout_s=2.0,
    )
    assert out is None


# =========================================================================
# Import separation (acceptance criterion 3)
# =========================================================================

_FORBIDDEN_REASONER_IMPORTS = {
    "clinibrium.redflag_engine",
    "clinibrium.differential_engine",
    "clinibrium.rails",
    "clinibrium.orchestrator",
    "clinibrium.ml_client",
    "clinibrium.api",
}


def _iter_imports(py_file: Path) -> list[tuple[int, str]]:
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


def test_reasoner_does_not_import_forbidden_modules() -> None:
    pkg_root = Path(__file__).resolve().parents[1] / "clinibrium" / "reasoner"
    offenders: list[str] = []
    for py_file in sorted(pkg_root.glob("*.py")):
        for lineno, mod in _iter_imports(py_file):
            for forbidden in _FORBIDDEN_REASONER_IMPORTS:
                if mod == forbidden or mod.startswith(forbidden + "."):
                    offenders.append(f"{py_file.name}:{lineno} → {mod}")
    assert not offenders, (
        "reasoner imported forbidden modules:\n  " + "\n  ".join(offenders)
    )


def test_reasoner_only_imports_from_allowed_modules() -> None:
    pkg_root = Path(__file__).resolve().parents[1] / "clinibrium" / "reasoner"
    allowed_roots = {
        "clinibrium.contracts", "clinibrium.grounding",
        "clinibrium.config", "clinibrium.reasoner",
    }
    offenders: list[str] = []
    for py_file in sorted(pkg_root.glob("*.py")):
        for lineno, mod in _iter_imports(py_file):
            if not mod.startswith("clinibrium."):
                continue
            if not any(mod == a or mod.startswith(a + ".") for a in allowed_roots):
                offenders.append(f"{py_file.name}:{lineno} → {mod}")
    assert not offenders, (
        "reasoner imported modules outside the allowlist:\n  " + "\n  ".join(offenders)
    )


# =========================================================================
# Grounding refs provenance
# =========================================================================


def test_reasoner_output_grounding_refs_default() -> None:
    out = ReasonerOutput(
        explanation="test",
        reconciliation="test",
        model_used="claude-haiku-4-5-20251001",
    )
    assert out.grounding_refs == []
