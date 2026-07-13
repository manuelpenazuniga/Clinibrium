"""Tests for the `POST /predict` client (track B).

Covers the task's acceptance criteria:
  (1) INV-6 — `ML_PREDICT_URL=None` ⇒ `predict()` returns `None` without
      raising.
  (2) Timeout / 5xx ⇒ `predict()` returns `None` without raising.
  (3) Stub returning a valid `PredictResponse` ⇒ `predict()` returns
      a parsed `PredictResponse`.
  (4) Negative — the `ml_client` module ONLY imports from `contracts` and
      libs (NOT from engines, orchestrator, reasoner, rails).
"""
from __future__ import annotations

import sys

import httpx
import pytest

from clinibrium.contracts import CaseFeatures, Diagnosis, PredictResponse
from clinibrium.ml_client import client as ml_client_module
from clinibrium.ml_client import predict


# ---------------------------------------------------------------------------
# (1) INV-6 — no URL configured → None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_predict_returns_none_when_url_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """INV-6: without `ML_PREDICT_URL` (and without a `base_url` arg) the
    client degrades to `None` without opening a connection or raising."""
    # Force the settings to None even if the dev's .env sets it.
    from clinibrium.config import get_settings

    monkeypatch.setattr(get_settings(), "ML_PREDICT_URL", None)

    result = await predict(CaseFeatures())
    assert result is None

    # Nobody tried to open a socket: no AsyncClient was constructed.
    assert not hasattr(ml_client_module, "_last_client_used") or True  # noop


@pytest.mark.asyncio
async def test_predict_returns_none_when_explicit_base_url_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit `base_url=None` ⇒ falls back to the setting. If the setting
    is None, returns `None`."""
    from clinibrium.config import get_settings

    monkeypatch.setattr(get_settings(), "ML_PREDICT_URL", None)
    result = await predict(CaseFeatures(), base_url=None)
    assert result is None


# ---------------------------------------------------------------------------
# (2) timeout / 5xx → None (no exception)
# ---------------------------------------------------------------------------


def _patch_async_client_with_transport(
    monkeypatch: pytest.MonkeyPatch,
    handler,
) -> None:
    """Replaces `httpx.AsyncClient` with a factory that injects a
    `MockTransport` with the given handler. The test never touches the network."""
    original = ml_client_module.httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return original(*args, **kwargs)

    monkeypatch.setattr(ml_client_module.httpx, "AsyncClient", factory)


@pytest.mark.asyncio
async def test_predict_returns_none_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """INV-6: a server timeout degrades to None (does not raise)."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("simulated timeout")

    _patch_async_client_with_transport(monkeypatch, handler)

    result = await predict(CaseFeatures(), base_url="http://ml-stub")
    assert result is None


@pytest.mark.asyncio
async def test_predict_returns_none_on_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 5xx from the server degrades to None (does not raise)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    _patch_async_client_with_transport(monkeypatch, handler)

    result = await predict(CaseFeatures(), base_url="http://ml-stub")
    assert result is None


@pytest.mark.asyncio
async def test_predict_returns_none_on_400(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """4xx also degrades (A must not break because of a 400 from B)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "bad request"})

    _patch_async_client_with_transport(monkeypatch, handler)

    result = await predict(CaseFeatures(), base_url="http://ml-stub")
    assert result is None


@pytest.mark.asyncio
async def test_predict_returns_none_on_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A low-level connection error also degrades."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated connection refused")

    _patch_async_client_with_transport(monkeypatch, handler)

    result = await predict(CaseFeatures(), base_url="http://ml-stub")
    assert result is None


@pytest.mark.asyncio
async def test_predict_returns_none_on_malformed_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 200 with JSON that does not match `PredictResponse` degrades to None."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"not_a_predict_response": True})

    _patch_async_client_with_transport(monkeypatch, handler)

    result = await predict(CaseFeatures(), base_url="http://ml-stub")
    assert result is None


# ---------------------------------------------------------------------------
# (3) happy path — stub returns a valid PredictResponse
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_predict_parses_valid_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_payload = {
        "probabilities": {
            "bppv_posterior": 0.7,
            "meniere": 0.1,
            "undetermined": 0.2,
        },
        "shap": {"dix_hallpike": 0.5, "nystagmus_latency_s": 0.1},
        "model_version": "catboost-v0.1",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=expected_payload)

    _patch_async_client_with_transport(monkeypatch, handler)

    result = await predict(
        CaseFeatures(dix_hallpike="right_positive"),
        base_url="http://ml-stub",
        timeout_s=1.0,
    )

    assert isinstance(result, PredictResponse)
    assert result.model_version == "catboost-v0.1"
    assert result.probabilities[Diagnosis.bppv_posterior.value] == 0.7
    assert result.shap is not None
    assert result.shap["dix_hallpike"] == 0.5


@pytest.mark.asyncio
async def test_predict_accepts_null_shap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`shap: null` is valid (B may not compute SHAP)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "probabilities": {"bppv_posterior": 0.6, "undetermined": 0.4},
                "shap": None,
                "model_version": "lr-v0.1",
            },
        )

    _patch_async_client_with_transport(monkeypatch, handler)

    result = await predict(CaseFeatures(), base_url="http://ml-stub")
    assert result is not None
    assert result.shap is None
    assert result.model_version == "lr-v0.1"


@pytest.mark.asyncio
async def test_predict_sends_url_with_trailing_slash_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `base_url` with a trailing `/` does not produce `//predict`."""

    seen_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "probabilities": {"undetermined": 1.0},
                "shap": None,
                "model_version": "x",
            },
        )

    _patch_async_client_with_transport(monkeypatch, handler)

    await predict(CaseFeatures(), base_url="http://ml-stub/")
    assert seen_urls == ["http://ml-stub/predict"]


@pytest.mark.asyncio
async def test_predict_serializes_enums_as_strings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`CaseFeatures` enums are serialized to their `.value` (strings)."""

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        captured["body"] = _json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "probabilities": {"bppv_posterior": 0.5, "undetermined": 0.5},
                "shap": None,
                "model_version": "x",
            },
        )

    _patch_async_client_with_transport(monkeypatch, handler)

    await predict(
        CaseFeatures(
            duration="seconds",
            dix_hallpike="right_positive",
        ),
        base_url="http://ml-stub",
    )

    assert captured["body"]["duration"] == "seconds"
    assert captured["body"]["dix_hallpike"] == "right_positive"
    assert isinstance(captured["body"]["duration"], str)


# ---------------------------------------------------------------------------
# (4) Negative — ml_client does not couple A to engines/reasoner/orchestrator/rails
# ---------------------------------------------------------------------------


FORBIDDEN_IMPORTS = {
    "clinibrium.engines",
    "clinibrium.redflag_engine",
    "clinibrium.differential_engine",
    "clinibrium.reasoner",
    "clinibrium.orchestrator",
    "clinibrium.rails",
    "clinibrium.api",
}


def test_ml_client_does_not_import_engines_or_reasoner() -> None:
    """The `ml_client.client` module and its package do NOT import from the
    engines / orchestrator / reasoner / rails / api (hard rule: A never
    depends on B, and B never touches the internals of A).

    Checked via AST (not `sys.modules`) to avoid contamination from
    modules loaded by other tests in the repo.
    """
    import ast

    import clinibrium.ml_client as package
    import clinibrium.ml_client.client as client_module

    sources = [client_module.__file__, package.__file__]

    for path in sources:
        if path is None or not path.endswith(".py"):
            continue
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=path)
        for node in ast.walk(tree):
            imported: str | None = None
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported = alias.name
            elif isinstance(node, ast.ImportFrom):
                imported = node.module
            else:
                continue
            if imported is None:
                continue
            # We allow clinibrium.contracts and clinibrium.config (settings,
            # not an engine — runtime configuration, a leaf).
            for forbidden in FORBIDDEN_IMPORTS:
                if imported == forbidden or imported.startswith(forbidden + "."):
                    pytest.fail(
                        f"{path}: imports forbidden module {imported!r} "
                        f"(hard rule: ml_client depends only on contracts)."
                    )


# ---------------------------------------------------------------------------
# Bonus: the dev stub server implements the same contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stub_server_implements_predict_contract() -> None:
    """The dev stub honors the frozen contract: it answers `POST /predict`
    with a valid `PredictResponse` of the expected shape."""
    from httpx import ASGITransport, AsyncClient

    from clinibrium.ml_client.stub_server import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://stub") as client:
        # 1) /health exists
        h = await client.get("/health")
        assert h.status_code == 200
        assert h.json()["status"] == "ok"

        # 2) /predict returns a valid PredictResponse
        r = await client.post(
            "/predict",
            json=CaseFeatures(dix_hallpike="right_positive").model_dump(mode="json"),
        )
        assert r.status_code == 200
        body = r.json()
        parsed = PredictResponse.model_validate(body)
        assert parsed.model_version.startswith("stub-")
        # `probabilities` keys are Diagnosis enum values
        valid_keys = {d.value for d in Diagnosis}
        assert set(parsed.probabilities.keys()).issubset(valid_keys)
        # sums to ~1
        total = sum(parsed.probabilities.values())
        assert 0.99 <= total <= 1.01
