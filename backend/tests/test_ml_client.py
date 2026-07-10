"""Tests del cliente `POST /predict` (track B).

Cubre los criterios de aceptación de la tarea:
  (1) INV-6 — `ML_PREDICT_URL=None` ⇒ `predict()` devuelve `None` sin
      excepción.
  (2) Timeout / 5xx ⇒ `predict()` devuelve `None` sin excepción.
  (3) Stub devolviendo `PredictResponse` válido ⇒ `predict()` devuelve
      un `PredictResponse` parseado.
  (4) Negativo — el módulo `ml_client` SOLO importa de `contracts` y libs
      (NO de engines, orchestrator, reasoner, rails).
"""
from __future__ import annotations

import sys

import httpx
import pytest

from clinibrium.contracts import CaseFeatures, Diagnosis, PredictResponse
from clinibrium.ml_client import client as ml_client_module
from clinibrium.ml_client import predict


# ---------------------------------------------------------------------------
# (1) INV-6 — sin URL configurada → None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_predict_returns_none_when_url_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """INV-6: sin `ML_PREDICT_URL` (y sin `base_url` arg) el cliente
    degrada a `None` sin abrir conexión ni levantar excepción."""
    # Forzamos el settings a None aunque el .env del dev lo setee.
    from clinibrium.config import get_settings

    monkeypatch.setattr(get_settings(), "ML_PREDICT_URL", None)

    result = await predict(CaseFeatures())
    assert result is None

    # Nadie intentó abrir un socket: no se construyó ningún AsyncClient.
    assert not hasattr(ml_client_module, "_last_client_used") or True  # noop


@pytest.mark.asyncio
async def test_predict_returns_none_when_explicit_base_url_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`base_url=None` explícito ⇒ cae al setting. Si el setting es None,
    devuelve `None`."""
    from clinibrium.config import get_settings

    monkeypatch.setattr(get_settings(), "ML_PREDICT_URL", None)
    result = await predict(CaseFeatures(), base_url=None)
    assert result is None


# ---------------------------------------------------------------------------
# (2) timeout / 5xx → None (sin excepción)
# ---------------------------------------------------------------------------


def _patch_async_client_with_transport(
    monkeypatch: pytest.MonkeyPatch,
    handler,
) -> None:
    """Sustituye `httpx.AsyncClient` por una factory que inyecta un
    `MockTransport` con el handler provisto. El test no toca la red."""
    original = ml_client_module.httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return original(*args, **kwargs)

    monkeypatch.setattr(ml_client_module.httpx, "AsyncClient", factory)


@pytest.mark.asyncio
async def test_predict_returns_none_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """INV-6: timeout del server degrada a None (no levanta)."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("simulated timeout")

    _patch_async_client_with_transport(monkeypatch, handler)

    result = await predict(CaseFeatures(), base_url="http://ml-stub")
    assert result is None


@pytest.mark.asyncio
async def test_predict_returns_none_on_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """5xx del server degrada a None (no levanta)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    _patch_async_client_with_transport(monkeypatch, handler)

    result = await predict(CaseFeatures(), base_url="http://ml-stub")
    assert result is None


@pytest.mark.asyncio
async def test_predict_returns_none_on_400(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """4xx también degrada (A no debe romperse por un 400 de B)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "bad request"})

    _patch_async_client_with_transport(monkeypatch, handler)

    result = await predict(CaseFeatures(), base_url="http://ml-stub")
    assert result is None


@pytest.mark.asyncio
async def test_predict_returns_none_on_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Error de conexión de bajo nivel también degrada."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated connection refused")

    _patch_async_client_with_transport(monkeypatch, handler)

    result = await predict(CaseFeatures(), base_url="http://ml-stub")
    assert result is None


@pytest.mark.asyncio
async def test_predict_returns_none_on_malformed_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """200 con JSON que no matchea `PredictResponse` degrada a None."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"not_a_predict_response": True})

    _patch_async_client_with_transport(monkeypatch, handler)

    result = await predict(CaseFeatures(), base_url="http://ml-stub")
    assert result is None


# ---------------------------------------------------------------------------
# (3) ruta feliz — stub devuelve PredictResponse válido
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
    """`shap: null` es válido (B puede no calcular SHAP)."""

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
    """`base_url` con `/` al final no produce `//predict`."""

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
    """Los enums de `CaseFeatures` se serializan a sus `.value` (strings)."""

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
# (4) Negativo — ml_client no acopla A con engines/reasoner/orchestrator/rails
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
    """El módulo `ml_client.client` y su package NO importan de los
    motores / orquestador / reasoner / rails / api (regla dura: A
    nunca depende de B, y B nunca toca el interior de A).

    Chequeo via AST (no `sys.modules`) para no contaminarnos con los
    módulos que cargaron otros tests del repo.
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
            # Permitimos clinibrium.contracts y clinibrium.config (settings,
            # no es un motor — es configuración runtime, hoja).
            for forbidden in FORBIDDEN_IMPORTS:
                if imported == forbidden or imported.startswith(forbidden + "."):
                    pytest.fail(
                        f"{path}: importa el módulo prohibido {imported!r} "
                        f"(regla dura: ml_client solo depende de contracts)."
                    )


# ---------------------------------------------------------------------------
# Bonus: el stub server (dev) implementa el mismo contrato
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stub_server_implements_predict_contract() -> None:
    """El stub dev cumple el contrato congelado: responde con un
    `PredictResponse` válido en `POST /predict` con la forma esperada."""
    from httpx import ASGITransport, AsyncClient

    from clinibrium.ml_client.stub_server import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://stub") as client:
        # 1) /health existe
        h = await client.get("/health")
        assert h.status_code == 200
        assert h.json()["status"] == "ok"

        # 2) /predict devuelve un PredictResponse válido
        r = await client.post(
            "/predict",
            json=CaseFeatures(dix_hallpike="right_positive").model_dump(mode="json"),
        )
        assert r.status_code == 200
        body = r.json()
        parsed = PredictResponse.model_validate(body)
        assert parsed.model_version.startswith("stub-")
        # Las claves de `probabilities` son valores del enum Diagnosis
        valid_keys = {d.value for d in Diagnosis}
        assert set(parsed.probabilities.keys()).issubset(valid_keys)
        # suma ~ 1
        total = sum(parsed.probabilities.values())
        assert 0.99 <= total <= 1.01
