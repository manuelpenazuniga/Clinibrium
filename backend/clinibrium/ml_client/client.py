"""Cliente async de `POST /predict` (track B — ML opcional).

Regla dura v7.3 §9 + INV-6: el cliente DEGRADA ELEGANTE. Si B no está
configurado, no responde, da error HTTP, o excede el timeout, devuelve
`None` sin levantar excepciones. El pipeline A completa igual y NUNCA
queda acoplado a la disponibilidad de B.

Importa SOLO de `clinibrium.contracts` + libs (httpx). NO toca engines,
reasoner, orchestrator, ni rails.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from clinibrium.contracts import CaseFeatures, PredictResponse

logger = logging.getLogger(__name__)


def _resolve_base_url(base_url: str | None) -> str | None:
    """Resuelve el `base_url` desde el argumento o desde `Settings`.

    Import diferido de `config` para mantener `ml_client` importable
    incluso si pydantic-settings no estuviera instalado (p.ej. en
    análisis estático aislado).
    """
    if base_url is not None:
        return base_url
    from clinibrium.config import get_settings

    return get_settings().ML_PREDICT_URL


async def predict(
    features: CaseFeatures,
    *,
    timeout_s: float = 2.0,
    base_url: str | None = None,
) -> PredictResponse | None:
    """Llama `POST {base_url}/predict` con `features` y devuelve el
    `PredictResponse` parseado.

    Degrada elegante a `None` (NO levanta) si:
      - `base_url` no está configurado (B desactivado),
      - timeout excedido,
      - respuesta con status >= 400,
      - cualquier excepción de red o de parseo.

    El shape de request/response está CONGELADO — ver
    `docs/CONTRACT_predict.md` (regla dura v7.3 §9).
    """
    resolved = _resolve_base_url(base_url)
    if not resolved:
        logger.debug("ml_client.predict: ML_PREDICT_URL no configurado → degradar (None)")
        return None

    url = f"{resolved.rstrip('/')}/predict"
    # mode="json" serializa enums a sus `.value` (string) y sets a list,
    # exactamente lo que la API de B espera.
    payload: dict[str, Any] = features.model_dump(mode="json")

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return PredictResponse.model_validate(data)
    except httpx.TimeoutException:
        logger.info(
            "ml_client.predict: timeout %.2fs hacia %s → degradar (None)",
            timeout_s,
            url,
        )
        return None
    except httpx.HTTPStatusError as exc:
        logger.info(
            "ml_client.predict: HTTP %s desde %s → degradar (None)",
            exc.response.status_code,
            url,
        )
        return None
    except httpx.HTTPError as exc:
        logger.info(
            "ml_client.predict: error de red (%s) hacia %s → degradar (None)",
            type(exc).__name__,
            url,
        )
        return None
    except Exception as exc:  # noqa: BLE001 — degradación amplia a propósito
        logger.info(
            "ml_client.predict: excepción inesperada (%s) → degradar (None)",
            type(exc).__name__,
        )
        return None
