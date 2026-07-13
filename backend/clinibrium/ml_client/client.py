"""Async client for `POST /predict` (track B — optional ML).

Hard rule v7.3 §9 + INV-6: the client DEGRADES GRACEFULLY. If B is not
configured, does not answer, returns an HTTP error, or exceeds the
timeout, it returns `None` without raising exceptions. Pipeline A
completes regardless and is NEVER coupled to B's availability.

Imports ONLY from `clinibrium.contracts` + libs (httpx). Does NOT touch
engines, reasoner, orchestrator, or rails.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from clinibrium.contracts import CaseFeatures, PredictResponse

logger = logging.getLogger(__name__)


def _resolve_base_url(base_url: str | None) -> str | None:
    """Resolves `base_url` from the argument or from `Settings`.

    Deferred import of `config` to keep `ml_client` importable even if
    pydantic-settings were not installed (e.g. in isolated static
    analysis).
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
    """Calls `POST {base_url}/predict` with `features` and returns the
    parsed `PredictResponse`.

    Degrades gracefully to `None` (does NOT raise) if:
      - `base_url` is not configured (B disabled),
      - timeout exceeded,
      - response with status >= 400,
      - any network or parsing exception.

    The request/response shape is FROZEN — see
    `docs/CONTRACT_predict.md` (hard rule v7.3 §9).
    """
    resolved = _resolve_base_url(base_url)
    if not resolved:
        logger.debug("ml_client.predict: ML_PREDICT_URL not configured → degrade (None)")
        return None

    url = f"{resolved.rstrip('/')}/predict"
    # mode="json" serializes enums to their `.value` (string) and sets to
    # list, exactly what B's API expects.
    payload: dict[str, Any] = features.model_dump(mode="json")

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return PredictResponse.model_validate(data)
    except httpx.TimeoutException:
        logger.warning(
            "ml_client.predict: timeout %.2fs towards %s → degrade (None)",
            timeout_s,
            url,
        )
        return None
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "ml_client.predict: HTTP %s from %s → degrade (None)",
            exc.response.status_code,
            url,
        )
        return None
    except httpx.HTTPError as exc:
        logger.warning(
            "ml_client.predict: network error (%s) towards %s → degrade (None)",
            type(exc).__name__,
            url,
        )
        return None
    except Exception as exc:  # noqa: BLE001 — deliberately broad degradation
        logger.warning(
            "ml_client.predict: unexpected exception (%s) → degrade (None)",
            type(exc).__name__,
        )
        return None
