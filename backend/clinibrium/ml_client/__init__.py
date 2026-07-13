"""POST /predict client (track B); degrades gracefully.

Public API:
    predict(features, *, timeout_s=2.0, base_url=None) -> PredictResponse | None

If service B is not configured, does not answer, returns an HTTP error,
or exceeds the timeout → returns `None` (does NOT raise). Pipeline A
completes regardless (INV-6).
"""
from __future__ import annotations

from clinibrium.ml_client.client import predict

__all__ = ["predict"]
