"""Cliente de POST /predict (track B); degrada elegante.

API pública:
    predict(features, *, timeout_s=2.0, base_url=None) -> PredictResponse | None

Si el servicio B no está configurado, no responde, da error HTTP, o
excede el timeout → devuelve `None` (NO levanta). El pipeline A
completa igual (INV-6).
"""
from __future__ import annotations

from clinibrium.ml_client.client import predict

__all__ = ["predict"]
