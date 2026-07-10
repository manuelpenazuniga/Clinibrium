"""Test async del endpoint /health."""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from clinibrium.api import create_app


@pytest.mark.asyncio
async def test_health_ok() -> None:
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "clinibrium"
