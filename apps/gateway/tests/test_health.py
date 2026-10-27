"""Liveness and readiness probes."""

from __future__ import annotations

from httpx import AsyncClient


async def test_healthz(client: AsyncClient) -> None:
    response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_readyz(client: AsyncClient) -> None:
    response = await client.get("/readyz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["onnx_warm"] is True
    assert body["warmup_ms"] > 0
    assert body["redis"] == "ok"
    assert body["postgres"] == "ok"
