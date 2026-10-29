"""Telemetry dispatcher: ledger rows and queue-full drops."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from httpx import AsyncClient

from app.db.postgres import fetch_ledger_by_request_id
from app.schemas.finops import CacheStatus, LatencyBreakdown, Tier, XAegisFinOps
from app.state import AppState
from app.telemetry.dispatcher import DispatchJob, TelemetryDispatcher, completed_job

_AUTH = {"Authorization": "Bearer sk-aegis-sandbox"}


async def test_completed_request_lands_in_ledger(
    client: AsyncClient, gateway_state: AppState
) -> None:
    prompt = f"dispatcher ledger {uuid4().hex}"
    response = await client.post(
        "/v1/chat/completions",
        headers=_AUTH,
        json={
            "model": "auto",
            "stream": False,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": f"ledger-{uuid4().hex}"},
                {"role": "user", "content": prompt},
            ],
        },
    )
    assert response.status_code == 200
    body = response.json()
    request_id = body["x_aegis_finops"]["request_id"]
    await gateway_state.dispatcher.wait_idle()
    row = await fetch_ledger_by_request_id(gateway_state.pg, request_id)
    assert row is not None
    assert row["tenant_id"] == "sandbox"
    assert row["cache_status"] == "MISS"
    assert row["status"] == "ok"


async def test_queue_full_drops(gateway_state: AppState) -> None:
    dispatcher = TelemetryDispatcher(
        gateway_state.redis,
        gateway_state.pg,
        gateway_state.cache,
        maxsize=1,
    )
    job = DispatchJob(event=b'{"type":"breaker.state"}')
    dispatcher.offer(job)
    dispatcher.offer(job)
    assert dispatcher.dropped == 1


def test_completed_job_builds_ledger_tuple() -> None:
    finops = XAegisFinOps(
        cache_status=CacheStatus.MISS,
        similarity_score=None,
        routed_tier=Tier.SIMPLE,
        routed_model="grok-3-mini",
        latency_breakdown_ms=LatencyBreakdown(guardrail=0.1, l1_lookup=0.2),
        cost_actual_usd=0.001,
        cost_saved_usd=0.01,
        request_id="chatcmpl-x",
        provider="mock-primary",
        tokens_in=10,
        tokens_out=4,
    )
    job = completed_job(
        request_id="chatcmpl-x",
        tenant_id="sandbox",
        conversation_id=None,
        finops=finops,
        breakdown=finops.latency_breakdown_ms,
        total_ms=12.0,
        status="ok",
        prompt_preview="hello",
        run_id=None,
        scenario=None,
        cache=None,
        affinity=None,
    )
    assert job.ledger is not None
    assert job.ledger[0] == "chatcmpl-x"
    assert isinstance(job.ledger[1], datetime)
    assert job.ledger[1].tzinfo is UTC
    assert job.ledger[-1] == "ok"
