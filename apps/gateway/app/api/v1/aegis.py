"""Sandbox control-plane endpoints: telemetry SSE and scenario triggers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from redis.exceptions import RedisError
from starlette.responses import StreamingResponse

from app.api.deps import openai_error_body
from app.db.redis import xread_block
from app.responses import ORJSONResponse
from app.sandbox.runner import SimulationBusy
from app.sandbox.scenarios import run_scenario
from app.state import get_state
from app.telemetry.dispatcher import EVENTS_STREAM

router = APIRouter()

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}

_SCENARIOS = frozenset({"burst", "paraphrase", "outage"})


@router.get("/v1/aegis/events")
async def events(request: Request) -> StreamingResponse:
    state = get_state(request)
    last_id = request.headers.get("last-event-id") or "$"

    async def gen() -> AsyncIterator[bytes]:
        cursor = last_id
        yield b": ping\n\n"
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    entries = await xread_block(
                        state.redis_blocking,
                        EVENTS_STREAM,
                        cursor,
                        block_ms=15_000,
                        count=100,
                    )
                except (TimeoutError, RedisError):
                    yield b": ping\n\n"
                    continue
                if not entries:
                    yield b": ping\n\n"
                    continue
                for entry_id, payload in entries:
                    cursor = entry_id.decode("utf-8")
                    yield b"id: " + entry_id + b"\ndata: " + payload + b"\n\n"
        except asyncio.CancelledError:
            raise

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.post("/v1/aegis/simulate/{scenario}")
async def simulate(scenario: str, request: Request) -> ORJSONResponse:
    if scenario not in _SCENARIOS:
        return ORJSONResponse(
            openai_error_body(
                f"Unknown scenario {scenario}",
                error_type="invalid_request_error",
                code="invalid_scenario",
            ),
            status_code=404,
        )
    state = get_state(request)
    try:
        run = state.simulations.begin(scenario)
    except SimulationBusy:
        return ORJSONResponse(
            openai_error_body(
                f"scenario {scenario} is already running",
                error_type="conflict",
                code="scenario_running",
            ),
            status_code=409,
        )
    state.dispatcher.offer_simulation_started(scenario, run.run_id, 0)
    task = asyncio.create_task(run_scenario(state, run), name=f"aegis-sim-{scenario}")
    state.simulations.track(run, task)
    return ORJSONResponse({"run_id": run.run_id}, status_code=202)
