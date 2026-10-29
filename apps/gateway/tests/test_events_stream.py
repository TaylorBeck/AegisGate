"""`GET /v1/aegis/events` honors Last-Event-ID and emits JSON payloads.

httpx ASGITransport deadlocks on infinite SSE, so this test drives the ASGI
app directly and sends `http.disconnect` after the target event arrives.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from typing import Any, cast
from uuid import uuid4

from app.db.redis import xread_block
from app.main import app
from app.state import AppState
from app.telemetry.dispatcher import EVENTS_STREAM


async def test_events_replay_from_last_event_id(gateway_state: AppState) -> None:
    run_id = uuid4().hex[:16]
    gateway_state.dispatcher.offer_simulation_started("burst", run_id, 0)
    await gateway_state.dispatcher.wait_idle()
    entries = await xread_block(
        gateway_state.redis_blocking,
        EVENTS_STREAM,
        "0-0",
        block_ms=1000,
        count=2000,
    )
    assert any(run_id.encode() in payload for _entry_id, payload in entries)

    needle = run_id.encode()
    bodies = await _read_asgi_sse(
        "/v1/aegis/events",
        headers={"last-event-id": "0-0", "accept": "text/event-stream"},
        stop_when=lambda chunk: needle in chunk and b"simulation.started" in chunk,
    )
    joined = b"".join(bodies)
    assert needle in joined
    assert b"simulation.started" in joined
    assert b"text/event-stream" in joined


async def test_events_idle_ping(gateway_state: AppState) -> None:
    bodies = await _read_asgi_sse(
        "/v1/aegis/events",
        headers={"accept": "text/event-stream"},
        stop_when=lambda chunk: b": ping" in chunk,
    )
    joined = b"".join(bodies)
    assert b": ping" in joined
    assert b"text/event-stream" in joined


async def _read_asgi_sse(
    path: str,
    *,
    headers: dict[str, str],
    stop_when: Callable[[bytes], bool],
    limit_s: float = 5.0,
) -> list[bytes]:
    header_pairs = [
        (key.lower().encode("latin-1"), value.encode("latin-1"))
        for key, value in headers.items()
    ]
    scope: dict[str, object] = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": header_pairs,
        "client": ("test", 50000),
        "server": ("test", 80),
    }
    incoming: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    await incoming.put({"type": "http.request", "body": b"", "more_body": False})
    bodies: list[bytes] = []
    loop = asyncio.get_running_loop()
    started = loop.time()
    disconnected = False

    async def receive() -> dict[str, object]:
        return await incoming.get()

    async def send(message: dict[str, object]) -> None:
        nonlocal disconnected
        kind = message.get("type")
        if kind == "http.response.start":
            bodies.append(b"headers:text/event-stream")
        if kind == "http.response.body":
            body = message.get("body")
            if isinstance(body, bytes):
                bodies.append(body)
                if stop_when(b"".join(bodies)) and not disconnected:
                    disconnected = True
                    await incoming.put({"type": "http.disconnect"})

    asgi = cast(Any, app)
    task = asyncio.create_task(asgi(scope, receive, send))
    try:
        while loop.time() - started < limit_s:
            if disconnected or task.done():
                break
            await asyncio.sleep(0.02)
        if not disconnected:
            disconnected = True
            await incoming.put({"type": "http.disconnect"})
        await asyncio.wait_for(task, timeout=2.0)
    finally:
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task
    return bodies
