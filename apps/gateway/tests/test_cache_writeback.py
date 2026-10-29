"""Cache writeback: populate on success, skip on disconnect, error, and tools."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from uuid import uuid4

from httpx import AsyncClient

from app.engine.complete import iter_sse, new_request_id
from app.engine.pipeline import CacheMiss, cache_keys_for, run_hot_path
from app.providers.registry import MOCK_PRIMARY, MOCK_SECONDARY, use_mock_providers
from app.schemas.finops import CacheStatus
from app.schemas.openai import ChatCompletionRequest, ChatMessage
from app.state import AppState

_AUTH = {"Authorization": "Bearer sk-aegis-sandbox"}


def _isolated_request(prompt: str, *, tools: list[dict[str, object]] | None = None):
    return ChatCompletionRequest(
        model="auto",
        temperature=0.2,
        messages=[
            ChatMessage(role="system", content=f"iso-{uuid4().hex}"),
            ChatMessage(role="user", content=prompt),
        ],
        tools=tools,
    )


def _payload(request: ChatCompletionRequest) -> dict[str, object]:
    messages = [
        {"role": message.role, "content": message.content}
        for message in request.messages
    ]
    body: dict[str, object] = {
        "model": "auto",
        "stream": False,
        "temperature": 0.2,
        "messages": messages,
    }
    if request.tools is not None:
        body["tools"] = request.tools
    return body


async def test_miss_populates_l1_and_l2(
    client: AsyncClient, gateway_state: AppState
) -> None:
    request = _isolated_request(f"writeback populate {uuid4().hex}")
    keys = cache_keys_for(
        request,
        "sandbox",
        request.messages,
        embed_max_chars=gateway_state.settings.embed_max_chars,
    )
    response = await client.post(
        "/v1/chat/completions",
        headers=_AUTH,
        json=_payload(request),
    )
    assert response.status_code == 200, response.text
    finops = response.json()["x_aegis_finops"]
    assert finops["cache_status"] == CacheStatus.MISS.value
    await gateway_state.dispatcher.wait_idle()
    assert await gateway_state.cache.get_l1(keys.l1_key) is not None
    vector = await gateway_state.embedder.embed(keys.embed_text)
    l2 = await gateway_state.cache.get_l2(
        tenant_id="sandbox", sys_hash=keys.sys_hash, vector=vector
    )
    assert l2 is not None


async def test_client_disconnect_caches_nothing(gateway_state: AppState) -> None:
    request = _isolated_request(f"writeback disconnect {uuid4().hex}")
    keys = cache_keys_for(
        request,
        "sandbox",
        request.messages,
        embed_max_chars=gateway_state.settings.embed_max_chars,
    )
    result = await run_hot_path(request, "sandbox", gateway_state)
    assert isinstance(result, CacheMiss)
    gateway_state.chaos.stall(MOCK_PRIMARY, after_chunks=1, sleep_s=60.0)
    gen: AsyncGenerator[bytes, None] = iter_sse(
        state=gateway_state,
        request=request,
        tenant="sandbox",
        result=result,
        t0=time.perf_counter(),
        request_id=new_request_id(),
        created=int(time.time()),
        use_mock=use_mock_providers("sandbox", gateway_state.settings, None),
    )
    try:
        first = await gen.__anext__()
        assert first.startswith(b"data:")
        await gen.aclose()
    finally:
        gateway_state.chaos.clear()
    await gateway_state.dispatcher.wait_idle()
    assert await gateway_state.cache.get_l1(keys.l1_key) is None


async def test_error_caches_nothing(
    client: AsyncClient, gateway_state: AppState
) -> None:
    request = _isolated_request(f"writeback error {uuid4().hex}")
    keys = cache_keys_for(
        request,
        "sandbox",
        request.messages,
        embed_max_chars=gateway_state.settings.embed_max_chars,
    )
    gateway_state.chaos.inject(MOCK_PRIMARY, 503, count=8)
    gateway_state.chaos.inject(MOCK_SECONDARY, 503, count=8)
    try:
        response = await client.post(
            "/v1/chat/completions",
            headers=_AUTH,
            json=_payload(request),
        )
        assert response.status_code >= 400
    finally:
        gateway_state.chaos.clear()
    await gateway_state.dispatcher.wait_idle()
    assert await gateway_state.cache.get_l1(keys.l1_key) is None


async def test_tool_call_request_is_not_cached(
    client: AsyncClient, gateway_state: AppState
) -> None:
    tools: list[dict[str, object]] = [
        {
            "type": "function",
            "function": {"name": "lookup", "parameters": {"type": "object"}},
        }
    ]
    request = _isolated_request(f"writeback tools {uuid4().hex}", tools=tools)
    keys = cache_keys_for(
        request,
        "sandbox",
        request.messages,
        embed_max_chars=gateway_state.settings.embed_max_chars,
    )
    response = await client.post(
        "/v1/chat/completions",
        headers=_AUTH,
        json=_payload(request),
    )
    assert response.status_code == 200
    await gateway_state.dispatcher.wait_idle()
    assert await gateway_state.cache.get_l1(keys.l1_key) is None
