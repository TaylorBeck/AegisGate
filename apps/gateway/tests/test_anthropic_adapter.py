"""Anthropic Messages adapter maps SSE events onto OpenAI chunks."""

from __future__ import annotations

import httpx
import orjson
import pytest

from app.providers.anthropic import AnthropicProvider
from app.schemas.openai import ChatCompletionRequest, ChatMessage

_SSE = (
    b"event: message_start\n"
    b'data: {"type":"message_start","message":{"id":"msg_1","type":"message",'
    b'"role":"assistant","content":[],"model":"claude-sonnet-4.5",'
    b'"usage":{"input_tokens":8,"output_tokens":1}}}\n\n'
    b"event: content_block_delta\n"
    b'data: {"type":"content_block_delta","index":0,'
    b'"delta":{"type":"text_delta","text":"Hola"}}\n\n'
    b"event: message_delta\n"
    b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
    b'"usage":{"output_tokens":1}}\n\n'
    b"event: message_stop\n"
    b'data: {"type":"message_stop"}\n\n'
)


@pytest.mark.asyncio
async def test_maps_anthropic_events() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/v1/messages")
        assert request.headers["x-api-key"] == "sk-ant-test"
        body = orjson.loads(request.content)
        assert body["stream"] is True
        assert body["system"] == "be brief"
        assert body["messages"][0]["role"] == "user"
        return httpx.Response(
            200,
            content=_SSE,
            headers={"Content-Type": "text/event-stream"},
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(
        "anthropic",
        http,
        base_url="https://api.anthropic.com",
        api_key="sk-ant-test",
    )
    request = ChatCompletionRequest(
        model="claude-sonnet-4.5",
        messages=[
            ChatMessage(role="system", content="be brief"),
            ChatMessage(role="user", content="hi"),
        ],
        max_tokens=64,
    )
    chunks = [
        chunk
        async for chunk in provider.stream(
            model="claude-sonnet-4.5",
            messages=request.messages,
            request=request,
            request_id="chatcmpl-1",
            created=1,
        )
    ]
    await http.aclose()
    content = "".join(chunk.content_delta for chunk in chunks)
    assert content == "Hola"
    assert any(chunk.finish_reason == "stop" for chunk in chunks)
    assert any(chunk.tokens_in == 8 for chunk in chunks)
    assert any(chunk.tokens_out == 1 for chunk in chunks)
    raw = orjson.loads(chunks[1].raw)
    assert raw["object"] == "chat.completion.chunk"
    assert raw["choices"][0]["delta"]["content"] == "Hola"
