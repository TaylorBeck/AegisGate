"""OpenAI-compatible adapter parses SSE fixtures via MockTransport."""

from __future__ import annotations

import httpx
import orjson
import pytest

from app.providers.base import ProviderError
from app.providers.openai_compat import OpenAICompatProvider
from app.schemas.openai import ChatCompletionRequest, ChatMessage

_SSE = (
    b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,'
    b'"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant","content":""},'
    b'"finish_reason":null}]}\n\n'
    b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,'
    b'"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"Hola"},'
    b'"finish_reason":null}]}\n\n'
    b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,'
    b'"model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,'
    b'"model":"gpt-4o-mini","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":1,'
    b'"total_tokens":6}}\n\n'
    b"data: [DONE]\n\n"
)


@pytest.mark.asyncio
async def test_forwards_content_and_usage() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        body = orjson.loads(request.content)
        assert body["stream"] is True
        assert body["stream_options"]["include_usage"] is True
        assert body["model"] == "gpt-4o-mini"
        return httpx.Response(
            200,
            content=_SSE,
            headers={"Content-Type": "text/event-stream"},
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatProvider(
        "openai", http, base_url="https://api.openai.com/v1", api_key="sk-test"
    )
    request = ChatCompletionRequest(
        model="gpt-4o-mini",
        messages=[ChatMessage(role="user", content="hi")],
        stream=True,
    )
    chunks = [
        chunk
        async for chunk in provider.stream(
            model="gpt-4o-mini",
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
    usage = [chunk for chunk in chunks if chunk.tokens_in is not None]
    assert usage[-1].tokens_in == 5
    assert usage[-1].tokens_out == 1


@pytest.mark.asyncio
async def test_retryable_http_error() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b'{"error":{"message":"unavailable"}}')

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatProvider(
        "openai", http, base_url="https://example.test/v1", api_key="sk-test"
    )
    request = ChatCompletionRequest(
        model="gpt-4o-mini",
        messages=[ChatMessage(role="user", content="hi")],
    )
    with pytest.raises(ProviderError) as caught:
        async for _ in provider.stream(
            model="gpt-4o-mini",
            messages=request.messages,
            request=request,
            request_id="x",
            created=1,
        ):
            pass
    await http.aclose()
    assert caught.value.status == 503
    assert caught.value.retryable is True
