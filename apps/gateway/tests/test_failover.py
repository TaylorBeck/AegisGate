"""Transparent failover before the first token, plus an injected 503 path."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Sequence
from uuid import uuid4

import pytest
from httpx import AsyncClient

from app.catalog import ModelSpec
from app.engine.failover import open_stream
from app.engine.router import RouteDecision
from app.providers.base import Provider, ProviderError, UpstreamChunk
from app.providers.breaker import BreakerSet, CircuitBreaker
from app.providers.registry import MOCK_PRIMARY, MOCK_SECONDARY, ProviderRegistry
from app.schemas.finops import Tier
from app.schemas.openai import ChatCompletionRequest, ChatMessage
from app.state import AppState

_AUTH = {"Authorization": "Bearer sk-aegis-sandbox"}


class _FakeProvider(Provider):
    def __init__(
        self,
        name: str,
        *,
        error: ProviderError | None = None,
        text: str = "ok",
    ) -> None:
        self.name = name
        self.last_sent_at = 0.0
        self._error = error
        self._text = text

    async def stream(
        self,
        *,
        model: str,
        messages: Sequence[ChatMessage],
        request: ChatCompletionRequest,
        request_id: str,
        created: int,
        tier: Tier | None = None,
    ) -> AsyncIterator[UpstreamChunk]:
        _ = messages, request, tier
        self.last_sent_at = time.perf_counter()
        if self._error is not None:
            raise self._error
        raw = (
            b'{"id":"%s","object":"chat.completion.chunk","created":%d,'
            b'"model":"%s","choices":[{"index":0,"delta":{"content":"%s"},'
            b'"finish_reason":null}]}'
            % (
                request_id.encode(),
                created,
                model.encode(),
                self._text.encode(),
            )
        )
        yield UpstreamChunk(
            raw=raw,
            content_delta=self._text,
            sent_at=self.last_sent_at,
        )
        yield UpstreamChunk(
            raw=b'{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
            finish_reason="stop",
            tokens_in=1,
            tokens_out=1,
        )


def _route() -> RouteDecision:
    spec = ModelSpec(
        name="grok-3-mini",
        provider=MOCK_PRIMARY,
        tier=Tier.SIMPLE,
        price_in_per_m=0.3,
        price_out_per_m=0.5,
    )
    return RouteDecision(
        tier=Tier.SIMPLE,
        model=spec,
        provider_chain=(MOCK_PRIMARY, MOCK_SECONDARY),
        sticky=False,
        escalated=False,
        scores={Tier.SIMPLE: 1.0, Tier.REASONING: 0.0, Tier.AGENTIC_CODE: 0.0},
    )


@pytest.mark.asyncio
async def test_failover_before_first_token_under_15ms() -> None:
    primary = _FakeProvider(
        MOCK_PRIMARY,
        error=ProviderError(503, True, "injected 503"),
    )
    secondary = _FakeProvider(MOCK_SECONDARY, text="from-secondary")
    registry = ProviderRegistry({MOCK_PRIMARY: primary, MOCK_SECONDARY: secondary})
    breakers = BreakerSet(
        {
            MOCK_PRIMARY: CircuitBreaker(MOCK_PRIMARY),
            MOCK_SECONDARY: CircuitBreaker(MOCK_SECONDARY),
        }
    )
    request = ChatCompletionRequest(
        model="auto",
        messages=[ChatMessage(role="user", content="hi")],
    )
    started = time.perf_counter()
    opened = await open_stream(
        _route(),
        request,
        messages=request.messages,
        request_id="chatcmpl-x",
        created=1,
        registry=registry,
        breakers=breakers,
        use_mock=True,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    assert opened.provider == MOCK_SECONDARY
    assert opened.first.content_delta == "from-secondary"
    assert opened.failover is not None
    assert opened.failover.from_provider == MOCK_PRIMARY
    assert elapsed_ms < 15.0


async def test_http_failover_is_one_uninterrupted_stream(
    client: AsyncClient, gateway_state: AppState
) -> None:
    gateway_state.chaos.inject(MOCK_PRIMARY, 503, count=1)
    prompt = f"failover {uuid4().hex}"
    try:
        response = await client.post(
            "/v1/chat/completions",
            headers=_AUTH,
            json={
                "model": "auto",
                "stream": True,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": f"fail-{uuid4().hex}"},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        assert response.status_code == 200
        text = response.text
        assert "data: [DONE]" in text
        assert '"error"' not in text.split("x_aegis_finops")[0]
        assert "from_provider" in text
        assert MOCK_PRIMARY in text
    finally:
        gateway_state.chaos.clear()
        await gateway_state.dispatcher.wait_idle()
