"""Provider-chain failover with circuit breakers.

A failure before the first token is transparent. After the first token the
caller owns the error chunk; this module only opens the stream.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from app.engine.router import RouteDecision
from app.providers.base import ProviderError, UpstreamChunk
from app.providers.breaker import BreakerSet, CircuitBreaker
from app.providers.registry import MOCK_CHAIN, ProviderRegistry
from app.schemas.finops import FailoverInfo
from app.schemas.openai import ChatCompletionRequest, ChatMessage


@dataclass(slots=True)
class OpenStream:
    """First upstream chunk plus the remainder of the stream."""

    provider: str
    first: UpstreamChunk
    rest: AsyncIterator[UpstreamChunk]
    failover: FailoverInfo | None
    sent_at: float


async def open_stream(
    route: RouteDecision,
    request: ChatCompletionRequest,
    *,
    messages: Sequence[ChatMessage],
    request_id: str,
    created: int,
    registry: ProviderRegistry,
    breakers: BreakerSet,
    use_mock: bool,
) -> OpenStream:
    """Try providers in chain order, skipping OPEN breakers."""
    chain = MOCK_CHAIN if use_mock else route.provider_chain
    last_error: ProviderError | None = None
    failover: FailoverInfo | None = None
    attempted = False
    for name in chain:
        breaker = breakers.get(name)
        if breaker is not None and not breaker.allow():
            continue
        provider = registry.get(name)
        if provider is None:
            last_error = ProviderError(503, True, f"unknown provider {name}")
            failover = FailoverInfo(from_provider=name, reason=last_error.reason)
            continue
        attempted = True
        try:
            stream = provider.stream(
                model=route.model.name,
                messages=messages,
                request=request,
                request_id=request_id,
                created=created,
                tier=route.tier,
            )
            first = await anext(stream)
        except StopAsyncIteration as exc:
            error = ProviderError(502, True, f"{name} produced an empty stream")
            _record_failure(breaker)
            last_error = error
            failover = FailoverInfo(from_provider=name, reason=error.reason)
            _ = exc
            continue
        except ProviderError as exc:
            _record_failure(breaker)
            last_error = exc
            failover = FailoverInfo(from_provider=name, reason=exc.reason)
            if not exc.retryable:
                raise
            continue
        _record_success(breaker)
        sent_at = first.sent_at if first.sent_at is not None else provider.last_sent_at
        if sent_at <= 0.0:
            sent_at = time.perf_counter()
        return OpenStream(
            provider=name,
            first=first,
            rest=stream,
            failover=failover,
            sent_at=sent_at,
        )
    if not attempted:
        raise ProviderError(503, True, "all providers are open")
    if last_error is not None:
        raise last_error
    raise ProviderError(503, True, "no upstream provider available")


def _record_success(breaker: CircuitBreaker | None) -> None:
    if breaker is not None:
        breaker.record_success()


def _record_failure(breaker: CircuitBreaker | None) -> None:
    if breaker is not None:
        breaker.record_failure()
