"""OpenAI-compatible SSE framing, cache replay, and upstream relay."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

import orjson

from app.engine.cache import CachedCompletion
from app.providers.base import ProviderError, UpstreamChunk
from app.schemas.finops import XAegisFinOps

CACHE_COMPLETION_ID = "chatcmpl-aegis-cache"
_DELTA_CHARS = 24


@dataclass(slots=True)
class RelayAccumulator:
    """Parsed fields collected while relaying an upstream stream."""

    content: str = ""
    finish_reason: str = "stop"
    tokens_in: int = 0
    tokens_out: int = 0
    tool_call: bool = False
    error: ProviderError | None = None
    saw_token: bool = False


def format_chunk(data: object) -> bytes:
    """Encode one SSE `data:` frame, including `[DONE]`."""
    if data == "[DONE]":
        return b"data: [DONE]\n\n"
    if isinstance(data, bytes):
        return b"data: " + data + b"\n\n"
    return b"data: " + orjson.dumps(data) + b"\n\n"


def replay_cached(completion: CachedCompletion, finops: XAegisFinOps) -> list[bytes]:
    """24-character deltas, the cache metadata chunk, then `[DONE]`."""
    frames: list[bytes] = []
    text = completion.content
    first = True
    span = max(len(text), 1)
    for index in range(0, span, _DELTA_CHARS):
        piece = text[index : index + _DELTA_CHARS]
        delta: dict[str, object] = {}
        if first:
            delta["role"] = "assistant"
            first = False
        if piece:
            delta["content"] = piece
        frames.append(
            format_chunk(
                _chunk_body(
                    CACHE_COMPLETION_ID,
                    completion.created_at,
                    completion.model,
                    [{"index": 0, "delta": delta, "finish_reason": None}],
                )
            )
        )
    frames.append(
        format_chunk(
            _chunk_body(
                CACHE_COMPLETION_ID,
                completion.created_at,
                completion.model,
                [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                extra={"x_aegis_finops": finops.model_dump(mode="json")},
            )
        )
    )
    frames.append(format_chunk("[DONE]"))
    return frames


async def relay_upstream(
    first: UpstreamChunk,
    rest: AsyncIterator[UpstreamChunk],
    *,
    request_id: str,
    created: int,
    model: str,
    accum: RelayAccumulator,
    finops_factory: Callable[[RelayAccumulator], XAegisFinOps],
) -> AsyncIterator[bytes]:
    """Forward upstream payloads, then emit the MISS metadata chunk and `[DONE]`.

    A failure after the first token yields an OpenAI-style error frame instead
    of raising, so the HTTP stream stays intact. `accum.error` is set.
    """
    absorb(accum, first)
    yield format_chunk(first.raw)
    try:
        async for chunk in rest:
            absorb(accum, chunk)
            yield format_chunk(chunk.raw)
    except ProviderError as exc:
        accum.error = exc
        yield format_chunk(
            {
                "error": {
                    "message": exc.reason,
                    "type": "api_error",
                    "code": str(exc.status),
                }
            }
        )
        yield format_chunk("[DONE]")
        return
    finops = finops_factory(accum)
    yield format_chunk(
        _chunk_body(
            request_id,
            created,
            model,
            [],
            extra={"x_aegis_finops": finops.model_dump(mode="json")},
        )
    )
    yield format_chunk("[DONE]")


def absorb(accum: RelayAccumulator, chunk: UpstreamChunk) -> None:
    if chunk.content_delta:
        accum.content += chunk.content_delta
        accum.saw_token = True
    if chunk.finish_reason:
        accum.finish_reason = chunk.finish_reason
    if chunk.tokens_in is not None:
        accum.tokens_in = chunk.tokens_in
    if chunk.tokens_out is not None:
        accum.tokens_out = chunk.tokens_out
    if chunk.tool_call:
        accum.tool_call = True


def _chunk_body(
    request_id: str,
    created: int,
    model: str,
    choices: list[dict[str, object]],
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    body: dict[str, object] = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": choices,
    }
    if extra:
        body.update(extra)
    return body
