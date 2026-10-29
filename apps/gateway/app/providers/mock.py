"""Deterministic mock providers with optional chaos and token-speed pacing."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Sequence

import orjson

from app.providers.base import (
    Provider,
    ProviderError,
    UpstreamChunk,
    is_retryable_status,
)
from app.sandbox.chaos import ChaosController
from app.schemas.finops import Tier
from app.schemas.openai import ChatCompletionRequest, ChatMessage

_TEMPLATES: dict[Tier, str] = {
    Tier.SIMPLE: (
        "Sure — here is the concise answer: {hint} "
        "This SIMPLE mock completion is deterministic so cache replay stays stable."
    ),
    Tier.REASONING: (
        "Let's reason step by step. First, identify the constraints in the prompt. "
        "Second, weigh the trade-offs. Final answer: {hint}."
    ),
    Tier.AGENTIC_CODE: (
        "```python\n"
        "def solve():\n"
        "    return {hint!r}\n"
        "```\n"
        "The AGENTIC_CODE mock returns a deterministic implementation sketch."
    ),
}


class MockProvider(Provider):
    """`mock-primary` / `mock-secondary`. Chaos only affects these instances."""

    def __init__(
        self,
        name: str,
        chaos: ChaosController,
        *,
        ttft_ms: float,
        tokens_per_sec: float,
    ) -> None:
        self.name = name
        self.last_sent_at = 0.0
        self._chaos = chaos
        self._ttft_ms = ttft_ms
        self._tokens_per_sec = tokens_per_sec if tokens_per_sec > 0 else 8_000.0

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
        _ = request
        fault = self._chaos.fault_for(self.name)
        if fault is not None:
            raise ProviderError(
                fault,
                is_retryable_status(fault),
                f"{self.name} injected HTTP {fault}",
            )
        self.last_sent_at = time.perf_counter()
        sent_at = self.last_sent_at
        if self._ttft_ms > 0:
            await asyncio.sleep(self._ttft_ms / 1000.0)

        chosen = tier if tier is not None else Tier.SIMPLE
        hint = _last_user_text(messages)[:80] or "ok"
        content = _TEMPLATES[chosen].format(hint=hint)
        tokens_in = _estimate_tokens(_flatten_messages(messages))
        tokens_out = _estimate_tokens(content)
        stall = self._chaos.stall_for(self.name)
        stall_after, stall_s = stall if stall is not None else (-1, 0.0)

        first = True
        emitted = 0
        step = 24
        delay = 0.0
        if self._tokens_per_sec > 0:
            delay = (step / 4.0) / self._tokens_per_sec
        for index in range(0, len(content), step):
            piece = content[index : index + step]
            delta: dict[str, object] = {}
            if first:
                delta["role"] = "assistant"
                first = False
            delta["content"] = piece
            raw = _chunk(
                request_id,
                created,
                model,
                [{"index": 0, "delta": delta, "finish_reason": None}],
            )
            yield UpstreamChunk(
                raw=raw,
                content_delta=piece,
                sent_at=sent_at if emitted == 0 else None,
            )
            emitted += 1
            if stall_after >= 0 and emitted >= stall_after and stall_s > 0:
                await asyncio.sleep(stall_s)
            if delay > 0:
                await asyncio.sleep(delay)

        yield UpstreamChunk(
            raw=_chunk(
                request_id,
                created,
                model,
                [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            ),
            finish_reason="stop",
        )
        usage_raw = _chunk(
            request_id,
            created,
            model,
            [],
            usage={
                "prompt_tokens": tokens_in,
                "completion_tokens": tokens_out,
                "total_tokens": tokens_in + tokens_out,
            },
        )
        yield UpstreamChunk(raw=usage_raw, tokens_in=tokens_in, tokens_out=tokens_out)


def _chunk(
    request_id: str,
    created: int,
    model: str,
    choices: list[dict[str, object]],
    usage: dict[str, int] | None = None,
) -> bytes:
    body: dict[str, object] = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": choices,
    }
    if usage is not None:
        body["usage"] = usage
    return orjson.dumps(body)


def _last_user_text(messages: Sequence[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role != "user":
            continue
        return _content_text(message.content)
    if not messages:
        return ""
    return _content_text(messages[-1].content)


def _flatten_messages(messages: Sequence[ChatMessage]) -> str:
    return "\n".join(_content_text(message.content) for message in messages)


def _content_text(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return ""


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)
