"""Provider protocol, errors, and upstream SSE chunks."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from app.schemas.finops import Tier
from app.schemas.openai import ChatCompletionRequest, ChatMessage


@dataclass(frozen=True, slots=True)
class UpstreamChunk:
    """One OpenAI-shaped `data:` payload plus parsed fields for accumulation."""

    raw: bytes
    content_delta: str = ""
    finish_reason: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    tool_call: bool = False
    sent_at: float | None = None


class ProviderError(Exception):
    """Retryable or terminal upstream failure before or during a stream."""

    def __init__(self, status: int, retryable: bool, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.retryable = retryable
        self.reason = reason


class Provider:
    """Streaming chat provider. Named instances share the Protocol surface."""

    name: str
    last_sent_at: float

    def stream(
        self,
        *,
        model: str,
        messages: Sequence[ChatMessage],
        request: ChatCompletionRequest,
        request_id: str,
        created: int,
        tier: Tier | None = None,
    ) -> AsyncIterator[UpstreamChunk]:
        raise NotImplementedError


def is_retryable_status(status: int) -> bool:
    """429 and 5xx fail over; other 4xx do not."""
    return status == 408 or status == 409 or status == 429 or status >= 500
