"""Native Anthropic Messages streaming adapter mapped onto OpenAI chunks."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Sequence

import httpx
import orjson

from app.engine.coerce import object_dict
from app.providers.base import (
    Provider,
    ProviderError,
    UpstreamChunk,
    is_retryable_status,
)
from app.schemas.finops import Tier
from app.schemas.openai import ChatCompletionRequest, ChatMessage

_STOP = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
}


class AnthropicProvider(Provider):
    """Translates Anthropic SSE events into `chat.completion.chunk` payloads."""

    def __init__(
        self,
        name: str,
        http: httpx.AsyncClient,
        *,
        base_url: str,
        api_key: str,
    ) -> None:
        self.name = name
        self.last_sent_at = 0.0
        self._http = http
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

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
        _ = tier
        url = f"{self._base_url}/v1/messages"
        payload = _messages_body(model, messages, request)
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        self.last_sent_at = time.perf_counter()
        sent_at = self.last_sent_at
        try:
            async with self._http.stream(
                "POST",
                url,
                content=orjson.dumps(payload),
                headers=headers,
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    raise ProviderError(
                        response.status_code,
                        is_retryable_status(response.status_code),
                        f"HTTP {response.status_code}: {body[:200]}",
                    )
                event_name = "message"
                first = True
                async for line in response.aiter_lines():
                    if line.startswith("event:"):
                        event_name = line[6:].strip()
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].lstrip()
                    if data == "" or data == "[DONE]":
                        continue
                    mapped = _map_event(
                        event_name,
                        data,
                        request_id=request_id,
                        created=created,
                        model=model,
                        sent_at=sent_at if first else None,
                    )
                    if mapped is None:
                        continue
                    first = False
                    yield mapped
        except httpx.TimeoutException as exc:
            raise ProviderError(408, True, "upstream timeout") from exc
        except httpx.TransportError as exc:
            raise ProviderError(503, True, "upstream transport error") from exc


def _messages_body(
    model: str,
    messages: Sequence[ChatMessage],
    request: ChatCompletionRequest,
) -> dict[str, object]:
    system_parts: list[str] = []
    converted: list[dict[str, object]] = []
    for message in messages:
        text = _content_text(message.content)
        if message.role == "system":
            if text:
                system_parts.append(text)
            continue
        role = "assistant" if message.role == "assistant" else "user"
        converted.append({"role": role, "content": text})
    body: dict[str, object] = {
        "model": model,
        "max_tokens": request.max_tokens if request.max_tokens is not None else 1024,
        "stream": True,
        "messages": converted,
    }
    if system_parts:
        body["system"] = "\n".join(system_parts)
    if request.temperature is not None:
        body["temperature"] = request.temperature
    return body


def _map_event(
    event_name: str,
    data: str,
    *,
    request_id: str,
    created: int,
    model: str,
    sent_at: float | None,
) -> UpstreamChunk | None:
    try:
        parsed: object = orjson.loads(data)
    except orjson.JSONDecodeError:
        return None
    body = object_dict(parsed) or {}
    kind = body.get("type")
    if not isinstance(kind, str):
        kind = event_name

    if kind == "message_start":
        message = object_dict(body.get("message")) or {}
        usage = object_dict(message.get("usage")) or {}
        tokens_in = _int(usage.get("input_tokens"))
        raw = _openai_chunk(
            request_id,
            created,
            model,
            [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
        )
        return UpstreamChunk(raw=raw, tokens_in=tokens_in, sent_at=sent_at)

    if kind == "content_block_delta":
        delta = object_dict(body.get("delta")) or {}
        text = delta.get("text")
        piece = text if isinstance(text, str) else ""
        raw = _openai_chunk(
            request_id,
            created,
            model,
            [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
        )
        return UpstreamChunk(raw=raw, content_delta=piece)

    if kind == "message_delta":
        delta = object_dict(body.get("delta")) or {}
        usage = object_dict(body.get("usage")) or {}
        stop = delta.get("stop_reason")
        finish = _STOP.get(stop, "stop") if isinstance(stop, str) else None
        tokens_out = _int(usage.get("output_tokens"))
        raw = _openai_chunk(
            request_id,
            created,
            model,
            [{"index": 0, "delta": {}, "finish_reason": finish}],
            usage={"completion_tokens": tokens_out} if tokens_out is not None else None,
        )
        return UpstreamChunk(raw=raw, finish_reason=finish, tokens_out=tokens_out)

    if kind == "message_stop":
        raw = _openai_chunk(
            request_id,
            created,
            model,
            [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        )
        return UpstreamChunk(raw=raw, finish_reason="stop")

    return None


def _openai_chunk(
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


def _content_text(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return ""


def _int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value
