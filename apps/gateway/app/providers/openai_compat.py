"""Generic OpenAI-compatible `/chat/completions` streaming adapter."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Sequence

import httpx
import orjson

from app.engine.coerce import object_dict, object_list
from app.providers.base import (
    Provider,
    ProviderError,
    UpstreamChunk,
    is_retryable_status,
)
from app.schemas.finops import Tier
from app.schemas.openai import ChatCompletionRequest, ChatMessage


class OpenAICompatProvider(Provider):
    """One named instance with its own base URL and API key."""

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
        _ = request_id, created, tier
        url = f"{self._base_url}/chat/completions"
        payload = _request_body(model, messages, request)
        headers = {
            "Authorization": f"Bearer {self._api_key}",
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
                        _error_reason(response.status_code, body),
                    )
                first = True
                async for line in response.aiter_lines():
                    chunk = _parse_data_line(line, sent_at if first else None)
                    if chunk is None:
                        continue
                    first = False
                    yield chunk
        except httpx.TimeoutException as exc:
            raise ProviderError(408, True, "upstream timeout") from exc
        except httpx.TransportError as exc:
            raise ProviderError(503, True, "upstream transport error") from exc


def _request_body(
    model: str,
    messages: Sequence[ChatMessage],
    request: ChatCompletionRequest,
) -> dict[str, object]:
    dumped = request.model_dump(mode="json", exclude_none=True)
    extra = request.model_extra or {}
    body: dict[str, object] = {str(key): value for key, value in dumped.items()}
    for key, value in extra.items():
        body[str(key)] = value
    body["model"] = model
    body["messages"] = [
        {
            str(key): value
            for key, value in message.model_dump(mode="json", exclude_none=True).items()
        }
        for message in messages
    ]
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    return body


def _parse_data_line(line: str, sent_at: float | None) -> UpstreamChunk | None:
    if not line.startswith("data:"):
        return None
    data = line[5:].lstrip()
    if data == "" or data == "[DONE]":
        return None
    raw = data.encode("utf-8")
    try:
        parsed: object = orjson.loads(raw)
    except orjson.JSONDecodeError:
        return UpstreamChunk(raw=raw, sent_at=sent_at)
    return _from_openai_payload(raw, parsed, sent_at)


def _from_openai_payload(
    raw: bytes, parsed: object, sent_at: float | None
) -> UpstreamChunk:
    body = object_dict(parsed) or {}
    error = object_dict(body.get("error"))
    if error is not None:
        message = error.get("message")
        reason = message if isinstance(message, str) else "upstream error"
        raise ProviderError(500, True, reason)
    content = ""
    finish: str | None = None
    tool_call = False
    choices = object_list(body.get("choices")) or []
    if choices:
        choice = object_dict(choices[0]) or {}
        finish_raw = choice.get("finish_reason")
        if isinstance(finish_raw, str):
            finish = finish_raw
        delta = (
            object_dict(choice.get("delta")) or object_dict(choice.get("message")) or {}
        )
        text = delta.get("content")
        if isinstance(text, str):
            content = text
        if (
            delta.get("tool_calls") is not None
            or delta.get("function_call") is not None
        ):
            tool_call = True
    tokens_in, tokens_out = _usage(body.get("usage"))
    return UpstreamChunk(
        raw=raw,
        content_delta=content,
        finish_reason=finish,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        tool_call=tool_call,
        sent_at=sent_at,
    )


def _usage(raw: object) -> tuple[int | None, int | None]:
    body = object_dict(raw)
    if body is None:
        return None, None
    prompt = body.get("prompt_tokens")
    completion = body.get("completion_tokens")
    tokens_in = (
        prompt if isinstance(prompt, int) and not isinstance(prompt, bool) else None
    )
    tokens_out = (
        completion
        if isinstance(completion, int) and not isinstance(completion, bool)
        else None
    )
    return tokens_in, tokens_out


def _error_reason(status: int, body: str) -> str:
    preview = body[:200].replace("\n", " ")
    if preview:
        return f"HTTP {status}: {preview}"
    return f"HTTP {status}"
