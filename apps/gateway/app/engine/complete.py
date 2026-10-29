"""Turn a hot-path result into an SSE stream or a ChatCompletion JSON body."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from uuid import uuid4

import orjson

from app.engine.cache import CachedCompletion
from app.engine.coerce import object_dict, object_list
from app.engine.failover import open_stream
from app.engine.pipeline import CacheHit, CacheMiss, HotPathResult
from app.engine.router import AffinityRecord, encode_affinity
from app.providers.base import ProviderError
from app.schemas.finops import CacheStatus, FailoverInfo, XAegisFinOps
from app.schemas.openai import ChatCompletionRequest, ChatMessage
from app.sse import (
    CACHE_COMPLETION_ID,
    RelayAccumulator,
    relay_upstream,
    replay_cached,
)
from app.state import AppState
from app.telemetry.dispatcher import (
    AffinityWrite,
    CacheWrite,
    DispatchJob,
    completed_job,
)
from app.telemetry.finops import compute_costs


def new_request_id() -> str:
    return "chatcmpl-aegis-" + uuid4().hex[:20]


def prompt_preview(messages: Sequence[ChatMessage]) -> str:
    text = ""
    for message in reversed(tuple(messages)):
        if message.role == "user":
            text = _content_text(message.content)
            break
    if text == "" and messages:
        text = _content_text(messages[-1].content)
    return text[:80]


async def iter_sse(
    *,
    state: AppState,
    request: ChatCompletionRequest,
    tenant: str,
    result: HotPathResult,
    t0: float,
    request_id: str,
    created: int,
    use_mock: bool,
    run_id: str | None = None,
    scenario: str | None = None,
) -> AsyncGenerator[bytes, None]:
    """Yield SSE frames and enqueue telemetry when the stream finishes."""
    cancelled = False
    job: DispatchJob | None = None
    try:
        if isinstance(result, CacheHit):
            async for frame in _hit_sse(
                state=state,
                tenant=tenant,
                result=result,
                request=request,
                t0=t0,
                request_id=request_id,
                run_id=run_id,
                scenario=scenario,
            ):
                yield frame
            job = _hit_job(
                state=state,
                tenant=tenant,
                result=result,
                request=request,
                t0=t0,
                request_id=request_id,
                run_id=run_id,
                scenario=scenario,
            )
            return
        async for frame, built in _miss_sse(
            state=state,
            request=request,
            tenant=tenant,
            result=result,
            t0=t0,
            request_id=request_id,
            created=created,
            use_mock=use_mock,
            run_id=run_id,
            scenario=scenario,
        ):
            if built is not None:
                job = built
            else:
                yield frame
    except asyncio.CancelledError:
        cancelled = True
        raise
    except ProviderError:
        cancelled = True
        raise
    finally:
        if not cancelled and job is not None:
            state.dispatcher.offer(job)


async def complete_json(
    *,
    state: AppState,
    request: ChatCompletionRequest,
    tenant: str,
    result: HotPathResult,
    t0: float,
    request_id: str,
    created: int,
    use_mock: bool,
    run_id: str | None = None,
    scenario: str | None = None,
) -> tuple[int, dict[str, object]]:
    """Consume the SSE path and assemble a non-stream ChatCompletion body."""
    try:
        frames = [
            frame
            async for frame in iter_sse(
                state=state,
                request=request,
                tenant=tenant,
                result=result,
                t0=t0,
                request_id=request_id,
                created=created,
                use_mock=use_mock,
                run_id=run_id,
                scenario=scenario,
            )
        ]
    except ProviderError as exc:
        return exc.status if 400 <= exc.status < 600 else 502, {
            "error": {
                "message": exc.reason,
                "type": "api_error",
                "code": str(exc.status),
                "param": None,
            }
        }
    return 200, assemble_completion(frames, fallback_id=request_id, created=created)


def assemble_completion(
    frames: list[bytes], *, fallback_id: str, created: int
) -> dict[str, object]:
    """Rebuild a `chat.completion` object from SSE frames."""
    content_parts: list[str] = []
    finops: dict[str, object] | None = None
    model = "auto"
    finish = "stop"
    tokens_in = 0
    tokens_out = 0
    completion_id = fallback_id
    error_body: dict[str, object] | None = None
    for frame in frames:
        text = frame.decode("utf-8")
        for line in text.split("\n"):
            if not line.startswith("data:"):
                continue
            data = line[5:].lstrip()
            if data == "" or data == "[DONE]":
                continue
            try:
                parsed: object = orjson.loads(data)
            except orjson.JSONDecodeError:
                continue
            payload = object_dict(parsed)
            if payload is None:
                continue
            err = object_dict(payload.get("error"))
            if err is not None:
                error_body = {"error": err}
                continue
            cid = payload.get("id")
            if isinstance(cid, str) and cid:
                completion_id = cid
            mid = payload.get("model")
            if isinstance(mid, str) and mid:
                model = mid
            extra = object_dict(payload.get("x_aegis_finops"))
            if extra is not None:
                finops = extra
            usage = object_dict(payload.get("usage"))
            if usage is not None:
                prompt = usage.get("prompt_tokens")
                completion = usage.get("completion_tokens")
                if isinstance(prompt, int) and not isinstance(prompt, bool):
                    tokens_in = prompt
                if isinstance(completion, int) and not isinstance(completion, bool):
                    tokens_out = completion
            choices = object_list(payload.get("choices")) or []
            if not choices:
                continue
            choice = object_dict(choices[0]) or {}
            finish_raw = choice.get("finish_reason")
            if isinstance(finish_raw, str):
                finish = finish_raw
            delta = object_dict(choice.get("delta")) or {}
            piece = delta.get("content")
            if isinstance(piece, str):
                content_parts.append(piece)
    if error_body is not None and not content_parts:
        return error_body
    if finops is not None:
        tin = finops.get("tokens_in")
        tout = finops.get("tokens_out")
        if isinstance(tin, int) and not isinstance(tin, bool):
            tokens_in = tin
        if isinstance(tout, int) and not isinstance(tout, bool):
            tokens_out = tout
    body: dict[str, object] = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "".join(content_parts)},
                "finish_reason": finish,
            }
        ],
        "usage": {
            "prompt_tokens": tokens_in,
            "completion_tokens": tokens_out,
            "total_tokens": tokens_in + tokens_out,
        },
    }
    if finops is not None:
        body["x_aegis_finops"] = finops
    return body


async def _hit_sse(
    *,
    state: AppState,
    tenant: str,
    result: CacheHit,
    request: ChatCompletionRequest,
    t0: float,
    request_id: str,
    run_id: str | None,
    scenario: str | None,
) -> AsyncIterator[bytes]:
    _ = tenant, request, request_id, run_id, scenario, state
    overhead = (time.perf_counter() - t0) * 1000.0
    finops = _hit_finops(state, result, overhead_ms=overhead, ttft_ms=overhead)
    for frame in replay_cached(result.completion, finops):
        yield frame


def _hit_job(
    *,
    state: AppState,
    tenant: str,
    result: CacheHit,
    request: ChatCompletionRequest,
    t0: float,
    request_id: str,
    run_id: str | None,
    scenario: str | None,
) -> DispatchJob:
    overhead = (time.perf_counter() - t0) * 1000.0
    finops = _hit_finops(
        state,
        result,
        overhead_ms=overhead,
        ttft_ms=overhead,
        request_id=CACHE_COMPLETION_ID,
    )
    return completed_job(
        request_id=request_id,
        tenant_id=tenant,
        conversation_id=None,
        finops=finops,
        breakdown=result.breakdown,
        total_ms=(time.perf_counter() - t0) * 1000.0,
        status="ok",
        prompt_preview=prompt_preview(request.messages),
        run_id=run_id,
        scenario=scenario,
        cache=None,
        affinity=None,
    )


async def _miss_sse(
    *,
    state: AppState,
    request: ChatCompletionRequest,
    tenant: str,
    result: CacheMiss,
    t0: float,
    request_id: str,
    created: int,
    use_mock: bool,
    run_id: str | None,
    scenario: str | None,
) -> AsyncIterator[tuple[bytes, DispatchJob | None]]:
    opened = await open_stream(
        result.route,
        request,
        messages=result.scrubbed_messages,
        request_id=request_id,
        created=created,
        registry=state.providers,
        breakers=state.breakers,
        use_mock=use_mock,
    )
    overhead = (opened.sent_at - t0) * 1000.0
    ttft = (time.perf_counter() - t0) * 1000.0
    accum = RelayAccumulator()

    def make_finops(state_acc: RelayAccumulator) -> XAegisFinOps:
        return _miss_finops(
            state,
            result,
            provider=opened.provider,
            failover=opened.failover,
            accum=state_acc,
            overhead_ms=overhead,
            ttft_ms=ttft,
            request_id=request_id,
        )

    async for frame in relay_upstream(
        opened.first,
        opened.rest,
        request_id=request_id,
        created=created,
        model=result.route.model.name,
        accum=accum,
        finops_factory=make_finops,
    ):
        yield frame, None

    status = "error" if accum.error is not None else "ok"
    cacheable = (
        accum.error is None
        and not accum.tool_call
        and request.tools is None
        and accum.finish_reason == "stop"
    )
    tokens_in, tokens_out = _tokens(accum, result.scrubbed_messages, accum.content)
    completion = CachedCompletion(
        content=accum.content,
        model=result.route.model.name,
        provider=opened.provider,
        tier=result.route.tier,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        finish_reason=accum.finish_reason,
        created_at=created,
    )
    finops = make_finops(accum)
    cache: CacheWrite | None = None
    if cacheable:
        cache = CacheWrite(
            l1_key=result.keys.l1_key,
            completion=completion,
            tenant_id=tenant,
            sys_hash=result.keys.sys_hash,
            vector=result.vector,
            write_l2=result.keys.l2_eligible,
        )
    affinity: AffinityWrite | None = None
    if accum.error is None and result.keys.conversation_id is not None:
        affinity = AffinityWrite(
            key=f"aff:{tenant}:{result.keys.conversation_id}",
            payload=encode_affinity(
                AffinityRecord(
                    tier=result.route.tier,
                    provider=opened.provider,
                    model=result.route.model.name,
                )
            ),
            ttl_seconds=state.settings.cache_ttl_seconds,
        )
    job = completed_job(
        request_id=request_id,
        tenant_id=tenant,
        conversation_id=result.keys.conversation_id,
        finops=finops,
        breakdown=result.breakdown,
        total_ms=(time.perf_counter() - t0) * 1000.0,
        status=status,
        prompt_preview=prompt_preview(result.scrubbed_messages),
        run_id=run_id,
        scenario=scenario,
        cache=cache,
        affinity=affinity,
    )
    yield b"", job


def _hit_finops(
    state: AppState,
    result: CacheHit,
    *,
    overhead_ms: float,
    ttft_ms: float,
    request_id: str = CACHE_COMPLETION_ID,
) -> XAegisFinOps:
    actual_usd, saved_usd = compute_costs(
        result.completion.tokens_in,
        result.completion.tokens_out,
        None,
        state.router.catalog.baseline,
        cache_hit=True,
    )
    return XAegisFinOps(
        cache_status=result.status,
        similarity_score=result.similarity,
        routed_tier=result.completion.tier,
        routed_model=result.completion.model,
        latency_breakdown_ms=result.breakdown,
        cost_actual_usd=actual_usd,
        cost_saved_usd=saved_usd,
        request_id=request_id,
        provider=result.completion.provider,
        failover=None,
        gateway_overhead_ms=overhead_ms,
        ttft_ms=ttft_ms,
        tokens_in=result.completion.tokens_in,
        tokens_out=result.completion.tokens_out,
    )


def _miss_finops(
    state: AppState,
    result: CacheMiss,
    *,
    provider: str,
    failover: FailoverInfo | None,
    accum: RelayAccumulator,
    overhead_ms: float,
    ttft_ms: float,
    request_id: str,
) -> XAegisFinOps:
    tokens_in, tokens_out = _tokens(accum, result.scrubbed_messages, accum.content)
    actual_usd, saved_usd = compute_costs(
        tokens_in,
        tokens_out,
        result.route.model,
        state.router.catalog.baseline,
        cache_hit=False,
    )
    return XAegisFinOps(
        cache_status=CacheStatus.MISS,
        similarity_score=None,
        routed_tier=result.route.tier,
        routed_model=result.route.model.name,
        latency_breakdown_ms=result.breakdown,
        cost_actual_usd=actual_usd,
        cost_saved_usd=saved_usd,
        request_id=request_id,
        provider=provider,
        failover=failover,
        gateway_overhead_ms=overhead_ms,
        ttft_ms=ttft_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )


def _tokens(
    accum: RelayAccumulator,
    messages: Sequence[ChatMessage],
    content: str,
) -> tuple[int, int]:
    tokens_in = accum.tokens_in
    tokens_out = accum.tokens_out
    if tokens_in <= 0:
        tokens_in = max(
            1, (sum(len(_content_text(m.content)) for m in messages) + 3) // 4
        )
    if tokens_out <= 0:
        tokens_out = max(1, (len(content) + 3) // 4)
    return tokens_in, tokens_out


def _content_text(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return ""
