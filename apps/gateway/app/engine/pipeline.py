"""Hot path with no provider I/O: scrub, admit, embed, search, route."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.engine.admission import AdmissionDeniedError
from app.engine.cache import CachedCompletion, decode_completion, require_tenant
from app.engine.keys import CacheKeys, derive_cache_keys
from app.engine.router import RouteDecision, decode_affinity
from app.engine.timing import StageTimer
from app.schemas.finops import CacheStatus, LatencyBreakdown
from app.schemas.openai import ChatCompletionRequest, ChatMessage
from app.state import AppState


@dataclass(frozen=True, slots=True)
class CacheHit:
    """L1 or L2 hit. The cached tier is the routed tier."""

    completion: CachedCompletion
    status: CacheStatus
    similarity: float | None
    breakdown: LatencyBreakdown


@dataclass(frozen=True, slots=True)
class CacheMiss:
    """Cache miss ready for the provider failover loop."""

    keys: CacheKeys
    vector: NDArray[np.float32]
    route: RouteDecision
    scrubbed_messages: list[ChatMessage]
    breakdown: LatencyBreakdown


type HotPathResult = CacheHit | CacheMiss


def cache_keys_for(
    request: ChatCompletionRequest,
    tenant_id: str,
    messages: Sequence[ChatMessage],
    *,
    embed_max_chars: int,
) -> CacheKeys:
    """Derive keys from the request policy and the supplied (scrubbed) messages."""
    return derive_cache_keys(
        tenant_id=tenant_id,
        model=request.model,
        messages=messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        tools=request.tools,
        response_format=request.response_format,
        embed_max_chars=embed_max_chars,
    )


async def run_hot_path(
    request: ChatCompletionRequest,
    tenant: str,
    state: AppState,
    *,
    now_ms: int | None = None,
    open_providers: frozenset[str] = frozenset(),
) -> HotPathResult:
    """Scrub, admit, and either replay a cache hit or return a route decision."""
    require_tenant(tenant)
    timer = StageTimer()

    timer.start("guardrail")
    try:
        scrubbed = state.guardrail.scrub_messages(request.messages)
    finally:
        timer.stop("guardrail")

    keys = cache_keys_for(
        request,
        tenant,
        scrubbed,
        embed_max_chars=state.settings.embed_max_chars,
    )
    affinity_key = ""
    if len(scrubbed) > 2 and keys.conversation_id is not None:
        affinity_key = f"aff:{tenant}:{keys.conversation_id}"

    timer.start("l1_lookup")
    try:
        admission = await state.admission.check(
            tenant_id=tenant,
            l1_key=keys.l1_key,
            affinity_key=affinity_key,
            now_ms=now_ms,
        )
    finally:
        timer.stop("l1_lookup")

    if not admission.allowed:
        raise AdmissionDeniedError(tenant)

    if admission.l1_payload is not None:
        return CacheHit(
            completion=decode_completion(admission.l1_payload),
            status=CacheStatus.HIT_L1,
            similarity=None,
            breakdown=timer.breakdown(),
        )

    timer.start("onnx_embed")
    try:
        vector = await state.embedder.embed(keys.embed_text)
    finally:
        timer.stop("onnx_embed")

    if keys.l2_eligible:
        timer.start("l2_hnsw")
        try:
            l2_hit = await state.cache.get_l2(
                tenant_id=tenant,
                sys_hash=keys.sys_hash,
                vector=vector,
            )
        finally:
            timer.stop("l2_hnsw")
        if l2_hit is not None:
            completion, similarity = l2_hit
            return CacheHit(
                completion=completion,
                status=CacheStatus.HIT_L2,
                similarity=similarity,
                breakdown=timer.breakdown(),
            )

    affinity = None
    if len(scrubbed) > 2:
        affinity = decode_affinity(admission.affinity_payload)

    timer.start("router")
    try:
        route = state.router.resolve(
            vector,
            message_count=len(scrubbed),
            affinity=affinity,
            open_providers=open_providers,
        )
    finally:
        timer.stop("router")

    return CacheMiss(
        keys=keys,
        vector=vector,
        route=route,
        scrubbed_messages=scrubbed,
        breakdown=timer.breakdown(),
    )
