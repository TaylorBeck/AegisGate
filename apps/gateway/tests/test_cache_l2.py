"""L2 HNSW isolation, similarity threshold, and TTL."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

import numpy as np
import pytest
from tests.support import new_tenant, sample_completion

from app.engine.cache import (
    SIMILARITY_THRESHOLD,
    escape_tag,
    require_tenant,
    semantic_similarity,
)
from app.engine.pipeline import CacheHit, CacheMiss, cache_keys_for, run_hot_path
from app.schemas.finops import CacheStatus
from app.schemas.openai import ChatCompletionRequest, ChatMessage
from app.state import AppState


def test_similarity_threshold_boundary() -> None:
    exact = semantic_similarity(1.0 - 0.94)
    assert exact is not None
    assert exact >= SIMILARITY_THRESHOLD
    assert semantic_similarity(1.0 - 0.939999) is None


def test_tenant_tag_grammar() -> None:
    assert escape_tag("tenant-a") == "tenant\\-a"
    assert escape_tag("abc_1") == "abc_1"
    assert require_tenant("tenant-a_1") == "tenant-a_1"
    with pytest.raises(ValueError):
        require_tenant("bad tenant")
    with pytest.raises(ValueError):
        require_tenant("")


async def test_tenant_and_sys_hash_isolation(gateway_state: AppState) -> None:
    tenant_a = new_tenant("a-")
    tenant_b = new_tenant("b-")
    vector = await gateway_state.embedder.embed("identical request text")
    completion = sample_completion("only tenant a")
    sys_a = "0123456789abcdef"
    sys_b = "fedcba9876543210"
    await gateway_state.cache.put(
        l1_key="l1:" + ("aa" * 32),
        completion=completion,
        tenant_id=tenant_a,
        sys_hash=sys_a,
        vector=vector,
    )

    own = await gateway_state.cache.get_l2(
        tenant_id=tenant_a, sys_hash=sys_a, vector=vector
    )
    other_tenant = await gateway_state.cache.get_l2(
        tenant_id=tenant_b, sys_hash=sys_a, vector=vector
    )
    other_hash = await gateway_state.cache.get_l2(
        tenant_id=tenant_a, sys_hash=sys_b, vector=vector
    )
    assert own is not None
    assert own[0].content == "only tenant a"
    assert own[1] >= 0.99
    assert other_tenant is None
    assert other_hash is None


async def test_far_vector_misses_threshold(gateway_state: AppState) -> None:
    tenant = new_tenant("far")
    vector = await gateway_state.embedder.embed("a stored sentence")
    await gateway_state.cache.put(
        l1_key="l1:" + ("bb" * 32),
        completion=sample_completion("stored"),
        tenant_id=tenant,
        sys_hash="abc123abc123abc1",
        vector=vector,
    )
    opposite = np.ascontiguousarray(-vector, dtype=np.float32)
    missed = await gateway_state.cache.get_l2(
        tenant_id=tenant,
        sys_hash="abc123abc123abc1",
        vector=opposite,
    )
    assert missed is None


async def test_l2_hit_skips_routing(
    gateway_state: AppState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tenant = new_tenant("l2")
    request = ChatCompletionRequest(
        model="auto",
        temperature=0.2,
        messages=[
            ChatMessage(role="system", content="You are a translator"),
            ChatMessage(role="user", content="Translate hello to Spanish"),
        ],
    )
    keys = cache_keys_for(
        request,
        tenant,
        gateway_state.guardrail.scrub_messages(request.messages),
        embed_max_chars=gateway_state.settings.embed_max_chars,
    )
    vector = await gateway_state.embedder.embed(keys.embed_text)
    await gateway_state.cache.put(
        l1_key="l1:" + ("ee" * 32),
        completion=sample_completion("hola"),
        tenant_id=tenant,
        sys_hash=keys.sys_hash,
        vector=vector,
    )

    def fail_resolve(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("L2 hit must not route")

    monkeypatch.setattr(gateway_state.router, "resolve", fail_resolve)
    result = await run_hot_path(request, tenant, gateway_state)
    assert isinstance(result, CacheHit)
    assert result.status is CacheStatus.HIT_L2
    assert result.similarity is not None
    assert result.similarity >= SIMILARITY_THRESHOLD
    assert result.completion.content == "hola"
    assert result.breakdown.onnx_embed is not None
    assert result.breakdown.l2_hnsw is not None
    assert result.breakdown.router is None


async def test_l2_entry_expires(gateway_state: AppState) -> None:
    tenant = new_tenant("l2ttl")
    vector = await gateway_state.embedder.embed("ttl sentence")
    sys_hash = "0123456789abcdef"
    await gateway_state.cache.put(
        l1_key="l1:" + ("ff" * 32),
        completion=sample_completion("gone"),
        tenant_id=tenant,
        sys_hash=sys_hash,
        vector=vector,
        ttl_seconds=1,
    )
    assert await gateway_state.cache.get_l2(
        tenant_id=tenant, sys_hash=sys_hash, vector=vector
    )

    async def fetch() -> object | None:
        return await gateway_state.cache.get_l2(
            tenant_id=tenant, sys_hash=sys_hash, vector=vector
        )

    await _wait_for_none(fetch)
    missed = await run_hot_path(
        ChatCompletionRequest(
            model="auto",
            temperature=1.0,
            messages=[ChatMessage(role="user", content="unrelated miss after ttl")],
        ),
        tenant,
        gateway_state,
    )
    assert isinstance(missed, CacheMiss)


async def _wait_for_none(fetch: Callable[[], Awaitable[object | None]]) -> None:
    deadline = time.perf_counter() + 5
    while time.perf_counter() < deadline:
        if await fetch() is None:
            return
        await asyncio.sleep(0.1)
    raise AssertionError("cached value did not expire")
