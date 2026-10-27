"""L1 exact-match hits, round-trip, and TTL."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

import pytest
from tests.support import new_tenant, sample_completion

from app.engine.cache import CachedCompletion, decode_completion, encode_completion
from app.engine.pipeline import CacheHit, cache_keys_for, run_hot_path
from app.schemas.finops import CacheStatus, Tier
from app.schemas.openai import ChatCompletionRequest, ChatMessage
from app.state import AppState


def test_completion_roundtrip() -> None:
    completion = sample_completion()
    assert decode_completion(encode_completion(completion)) == completion


async def test_l1_hit_uses_cached_tier_and_skips_embedding(
    gateway_state: AppState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tenant = new_tenant("l1")
    request = _request("Translate hello to Spanish")
    keys = cache_keys_for(
        request,
        tenant,
        gateway_state.guardrail.scrub_messages(request.messages),
        embed_max_chars=gateway_state.settings.embed_max_chars,
    )
    vector = await gateway_state.embedder.embed(keys.embed_text)
    completion = sample_completion(tier=Tier.AGENTIC_CODE)
    await gateway_state.cache.put(
        l1_key=keys.l1_key,
        completion=completion,
        tenant_id=tenant,
        sys_hash=keys.sys_hash,
        vector=vector,
    )

    async def fail_embed(text: str) -> object:
        raise AssertionError(f"L1 hit must not embed: {text}")

    monkeypatch.setattr(gateway_state.embedder, "embed", fail_embed)
    result = await run_hot_path(request, tenant, gateway_state)
    assert isinstance(result, CacheHit)
    assert result.status is CacheStatus.HIT_L1
    assert result.similarity is None
    assert result.completion == completion
    assert result.completion.tier is Tier.AGENTIC_CODE
    assert result.breakdown.onnx_embed is None
    assert result.breakdown.l2_hnsw is None
    assert result.breakdown.router is None
    assert result.breakdown.l1_lookup >= 0.0


async def test_l1_entry_expires(gateway_state: AppState) -> None:
    tenant = new_tenant("l1ttl")
    request = _request("expire this exact prompt")
    keys = cache_keys_for(
        request,
        tenant,
        request.messages,
        embed_max_chars=gateway_state.settings.embed_max_chars,
    )
    vector = await gateway_state.embedder.embed(keys.embed_text)
    await gateway_state.cache.put(
        l1_key=keys.l1_key,
        completion=sample_completion(),
        tenant_id=tenant,
        sys_hash=keys.sys_hash,
        vector=vector,
        ttl_seconds=1,
    )
    assert await gateway_state.cache.get_l1(keys.l1_key) is not None

    async def fetch() -> CachedCompletion | None:
        return await gateway_state.cache.get_l1(keys.l1_key)

    await _wait_for_none(fetch)


def _request(content: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="auto",
        temperature=0.0,
        messages=[ChatMessage(role="user", content=content)],
    )


async def _wait_for_none(fetch: Callable[[], Awaitable[object | None]]) -> None:
    deadline = time.perf_counter() + 5
    while time.perf_counter() < deadline:
        if await fetch() is None:
            return
        await asyncio.sleep(0.1)
    raise AssertionError("cached value did not expire")
