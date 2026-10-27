"""Hot-path orchestration: miss, ineligible L2, scrubbing, affinity, rate limit."""

from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

import numpy as np
import pytest
from numpy.typing import NDArray
from tests.support import new_tenant

from app.db.redis import set_bytes
from app.engine.admission import Admission, AdmissionDeniedError
from app.engine.pipeline import CacheMiss, cache_keys_for, run_hot_path
from app.engine.router import AffinityRecord, encode_affinity
from app.schemas.finops import Tier
from app.schemas.openai import ChatCompletionRequest, ChatMessage
from app.state import AppState


async def test_eligible_miss_records_embed_search_and_route(
    gateway_state: AppState,
) -> None:
    request = ChatCompletionRequest(
        model="auto",
        temperature=0.2,
        messages=[
            ChatMessage(role="system", content="You are a translator"),
            ChatMessage(role="user", content=f"Translate hello {uuid4().hex}"),
        ],
    )
    result = await run_hot_path(request, new_tenant("miss"), gateway_state)
    assert isinstance(result, CacheMiss)
    assert result.keys.l2_eligible is True
    assert result.vector.shape == (384,)
    assert result.vector.dtype == np.float32
    assert abs(float(np.linalg.norm(result.vector)) - 1.0) < 1e-5
    assert result.breakdown.onnx_embed is not None
    assert result.breakdown.l2_hnsw is not None
    assert result.breakdown.router is not None
    assert len(result.route.scores) == 3
    assert result.route.provider_chain
    assert result.route.model.provider == result.route.provider_chain[0]
    assert result.route.sticky is False


async def test_ineligible_request_skips_l2(gateway_state: AppState) -> None:
    request = ChatCompletionRequest(
        model="auto",
        messages=[ChatMessage(role="user", content=f"Explain this {uuid4().hex}")],
    )
    result = await run_hot_path(request, new_tenant("skip"), gateway_state)
    assert isinstance(result, CacheMiss)
    assert result.keys.l2_eligible is False
    assert result.breakdown.l2_hnsw is None
    assert result.breakdown.onnx_embed is not None
    assert result.breakdown.router is not None


async def test_scrubbed_text_is_what_gets_hashed(gateway_state: AppState) -> None:
    secret = "sk-" + ("a" * 32)
    tenant = new_tenant("scrub")
    request = ChatCompletionRequest(
        model="auto",
        temperature=0.0,
        messages=[ChatMessage(role="user", content=f"my key is {secret}")],
    )
    result = await run_hot_path(request, tenant, gateway_state)
    assert isinstance(result, CacheMiss)
    content = result.scrubbed_messages[0].content
    assert isinstance(content, str)
    assert secret not in content
    assert "[REDACTED_OPENAI]" in content
    limit = gateway_state.settings.embed_max_chars
    raw_keys = cache_keys_for(request, tenant, request.messages, embed_max_chars=limit)
    scrubbed_keys = cache_keys_for(
        request, tenant, result.scrubbed_messages, embed_max_chars=limit
    )
    assert result.keys.l1_key != raw_keys.l1_key
    assert result.keys.l1_key == scrubbed_keys.l1_key


async def test_affinity_record_reaches_the_router(
    gateway_state: AppState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tenant = new_tenant("aff")
    request = ChatCompletionRequest(
        model="auto",
        temperature=1.0,
        messages=[
            ChatMessage(role="user", content="hello"),
            ChatMessage(role="assistant", content="hi"),
            ChatMessage(role="user", content="say more"),
        ],
    )
    keys = cache_keys_for(
        request,
        tenant,
        request.messages,
        embed_max_chars=gateway_state.settings.embed_max_chars,
    )
    assert keys.conversation_id is not None
    record = AffinityRecord(
        tier=Tier.REASONING,
        provider="anthropic",
        model="claude-sonnet-4.5",
    )
    await set_bytes(
        gateway_state.redis,
        f"aff:{tenant}:{keys.conversation_id}",
        encode_affinity(record),
    )

    def classify_simple(
        _vector: NDArray[np.float32],
    ) -> tuple[Tier, dict[Tier, float]]:
        return (
            Tier.SIMPLE,
            {Tier.SIMPLE: 1.0, Tier.REASONING: 0.0, Tier.AGENTIC_CODE: 0.0},
        )

    monkeypatch.setattr(gateway_state.router, "classify", classify_simple)
    result = await run_hot_path(request, tenant, gateway_state)
    assert isinstance(result, CacheMiss)
    assert result.route.sticky is True
    assert result.route.tier is Tier.REASONING
    assert result.route.model.name == "claude-sonnet-4.5"


async def test_exhausted_bucket_raises(gateway_state: AppState) -> None:
    tenant = new_tenant("deny")
    limited = replace(
        gateway_state,
        admission=Admission(gateway_state.redis, capacity=1, refill_per_sec=1),
    )
    request = ChatCompletionRequest(
        model="auto",
        temperature=1.0,
        messages=[ChatMessage(role="user", content="hello")],
    )
    now = 9_000_000
    first = await run_hot_path(request, tenant, limited, now_ms=now)
    assert isinstance(first, CacheMiss)
    with pytest.raises(AdmissionDeniedError) as caught:
        await run_hot_path(request, tenant, limited, now_ms=now)
    assert caught.value.tenant_id == tenant
