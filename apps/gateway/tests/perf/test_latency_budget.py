"""Latency budgets for the L1 hit, L2 hit, and cache-miss hot paths."""

from __future__ import annotations

import gc
import math
import time
from collections.abc import Awaitable, Callable

import numpy as np
import pytest
from tests.support import new_tenant, sample_completion

from app.engine.cache import SIMILARITY_THRESHOLD
from app.engine.pipeline import (
    CacheHit,
    CacheMiss,
    HotPathResult,
    cache_keys_for,
    run_hot_path,
)
from app.schemas.finops import CacheStatus, LatencyBreakdown
from app.schemas.openai import ChatCompletionRequest, ChatMessage
from app.state import AppState

pytestmark = pytest.mark.perf

_WARMUP = 20
_ITERATIONS = 500


async def test_latency_budgets(gateway_state: AppState) -> None:
    left = await gateway_state.embedder.embed(
        "How do I reset my production AWS IAM root MFA?"
    )
    right = await gateway_state.embedder.embed(
        "Steps to recover lost MFA device on root AWS account"
    )
    similarity = float(np.dot(left, right))
    print(
        f"aws_mfa_paraphrase_similarity={similarity:.4f} "
        f"threshold={SIMILARITY_THRESHOLD:.2f}"
    )

    l1_totals, l1_rows = await _measure_l1(gateway_state)
    l2_totals, l2_rows = await _measure_l2(gateway_state)
    miss_totals, miss_rows = await _measure_miss(gateway_state)

    _print_stages("L1", l1_rows)
    _print_stages("L2", l2_rows)
    _print_stages("MISS", miss_rows)
    _assert_budget("L1", l1_totals, 2.0)
    _assert_budget("L2", l2_totals, 15.0)
    _assert_budget("MISS", miss_totals, 12.0)


async def _measure_l1(
    state: AppState,
) -> tuple[list[float], list[LatencyBreakdown]]:
    tenant = new_tenant("perfL1")
    request = ChatCompletionRequest(
        model="auto",
        temperature=0.0,
        messages=[ChatMessage(role="user", content="Translate hello to Spanish")],
    )
    keys = cache_keys_for(
        request,
        tenant,
        request.messages,
        embed_max_chars=state.settings.embed_max_chars,
    )
    vector = await state.embedder.embed(keys.embed_text)
    await state.cache.put(
        l1_key=keys.l1_key,
        completion=sample_completion(),
        tenant_id=tenant,
        sys_hash=keys.sys_hash,
        vector=vector,
    )

    async def call() -> HotPathResult:
        return await run_hot_path(request, tenant, state)

    return await _measure(call, CacheStatus.HIT_L1)


async def _measure_l2(
    state: AppState,
) -> tuple[list[float], list[LatencyBreakdown]]:
    tenant = new_tenant("perfL2")
    request = ChatCompletionRequest(
        model="auto",
        temperature=0.0,
        messages=[
            ChatMessage(role="system", content="perf-l2"),
            ChatMessage(role="user", content="Translate hello to Spanish"),
        ],
    )
    keys = cache_keys_for(
        request,
        tenant,
        request.messages,
        embed_max_chars=state.settings.embed_max_chars,
    )
    vector = await state.embedder.embed(keys.embed_text)
    await state.cache.put(
        l1_key="l1:" + ("ab" * 32),
        completion=sample_completion(),
        tenant_id=tenant,
        sys_hash=keys.sys_hash,
        vector=vector,
    )

    async def call() -> HotPathResult:
        return await run_hot_path(request, tenant, state)

    return await _measure(call, CacheStatus.HIT_L2)


async def _measure_miss(
    state: AppState,
) -> tuple[list[float], list[LatencyBreakdown]]:
    tenant = new_tenant("perfMiss")
    request = ChatCompletionRequest(
        model="auto",
        temperature=0.0,
        messages=[
            ChatMessage(role="system", content=f"perf-miss-{tenant}"),
            ChatMessage(role="user", content="Translate hello to Spanish"),
        ],
    )

    async def call() -> HotPathResult:
        return await run_hot_path(request, tenant, state)

    return await _measure(call, CacheStatus.MISS)


async def _measure(
    call: Callable[[], Awaitable[HotPathResult]],
    expect: CacheStatus,
) -> tuple[list[float], list[LatencyBreakdown]]:
    for _ in range(_WARMUP):
        _expect(await call(), expect)
    gc.collect()
    totals: list[float] = []
    rows: list[LatencyBreakdown] = []
    gc.disable()
    try:
        for _ in range(_ITERATIONS):
            started = time.perf_counter_ns()
            result = await call()
            totals.append((time.perf_counter_ns() - started) / 1_000_000)
            rows.append(result.breakdown)
            _expect(result, expect)
    finally:
        gc.enable()
    return totals, rows


def _expect(result: HotPathResult, expect: CacheStatus) -> None:
    if expect is CacheStatus.MISS:
        if not isinstance(result, CacheMiss):
            raise AssertionError(f"expected MISS, got {type(result).__name__}")
        return
    if not isinstance(result, CacheHit) or result.status is not expect:
        got = result.status.value if isinstance(result, CacheHit) else "MISS"
        raise AssertionError(f"expected {expect.value}, got {got}")


def _assert_budget(name: str, samples: list[float], limit_ms: float) -> None:
    p50 = _percentile(samples, 50)
    p95 = _percentile(samples, 95)
    print(f"{name} total p50={p50:.3f}ms p95={p95:.3f}ms n={len(samples)}")
    assert p95 < limit_ms, f"{name} p95 {p95:.3f}ms exceeds {limit_ms:.1f}ms"


def _print_stages(label: str, rows: list[LatencyBreakdown]) -> None:
    for stage in ("guardrail", "l1_lookup", "onnx_embed", "l2_hnsw", "router"):
        values = [
            value for value in (_stage(row, stage) for row in rows) if value is not None
        ]
        if not values:
            print(f"{label} {stage} skipped")
            continue
        print(
            f"{label} {stage} p50={_percentile(values, 50):.3f}ms "
            f"p95={_percentile(values, 95):.3f}ms"
        )


def _stage(row: LatencyBreakdown, name: str) -> float | None:
    if name == "guardrail":
        return row.guardrail
    if name == "l1_lookup":
        return row.l1_lookup
    if name == "onnx_embed":
        return row.onnx_embed
    if name == "l2_hnsw":
        return row.l2_hnsw
    if name == "router":
        return row.router
    return None


def _percentile(samples: list[float], pct: float) -> float:
    ordered = sorted(samples)
    index = math.ceil(pct / 100.0 * len(ordered)) - 1
    if index < 0:
        return ordered[0]
    if index >= len(ordered):
        return ordered[-1]
    return ordered[index]
