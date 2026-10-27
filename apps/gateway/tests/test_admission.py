"""Admission Lua: token bucket exhaustion, refill, and payload echo."""

from __future__ import annotations

from tests.support import new_tenant

from app.db.redis import set_bytes
from app.engine.admission import Admission
from app.state import AppState


async def test_token_bucket_exhaustion_and_refill(gateway_state: AppState) -> None:
    admission = Admission(gateway_state.redis, capacity=2, refill_per_sec=1)
    tenant = new_tenant("rl")
    now = 50_000_000

    first = await admission.check(
        tenant_id=tenant, l1_key="l1:missing", affinity_key="", now_ms=now
    )
    second = await admission.check(
        tenant_id=tenant, l1_key="l1:missing", affinity_key="", now_ms=now
    )
    third = await admission.check(
        tenant_id=tenant, l1_key="l1:missing", affinity_key="", now_ms=now
    )
    assert first.allowed is True
    assert second.allowed is True
    assert third.allowed is False

    refilled = await admission.check(
        tenant_id=tenant, l1_key="l1:missing", affinity_key="", now_ms=now + 1000
    )
    assert refilled.allowed is True
    still_empty = await admission.check(
        tenant_id=tenant, l1_key="l1:missing", affinity_key="", now_ms=now + 1000
    )
    assert still_empty.allowed is False

    partial = await admission.check(
        tenant_id=tenant, l1_key="l1:missing", affinity_key="", now_ms=now + 1500
    )
    assert partial.allowed is False
    restored = await admission.check(
        tenant_id=tenant, l1_key="l1:missing", affinity_key="", now_ms=now + 2000
    )
    assert restored.allowed is True

    capped = Admission(gateway_state.redis, capacity=2, refill_per_sec=1)
    other = new_tenant("cap")
    await capped.check(
        tenant_id=other, l1_key="l1:missing", affinity_key="", now_ms=now
    )
    await capped.check(
        tenant_id=other, l1_key="l1:missing", affinity_key="", now_ms=now
    )
    jumped = await capped.check(
        tenant_id=other, l1_key="l1:missing", affinity_key="", now_ms=now + 10_000
    )
    assert jumped.allowed is True
    # A full bucket was restored, one token was spent, one remains.
    follow = await capped.check(
        tenant_id=other, l1_key="l1:missing", affinity_key="", now_ms=now + 10_000
    )
    assert follow.allowed is True
    denied = await capped.check(
        tenant_id=other, l1_key="l1:missing", affinity_key="", now_ms=now + 10_000
    )
    assert denied.allowed is False


async def test_admission_returns_l1_and_affinity_payloads(
    gateway_state: AppState,
) -> None:
    admission = Admission(gateway_state.redis, capacity=10, refill_per_sec=1)
    tenant = new_tenant("pay")
    l1_key = "l1:" + ("ab" * 32)
    affinity_key = f"aff:{tenant}:conv"
    payload = b'{"content":"hola"}'
    affinity = b'{"tier":"SIMPLE","provider":"xai","model":"grok-3-mini"}'
    await set_bytes(gateway_state.redis, l1_key, payload)
    await set_bytes(gateway_state.redis, affinity_key, affinity)
    now = 60_000_000

    hit = await admission.check(
        tenant_id=tenant, l1_key=l1_key, affinity_key=affinity_key, now_ms=now
    )
    assert hit.allowed is True
    assert hit.l1_payload == payload
    assert hit.affinity_payload == affinity

    denied_bucket = Admission(gateway_state.redis, capacity=1, refill_per_sec=1)
    await denied_bucket.check(
        tenant_id=tenant, l1_key=l1_key, affinity_key=affinity_key, now_ms=now
    )
    rejected = await denied_bucket.check(
        tenant_id=tenant, l1_key=l1_key, affinity_key=affinity_key, now_ms=now
    )
    assert rejected.allowed is False
    assert rejected.l1_payload == payload
    assert rejected.affinity_payload == affinity

    blank = await admission.check(
        tenant_id=new_tenant("blank"),
        l1_key=l1_key,
        affinity_key="",
        now_ms=now,
    )
    assert blank.l1_payload == payload
    assert blank.affinity_payload is None
