"""Cache key stability, tenant isolation, and L2 eligibility edges."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from app.engine.keys import CacheKeys, derive_cache_keys
from app.schemas.openai import ChatMessage


def derive(
    messages: Sequence[ChatMessage | Mapping[str, object]],
    *,
    tenant_id: str = "tenant-a",
    temperature: float | None = 0.0,
    tools: object = None,
    embed_max_chars: int = 1024,
    model: str = "auto",
    max_tokens: int | None = None,
    response_format: object = None,
) -> CacheKeys:
    return derive_cache_keys(
        tenant_id=tenant_id,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools,
        response_format=response_format,
        embed_max_chars=embed_max_chars,
    )


def test_keys_are_stable_under_dict_reordering() -> None:
    first = [{"role": "user", "content": "hi", "name": "a"}]
    second = [{"name": "a", "content": "hi", "role": "user"}]
    tools_a = [
        {"type": "function", "function": {"name": "lookup", "description": "find"}}
    ]
    tools_b = [
        {"function": {"description": "find", "name": "lookup"}, "type": "function"}
    ]
    assert derive(first, tools=tools_a).l1_key == derive(second, tools=tools_b).l1_key


def test_keys_differ_across_tenants() -> None:
    messages = [{"role": "user", "content": "hello"}]
    alpha = derive(messages, tenant_id="alpha")
    beta = derive(messages, tenant_id="beta")
    assert alpha.l1_key != beta.l1_key
    assert alpha.l1_key.startswith("l1:")
    assert len(alpha.l1_key) == 3 + 64
    assert len(alpha.sys_hash) == 16


def test_system_message_changes_sys_hash() -> None:
    user = {"role": "user", "content": "hi"}
    first = derive([{"role": "system", "content": "one"}, user])
    second = derive([{"role": "system", "content": "two"}, user])
    assert first.sys_hash != second.sys_hash
    assert first.l1_key != second.l1_key


def test_eligibility_three_versus_four_non_system_messages() -> None:
    three = [{"role": "user", "content": f"m{index}"} for index in range(3)]
    four = [{"role": "user", "content": f"m{index}"} for index in range(4)]
    assert derive(three, temperature=0.0).l2_eligible is True
    assert derive(four, temperature=0.0).l2_eligible is False
    with_system = [{"role": "system", "content": "rules"}, *three]
    assert derive(with_system, temperature=0.0).l2_eligible is True


def test_eligibility_temperature_edges_and_default() -> None:
    messages = [{"role": "user", "content": "hello"}]
    low = derive(messages, temperature=0.3)
    high = derive(messages, temperature=0.31)
    omitted = derive(messages, temperature=None)
    explicit = derive(messages, temperature=1.0)
    assert low.l2_eligible is True
    assert high.l2_eligible is False
    assert low.l1_key != high.l1_key
    assert omitted.l2_eligible is False
    assert omitted.l1_key == explicit.l1_key
    assert omitted.sys_hash == explicit.sys_hash


def test_conversation_id_and_embed_text() -> None:
    single = derive([{"role": "user", "content": "hi"}])
    assert single.conversation_id is None
    assert single.embed_text == "user: hi"

    pair = [
        {"role": "system", "content": "secret context"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "yo"},
    ]
    keyed = derive(pair)
    again = derive(pair)
    other_tenant = derive(pair, tenant_id="other-tenant")
    assert keyed.conversation_id is not None
    assert len(keyed.conversation_id) == 16
    assert keyed.conversation_id == again.conversation_id
    assert keyed.conversation_id != other_tenant.conversation_id
    assert keyed.embed_text == "user: hello\nassistant: yo"
    assert "secret" not in keyed.embed_text

    long = "x" * 2000
    truncated = derive([{"role": "user", "content": long}], embed_max_chars=64)
    full = f"user: {long}"
    assert len(truncated.embed_text) == 64
    assert truncated.embed_text == full[-64:]
    assert not truncated.embed_text.startswith("user:")
