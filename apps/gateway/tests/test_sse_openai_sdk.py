"""OpenAI SDK compatibility over ASGI for MISS, HIT_L1, and HIT_L2."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportCallIssue=false

from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

from httpx import AsyncClient
from openai import AsyncOpenAI
from tests.support import sample_completion

from app.engine.pipeline import cache_keys_for
from app.schemas.finops import CacheStatus, XAegisFinOps
from app.state import AppState


def _sdk(client: AsyncClient) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key="sk-aegis-sandbox",
        base_url="http://test/v1",
        http_client=cast(Any, client),
        max_retries=0,
    )


def _finops(payload: object) -> XAegisFinOps:
    return XAegisFinOps.model_validate(payload)


def _extra(chunk: object) -> dict[str, object]:
    extra = getattr(chunk, "model_extra", None)
    if isinstance(extra, dict):
        return {str(key): value for key, value in extra.items()}
    return {}


async def test_stream_miss_then_hit_l1(
    client: AsyncClient, gateway_state: AppState
) -> None:
    oai = _sdk(client)
    prompt = f"Translate hello to Spanish {uuid4().hex}"
    messages = [
        {"role": "system", "content": f"sdk-l1-{uuid4().hex}"},
        {"role": "user", "content": prompt},
    ]
    first_text, first_finops = await _read_stream(oai, messages)
    assert first_text
    assert first_finops.cache_status is CacheStatus.MISS
    await gateway_state.dispatcher.wait_idle()
    second_text, second_finops = await _read_stream(oai, messages)
    assert second_text == first_text
    assert second_finops.cache_status is CacheStatus.HIT_L1


async def test_nonstream_miss_then_hit_l1(
    client: AsyncClient, gateway_state: AppState
) -> None:
    oai = _sdk(client)
    prompt = f"Translate goodbye to Spanish {uuid4().hex}"
    messages = [
        {"role": "system", "content": f"sdk-json-{uuid4().hex}"},
        {"role": "user", "content": prompt},
    ]
    first = await oai.chat.completions.create(
        model="auto",
        messages=cast(Any, messages),
        temperature=0.2,
        stream=False,
    )
    first_finops = _finops(_extra(first)["x_aegis_finops"])
    assert first.choices[0].message.content
    assert first_finops.cache_status is CacheStatus.MISS
    await gateway_state.dispatcher.wait_idle()
    second = await oai.chat.completions.create(
        model="auto",
        messages=cast(Any, messages),
        temperature=0.2,
        stream=False,
    )
    second_finops = _finops(_extra(second)["x_aegis_finops"])
    assert second.choices[0].message.content == first.choices[0].message.content
    assert second_finops.cache_status is CacheStatus.HIT_L1


async def test_stream_hit_l2(client: AsyncClient, gateway_state: AppState) -> None:
    oai = _sdk(client)
    system = f"l2-sys-{uuid4().hex}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "Translate hello to Spanish"},
    ]
    from app.schemas.openai import ChatCompletionRequest, ChatMessage

    request = ChatCompletionRequest(
        model="auto",
        temperature=0.2,
        messages=[
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content="Translate hello to Spanish"),
        ],
    )
    keys = cache_keys_for(
        request,
        "sandbox",
        request.messages,
        embed_max_chars=gateway_state.settings.embed_max_chars,
    )
    vector = await gateway_state.embedder.embed(keys.embed_text)
    await gateway_state.cache.put(
        l1_key="l1:" + ("cd" * 32),
        completion=sample_completion("cached-l2-hola"),
        tenant_id="sandbox",
        sys_hash=keys.sys_hash,
        vector=vector,
    )
    text, finops = await _read_stream(oai, messages)
    assert text == "cached-l2-hola"
    assert finops.cache_status is CacheStatus.HIT_L2
    assert finops.similarity_score is not None
    assert finops.similarity_score >= 0.94


async def _read_stream(
    oai: AsyncOpenAI, messages: list[dict[str, str]]
) -> tuple[str, XAegisFinOps]:
    stream = await oai.chat.completions.create(
        model="auto",
        messages=cast(Any, messages),
        temperature=0.2,
        stream=True,
    )
    parts: list[str] = []
    finops_raw: object | None = None
    async for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta.content
            if isinstance(delta, str):
                parts.append(delta)
        extra = _extra(chunk)
        if extra.get("x_aegis_finops") is not None:
            finops_raw = extra["x_aegis_finops"]
    assert finops_raw is not None
    return "".join(parts), _finops(finops_raw)
