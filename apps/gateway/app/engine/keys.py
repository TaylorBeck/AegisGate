"""Canonical L1 keys, system-context hashes, and L2 eligibility.

`l1_key` is `l1:` plus the blake3 hex of sorted JSON `{t, p, m}`.
`sys_hash` is the first 16 hex chars of blake3(system contents + model policy).
`conversation_id` is set when there are at least two messages.
L2 is eligible when non-system messages are at most 3 and temperature is at most 0.3.
A missing temperature defaults to 1.0. Embed text keeps the tail within
`embed_max_chars`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import blake3
import orjson

from app.engine.coerce import object_dict, object_list
from app.schemas.openai import ChatMessage


@dataclass(frozen=True, slots=True)
class CacheKeys:
    """Derived cache identity for one scrubbed request."""

    l1_key: str
    sys_hash: str
    conversation_id: str | None
    l2_eligible: bool
    embed_text: str


def derive_cache_keys(
    *,
    tenant_id: str,
    model: str,
    messages: Sequence[ChatMessage | Mapping[str, object]],
    temperature: float | None,
    max_tokens: int | None,
    tools: object = None,
    response_format: object = None,
    embed_max_chars: int,
) -> CacheKeys:
    """Build the L1 key, system hash, conversation id, and embed text."""
    canonical = [_canonical_message(message) for message in messages]
    policy_bytes = _dumps(
        {
            "max_tokens": max_tokens,
            "model": model,
            "response_format": response_format,
            "temperature_bucket": _temperature_bucket(temperature),
            "tools": tools,
        }
    )
    system_contents = [
        _content_text(message.get("content"))
        for message in canonical
        if message.get("role") == "system"
    ]
    sys_hash = blake3.blake3(_dumps(system_contents) + policy_bytes).hexdigest()[:16]
    l1_key = (
        "l1:"
        + blake3.blake3(
            _dumps({"m": canonical, "p": policy_bytes.decode("utf-8"), "t": tenant_id})
        ).hexdigest()
    )

    conversation_id: str | None = None
    if len(canonical) >= 2:
        conversation_id = blake3.blake3(
            _dumps([tenant_id, canonical[0], canonical[1]])
        ).hexdigest()[:16]

    non_system = 0
    lines: list[str] = []
    for message in canonical:
        role = message.get("role")
        role_text = role if isinstance(role, str) else ""
        if role_text == "system":
            continue
        non_system += 1
        lines.append(f"{role_text}: {_content_text(message.get('content'))}")
    embed_text = "\n".join(lines)
    if len(embed_text) > embed_max_chars:
        embed_text = embed_text[-embed_max_chars:]

    effective_temperature = 1.0 if temperature is None else float(temperature)
    return CacheKeys(
        l1_key=l1_key,
        sys_hash=sys_hash,
        conversation_id=conversation_id,
        l2_eligible=non_system <= 3 and effective_temperature <= 0.3,
        embed_text=embed_text,
    )


def _temperature_bucket(temperature: float | None) -> str:
    """Hundredths bucket so float noise does not split an exact-match key."""
    value = 1.0 if temperature is None else float(temperature)
    hundredths = round(value * 100.0)
    return f"{hundredths / 100:.2f}"


def _canonical_message(
    message: ChatMessage | Mapping[str, object],
) -> dict[str, object]:
    if isinstance(message, ChatMessage):
        dumped = message.model_dump(mode="json", exclude_none=True)
        return {str(key): value for key, value in dumped.items()}
    return {str(key): value for key, value in message.items() if value is not None}


def _content_text(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    values = object_list(content)
    if values is None:
        return ""
    chunks: list[str] = []
    for part in values:
        if isinstance(part, str):
            chunks.append(part)
            continue
        mapped = object_dict(part)
        if mapped is None:
            continue
        text = mapped.get("text")
        if isinstance(text, str):
            chunks.append(text)
    return "\n".join(chunks)


def _dumps(value: object) -> bytes:
    return orjson.dumps(value, option=orjson.OPT_SORT_KEYS)
