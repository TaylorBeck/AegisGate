"""Redis / Valkey connection pools.

Two pools share the same URL: a hot-path pool for L1/L2/Lua, and a
blocking pool reserved for `XREAD` so SSE subscribers cannot starve
admission. Clients speak RESP2 with `decode_responses=False`.

Redis-py stubs use `**kwargs: Unknown`; unknown-member noise stays in this
module so the rest of the gateway can stay strict.
"""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from redis.asyncio import Redis
from redis.commands.core import AsyncScript

from app.config import Settings


async def create_pools(settings: Settings) -> tuple[Redis, Redis]:
    """Return `(hot_path, blocking)` Redis clients and fail fast on ping."""
    hot = Redis.from_url(
        settings.redis_url,
        decode_responses=False,
        protocol=2,
        max_connections=settings.redis_hot_max_connections,
    )
    blocking = Redis.from_url(
        settings.redis_url,
        decode_responses=False,
        protocol=2,
        max_connections=settings.redis_blocking_max_connections,
    )
    await ping(hot)
    await ping(blocking)
    return hot, blocking


async def ping(client: Redis) -> bool:
    """Typed wrapper around `Redis.ping`."""
    return bool(await client.ping())


async def aclose(client: Redis) -> None:
    """Typed wrapper around `Redis.aclose`."""
    await client.aclose()


async def raw_command(client: Redis, *args: str | bytes | int | float) -> object:
    """Execute a raw Redis command and return the untyped reply."""
    reply: object = cast(object, await client.execute_command(*args))
    return reply


def register_script(client: Redis, source: str) -> AsyncScript:
    """Register a Lua script. Calls use EVALSHA and fall back to EVAL."""
    return client.register_script(source)


async def eval_script(
    script: AsyncScript,
    *,
    keys: Sequence[str],
    args: Sequence[str | int | float],
) -> object:
    """Run a registered script and return the untyped reply."""
    reply: object = cast(object, await script(keys=list(keys), args=list(args)))
    return reply


async def get_bytes(client: Redis, key: str) -> bytes | None:
    """GET a key and return raw bytes."""
    raw: object = cast(object, await client.get(key))
    if raw is None:
        return None
    if isinstance(raw, bytes):
        return raw
    if isinstance(raw, str):
        return raw.encode()
    raise TypeError("expected bytes from Redis GET")


async def set_bytes(client: Redis, key: str, value: bytes) -> None:
    """SET a key from raw bytes."""
    await client.set(key, value)


async def setex_bytes(client: Redis, key: str, value: bytes, ttl_seconds: int) -> None:
    """SET a key from raw bytes with a TTL."""
    await client.set(key, value, ex=ttl_seconds)


async def xadd_approx(
    client: Redis,
    stream: str,
    fields: dict[str, str | bytes],
    *,
    maxlen: int,
) -> bytes:
    """XADD with approximate MAXLEN. Returns the stream entry id."""
    raw: object = cast(
        object,
        await client.xadd(stream, cast(Any, fields), maxlen=maxlen, approximate=True),
    )
    if isinstance(raw, bytes):
        return raw
    if isinstance(raw, str):
        return raw.encode()
    raise TypeError("expected stream id from XADD")


async def xread_block(
    client: Redis,
    stream: str,
    last_id: str,
    *,
    block_ms: int,
    count: int,
) -> list[tuple[bytes, bytes]]:
    """Blocking XREAD. Returns `(id, payload)` pairs from the `payload` field."""
    raw: object = cast(
        object,
        await client.xread({stream: last_id}, block=block_ms, count=count),
    )
    return _parse_xread(raw)


def _parse_xread(raw: object) -> list[tuple[bytes, bytes]]:
    if raw is None:
        return []
    streams = raw if isinstance(raw, list) else None
    if not streams:
        return []
    out: list[tuple[bytes, bytes]] = []
    for stream in streams:
        if not isinstance(stream, list | tuple) or len(stream) < 2:
            continue
        entries = stream[1]
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, list | tuple) or len(entry) < 2:
                continue
            entry_id = _as_bytes(entry[0])
            payload = _payload_field(entry[1])
            if entry_id is None or payload is None:
                continue
            out.append((entry_id, payload))
    return out


def _payload_field(fields: object) -> bytes | None:
    if isinstance(fields, dict):
        items = list(cast(dict[object, object], fields).items())
    elif isinstance(fields, list | tuple):
        values = list(fields)
        if len(values) % 2 != 0:
            return None
        items = [(values[i], values[i + 1]) for i in range(0, len(values), 2)]
    else:
        return None
    for key, value in items:
        name = _as_text(key)
        if name == "payload":
            return _as_bytes(value)
    return None


def _as_bytes(value: object) -> bytes | None:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode()
    return None


def _as_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, str):
        return value
    return str(value)


async def write_expiring_pair(
    client: Redis,
    *,
    string_key: str,
    string_value: bytes,
    hash_key: str,
    mapping: dict[str, str | bytes],
    ttl_seconds: int,
) -> None:
    """SETEX a string and HSET+EXPIRE a hash in one MULTI/EXEC pipeline."""
    pipe = client.pipeline(transaction=True)
    pipe.set(string_key, string_value, ex=ttl_seconds)
    pipe.hset(hash_key, mapping=cast(Any, mapping))
    pipe.expire(hash_key, ttl_seconds)
    await pipe.execute()
