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
