"""Token bucket plus L1 and affinity reads in one EVALSHA round trip."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from redis.asyncio import Redis

from app.db.redis import eval_script, register_script
from app.engine.coerce import object_list

_LUA = Path(__file__).with_name("admission.lua").read_text(encoding="utf-8")


class AdmissionDeniedError(Exception):
    """The tenant token bucket rejected the request."""

    def __init__(self, tenant_id: str) -> None:
        super().__init__(f"rate limit exceeded for tenant {tenant_id}")
        self.tenant_id = tenant_id


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    """`(allowed, l1_payload, affinity_payload)` from the admission script."""

    allowed: bool
    l1_payload: bytes | None
    affinity_payload: bytes | None


class Admission:
    """Lua token bucket on `rl:{tenant}`, plus the L1 and affinity GETs."""

    def __init__(self, redis: Redis, *, capacity: float, refill_per_sec: float) -> None:
        if capacity <= 0 or refill_per_sec <= 0:
            raise ValueError("rate limit capacity and refill must be positive")
        self._redis = redis
        self._capacity = capacity
        self._refill_per_sec = refill_per_sec
        self._script = register_script(redis, _LUA)

    async def check(
        self,
        *,
        tenant_id: str,
        l1_key: str,
        affinity_key: str,
        now_ms: int | None = None,
    ) -> AdmissionResult:
        """Consume one token and return the L1 and affinity payloads."""
        if now_ms is None:
            now_ms = time.time_ns() // 1_000_000
        raw = await eval_script(
            self._script,
            keys=[f"rl:{tenant_id}", l1_key, affinity_key],
            args=[self._capacity, self._refill_per_sec, now_ms, 1],
        )
        return _parse_result(raw)


def _parse_result(raw: object) -> AdmissionResult:
    values = object_list(raw)
    if values is None or len(values) != 3:
        raise RuntimeError("unexpected admission reply")
    allowed_raw, l1_raw, affinity_raw = values
    return AdmissionResult(
        allowed=_as_int(allowed_raw) == 1,
        l1_payload=_as_optional_bytes(l1_raw),
        affinity_payload=_as_optional_bytes(affinity_raw),
    )


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        raise RuntimeError("unexpected admission flag")
    if isinstance(value, int):
        return value
    if isinstance(value, bytes):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise RuntimeError("unexpected admission flag")


def _as_optional_bytes(value: object) -> bytes | None:
    if value is None or value is False:
        return None
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode()
    return None
