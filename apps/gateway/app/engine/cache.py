"""L1 exact cache and tenant-scoped L2 HNSW semantic cache."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import orjson
from numpy.typing import NDArray
from redis.asyncio import Redis
from redis.exceptions import ResponseError

from app.db.redis import get_bytes, raw_command, write_expiring_pair
from app.engine.coerce import object_dict, object_list, object_pairs
from app.schemas.finops import Tier

SIMILARITY_THRESHOLD = 0.94
_INDEX = "idx:l2"
_TENANT = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_TAG_SPECIAL = re.compile(r"([,.<>{}\[\]\"':;!@#$%^&*()\-+=~|/\\ ])")
_DIM = 384


@dataclass(frozen=True, slots=True)
class CachedCompletion:
    """Stored assistant completion replayed on an L1 or L2 hit."""

    content: str
    model: str
    provider: str
    tier: Tier
    tokens_in: int
    tokens_out: int
    finish_reason: str
    created_at: int


def require_tenant(tenant_id: str) -> str:
    """Reject tenant ids that could break out of a RediSearch tag filter."""
    if _TENANT.fullmatch(tenant_id) is None:
        raise ValueError("tenant_id must match ^[A-Za-z0-9_-]{1,64}$")
    return tenant_id


def escape_tag(value: str) -> str:
    """Escape a RediSearch tag value."""
    return _TAG_SPECIAL.sub(r"\\\1", value)


def semantic_similarity(distance: float) -> float | None:
    """Convert cosine distance to similarity. Hits require similarity >= 0.94."""
    similarity = 1.0 - distance
    if math.isnan(similarity) or similarity < SIMILARITY_THRESHOLD:
        return None
    return similarity


def encode_completion(completion: CachedCompletion) -> bytes:
    return orjson.dumps(
        {
            "content": completion.content,
            "created_at": completion.created_at,
            "finish_reason": completion.finish_reason,
            "model": completion.model,
            "provider": completion.provider,
            "tier": completion.tier.value,
            "tokens_in": completion.tokens_in,
            "tokens_out": completion.tokens_out,
        }
    )


def decode_completion(payload: bytes) -> CachedCompletion:
    raw = _object(orjson.loads(payload), "cached completion")
    return CachedCompletion(
        content=_text(raw, "content"),
        model=_text(raw, "model"),
        provider=_text(raw, "provider"),
        tier=Tier(_text(raw, "tier")),
        tokens_in=_integer(raw, "tokens_in"),
        tokens_out=_integer(raw, "tokens_out"),
        finish_reason=_text(raw, "finish_reason"),
        created_at=_integer(raw, "created_at"),
    )


def l2_document_key(l1_key: str) -> str:
    return "l2:" + l1_key.removeprefix("l1:")


class SemanticCache:
    """L1 SETEX values and L2 HNSW hashes scoped by tenant and system hash."""

    def __init__(self, redis: Redis, *, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._redis = redis
        self._ttl_seconds = ttl_seconds

    async def ensure_index(self) -> None:
        """Create the HNSW index. An existing index is success."""
        try:
            await raw_command(
                self._redis,
                "FT.CREATE",
                _INDEX,
                "ON",
                "HASH",
                "PREFIX",
                "1",
                "l2:",
                "SCHEMA",
                "tenant_id",
                "TAG",
                "sys_hash",
                "TAG",
                "embedding",
                "VECTOR",
                "HNSW",
                "10",
                "TYPE",
                "FLOAT32",
                "DIM",
                "384",
                "DISTANCE_METRIC",
                "COSINE",
                "M",
                "16",
                "EF_CONSTRUCTION",
                "200",
            )
        except ResponseError as exc:
            if "already exists" not in str(exc).lower():
                raise

    async def get_l1(self, l1_key: str) -> CachedCompletion | None:
        payload = await get_bytes(self._redis, l1_key)
        if payload is None:
            return None
        return decode_completion(payload)

    async def get_l2(
        self,
        *,
        tenant_id: str,
        sys_hash: str,
        vector: NDArray[np.float32],
    ) -> tuple[CachedCompletion, float] | None:
        """KNN-1 inside the tenant and system-hash pre-filter."""
        require_tenant(tenant_id)
        if sys_hash == "":
            raise ValueError("sys_hash is required")
        query = (
            f"(@tenant_id:{{{escape_tag(tenant_id)}}} "
            f"@sys_hash:{{{escape_tag(sys_hash)}}})"
            "=>[KNN 1 @embedding $vec AS vector_distance]"
        )
        raw = await raw_command(
            self._redis,
            "FT.SEARCH",
            _INDEX,
            query,
            "PARAMS",
            "2",
            "vec",
            _vector_bytes(vector),
            "RETURN",
            "2",
            "payload",
            "vector_distance",
            "DIALECT",
            "2",
        )
        parsed = _parse_search(raw)
        if parsed is None:
            return None
        payload, distance = parsed
        similarity = semantic_similarity(distance)
        if similarity is None:
            return None
        return decode_completion(payload), similarity

    async def put(
        self,
        *,
        l1_key: str,
        completion: CachedCompletion,
        tenant_id: str,
        sys_hash: str,
        vector: NDArray[np.float32],
        ttl_seconds: int | None = None,
    ) -> None:
        """Write the L1 string and the L2 hash with the same TTL in one pipeline."""
        require_tenant(tenant_id)
        if sys_hash == "":
            raise ValueError("sys_hash is required")
        ttl = self._ttl_seconds if ttl_seconds is None else ttl_seconds
        if ttl <= 0:
            raise ValueError("ttl_seconds must be positive")
        payload = encode_completion(completion)
        await write_expiring_pair(
            self._redis,
            string_key=l1_key,
            string_value=payload,
            hash_key=l2_document_key(l1_key),
            mapping={
                "embedding": _vector_bytes(vector),
                "payload": payload,
                "sys_hash": sys_hash,
                "tenant_id": tenant_id,
            },
            ttl_seconds=ttl,
        )


def _vector_bytes(vector: NDArray[np.float32]) -> bytes:
    array = np.ascontiguousarray(vector, dtype=np.float32).reshape(-1)
    if array.shape != (_DIM,):
        raise ValueError("embedding must be 384-dimensional")
    blob: bytes = array.tobytes()
    return blob


def _parse_search(raw: object) -> tuple[bytes, float] | None:
    values = object_list(raw)
    if not values:
        return None
    count = _as_int(values[0])
    if count == 0 or len(values) < 3:
        return None
    pairs = object_pairs(values[2])
    # An expired hash can linger in the index with a null field list until GC.
    if pairs is None:
        return None
    payload: bytes | None = None
    distance: float | None = None
    for key, value in pairs:
        name = _as_text(key)
        if name == "payload":
            payload = _as_bytes(value)
        elif name in {"vector_distance", "__vector_score"}:
            distance = _as_float(value)
    if payload is None or distance is None:
        return None
    return payload, distance


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        raise RuntimeError("unexpected FT.SEARCH count")
    if isinstance(value, int):
        return value
    if isinstance(value, bytes | str):
        return int(value)
    raise RuntimeError("unexpected FT.SEARCH count")


def _as_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, str):
        return value
    return str(value)


def _as_bytes(value: object) -> bytes | None:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode()
    return None


def _as_float(value: object) -> float:
    if isinstance(value, bool) or value is None:
        raise RuntimeError("vector distance missing")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, bytes):
        return float(value.decode())
    if isinstance(value, str):
        return float(value)
    raise RuntimeError("vector distance missing")


def _object(value: object, label: str) -> dict[str, object]:
    typed = object_dict(value)
    if typed is None:
        raise ValueError(f"{label} must be an object")
    return typed


def _text(body: dict[str, object], field: str) -> str:
    value = body.get(field)
    if not isinstance(value, str):
        raise ValueError(f"cached completion {field} must be a string")
    return value


def _integer(body: dict[str, object], field: str) -> int:
    value = body.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"cached completion {field} must be an integer")
    return value
