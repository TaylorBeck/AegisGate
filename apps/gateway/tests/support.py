"""Shared helpers for gateway tests."""

from __future__ import annotations

from uuid import uuid4

from app.engine.cache import CachedCompletion
from app.schemas.finops import Tier


def new_tenant(prefix: str = "t") -> str:
    """Return a tenant id that matches the cache tag grammar."""
    return f"{prefix}{uuid4().hex[:12]}"


def sample_completion(
    content: str = "hola",
    *,
    tier: Tier = Tier.SIMPLE,
) -> CachedCompletion:
    """A stored completion with stable token counts."""
    return CachedCompletion(
        content=content,
        model="grok-3-mini",
        provider="xai",
        tier=tier,
        tokens_in=12,
        tokens_out=4,
        finish_reason="stop",
        created_at=1_700_000_000,
    )
