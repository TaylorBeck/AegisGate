"""Liveness and readiness probes."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request

from app.db.postgres import ping as ping_postgres
from app.db.redis import ping as ping_redis
from app.responses import ORJSONResponse
from app.state import get_state

logger = logging.getLogger("aegis.gateway.health")

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Process is up. Does not touch Redis, Postgres, or ONNX."""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request) -> ORJSONResponse:
    """Ping Redis and Postgres and report ONNX warmup status."""
    state = get_state(request)
    redis_ok = False
    postgres_ok = False
    try:
        redis_ok = await ping_redis(state.redis)
    except Exception:
        logger.exception("redis ping failed")
    try:
        postgres_ok = await ping_postgres(state.pg)
    except Exception:
        logger.exception("postgres ping failed")

    ready = redis_ok and postgres_ok and state.embedder.warm
    body: dict[str, Any] = {
        "status": "ok" if ready else "degraded",
        "onnx_warm": state.embedder.warm,
        "warmup_ms": round(state.embedder.warmup_ms, 3),
        "redis": "ok" if redis_ok else "error",
        "postgres": "ok" if postgres_ok else "error",
    }
    return ORJSONResponse(body, status_code=200 if ready else 503)
