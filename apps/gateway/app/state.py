"""Process-wide gateway singletons attached during lifespan."""

from __future__ import annotations

from dataclasses import dataclass

import asyncpg
import httpx
from fastapi import Request
from redis.asyncio import Redis

from app.config import Settings
from app.engine.admission import Admission
from app.engine.cache import SemanticCache
from app.engine.embedder import LocalVectorEngine
from app.engine.guardrail import Guardrail
from app.engine.router import ComplexityRouter


@dataclass(slots=True)
class AppState:
    """Lifespan-owned resources. Milestone 3 adds providers, breakers, and telemetry."""

    settings: Settings
    redis: Redis
    redis_blocking: Redis
    pg: asyncpg.Pool
    http: httpx.AsyncClient
    embedder: LocalVectorEngine
    cache: SemanticCache
    router: ComplexityRouter
    guardrail: Guardrail
    admission: Admission


def get_state(request: Request) -> AppState:
    """FastAPI dependency that returns the lifespan `AppState`."""
    core = getattr(request.app.state, "core", None)
    if not isinstance(core, AppState):
        raise RuntimeError("gateway AppState is not initialized")
    return core
