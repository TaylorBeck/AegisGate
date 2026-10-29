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
from app.providers.breaker import BreakerSet
from app.providers.registry import ProviderRegistry
from app.sandbox.chaos import ChaosController
from app.sandbox.runner import SimulationRunner
from app.telemetry.dispatcher import TelemetryDispatcher
from app.tenancy import VirtualKeyStore


@dataclass(slots=True)
class AppState:
    """Lifespan-owned resources, including providers and telemetry."""

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
    providers: ProviderRegistry
    breakers: BreakerSet
    dispatcher: TelemetryDispatcher
    chaos: ChaosController
    virtual_keys: VirtualKeyStore
    simulations: SimulationRunner


def get_state(request: Request) -> AppState:
    """FastAPI dependency that returns the lifespan `AppState`."""
    core = getattr(request.app.state, "core", None)
    if not isinstance(core, AppState):
        raise RuntimeError("gateway AppState is not initialized")
    return core
