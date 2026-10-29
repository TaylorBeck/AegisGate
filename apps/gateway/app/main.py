"""ASGI entrypoint: lifespan singletons, ORJSON responses, health routes."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

from app.api.deps import GatewayError
from app.api.v1.aegis import router as aegis_router
from app.api.v1.chat import router as chat_router
from app.api.v1.health import router as health_router
from app.catalog import load_catalog
from app.config import Settings
from app.db.postgres import close_pool, create_pool, ensure_partitions
from app.db.redis import aclose, create_pools
from app.engine.admission import Admission
from app.engine.cache import SemanticCache
from app.engine.embedder import LocalVectorEngine
from app.engine.guardrail import Guardrail
from app.engine.router import ComplexityRouter
from app.providers.registry import build_breakers, build_providers
from app.responses import ORJSONResponse
from app.sandbox.chaos import ChaosController
from app.sandbox.runner import SimulationRunner
from app.state import AppState
from app.telemetry.dispatcher import TelemetryDispatcher
from app.tenancy import VirtualKeyStore

logger = logging.getLogger("aegis.gateway")


def _cors_origins(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Build AppState in invariant order, then tear it down on shutdown."""
    settings = Settings()
    redis, redis_blocking = await create_pools(settings)
    pg = await create_pool(settings)
    await ensure_partitions(pg)
    http = httpx.AsyncClient(
        http2=True,
        limits=httpx.Limits(
            max_keepalive_connections=settings.http_max_keepalive_connections,
            max_connections=settings.http_max_connections,
        ),
    )
    embedder = LocalVectorEngine(
        model_name=settings.embed_model,
        cache_dir=settings.fastembed_cache_path,
        threads=settings.embed_threads,
    )
    cache = SemanticCache(redis, ttl_seconds=settings.cache_ttl_seconds)
    admission = Admission(
        redis,
        capacity=settings.rate_limit_capacity,
        refill_per_sec=settings.rate_limit_refill_per_sec,
    )
    guardrail = Guardrail()
    catalog = load_catalog(settings.aegis_catalog)
    router = ComplexityRouter(catalog, margin=settings.router_margin)
    chaos = ChaosController()
    providers = build_providers(settings, http, chaos)
    dispatcher = TelemetryDispatcher(
        redis,
        pg,
        cache,
        otlp_endpoint=settings.otel_exporter_otlp_endpoint,
    )
    breakers = build_breakers(providers, catalog, dispatcher.offer_breaker)
    virtual_keys = VirtualKeyStore(settings.aegis_virtual_keys.get_secret_value())
    simulations = SimulationRunner()
    try:
        warmup_ms = await embedder.warmup_async()
        logger.info("onnx warmup complete in %.2f ms", warmup_ms)
        await cache.ensure_index()
        started = time.perf_counter()
        await router.build_async(embedder)
        anchor_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "anchor matrix ready in %.2f ms (%d anchors)",
            anchor_ms,
            router.anchor_count,
        )
        dispatcher.start()
        app.state.core = AppState(
            settings=settings,
            redis=redis,
            redis_blocking=redis_blocking,
            pg=pg,
            http=http,
            embedder=embedder,
            cache=cache,
            router=router,
            guardrail=guardrail,
            admission=admission,
            providers=providers,
            breakers=breakers,
            dispatcher=dispatcher,
            chaos=chaos,
            virtual_keys=virtual_keys,
            simulations=simulations,
        )
        yield
    finally:
        await dispatcher.stop()
        embedder.close()
        await http.aclose()
        await close_pool(pg)
        await aclose(redis)
        await aclose(redis_blocking)


app = FastAPI(
    title="AegisGate",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(Settings().aegis_cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(aegis_router)


@app.exception_handler(GatewayError)
async def gateway_error_handler(_request: Request, exc: GatewayError) -> Response:
    return ORJSONResponse(exc.body, status_code=exc.status)
