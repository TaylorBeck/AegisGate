"""ASGI entrypoint: lifespan singletons, ORJSON responses, health routes."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

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
from app.responses import ORJSONResponse
from app.state import AppState

logger = logging.getLogger("aegis.gateway")


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
    router = ComplexityRouter(
        load_catalog(settings.aegis_catalog),
        margin=settings.router_margin,
    )
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
        )
        yield
    finally:
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
app.include_router(health_router)
