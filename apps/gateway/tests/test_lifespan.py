"""Lifespan singletons, ONNX warmup, and embedding invariants."""

from __future__ import annotations

import numpy as np
from httpx import AsyncClient
from numpy.typing import NDArray

from app.main import app
from app.state import AppState


async def test_singletons_exist(client: AsyncClient) -> None:
    _ = client
    core = getattr(app.state, "core", None)
    assert isinstance(core, AppState)
    assert core.redis is not None
    assert core.redis_blocking is not core.redis
    assert (
        core.redis_blocking.connection_pool.connection_kwargs.get("socket_timeout")
        is None
    )
    assert core.pg is not None
    assert core.http is not None
    assert core.embedder.warm is True
    assert core.embedder.warmup_ms > 0.0
    assert core.cache is not None
    assert core.guardrail is not None
    assert core.admission is not None
    assert core.router.anchor_count == 150
    assert core.providers.get("mock-primary") is not None
    assert core.breakers.get("mock-primary") is not None
    assert core.dispatcher is not None
    assert core.chaos is not None
    assert core.virtual_keys.lookup("sk-aegis-sandbox") == "sandbox"


async def test_embedding_is_384d_float32_unit_norm(client: AsyncClient) -> None:
    _ = client
    core = getattr(app.state, "core", None)
    assert isinstance(core, AppState)
    vecs: NDArray[np.float32] = core.embedder.embed_batch_sync(["hello aegis"])
    assert vecs.shape == (1, 384)
    assert vecs.dtype == np.float32
    first = np.asarray(vecs[0], dtype=np.float32)
    assert abs(float(np.linalg.norm(first)) - 1.0) < 1e-5


async def test_async_embed_is_384d_float32_unit_norm(client: AsyncClient) -> None:
    _ = client
    core = getattr(app.state, "core", None)
    assert isinstance(core, AppState)
    vec = await core.embedder.embed("hello aegis")
    assert vec.shape == (384,)
    assert vec.dtype == np.float32
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5
