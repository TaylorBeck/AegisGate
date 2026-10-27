"""Session-scoped ASGI client backed by the real lifespan.

Fails fast with a clear infra hint if Redis or Postgres cannot be reached.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.state import AppState

_INFRA_HINT = "run `docker compose up -d valkey postgres`"


@pytest.fixture(scope="session")
async def client() -> AsyncIterator[AsyncClient]:
    try:
        async with LifespanManager(app, startup_timeout=120) as manager:
            transport = ASGITransport(app=manager.app)
            async with AsyncClient(
                transport=transport,
                base_url="http://test",
            ) as http:
                yield http
    except Exception as exc:
        pytest.fail(
            f"Gateway lifespan failed to start. {_INFRA_HINT}. "
            f"Underlying error: {type(exc).__name__}: {exc}"
        )


@pytest.fixture(scope="session")
async def gateway_state(client: AsyncClient) -> AppState:
    """Lifespan AppState, available after the session client has started."""
    _ = client
    core = getattr(app.state, "core", None)
    assert isinstance(core, AppState)
    return core
