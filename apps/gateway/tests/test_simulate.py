"""Sandbox simulate endpoints: 202, single-flight, paraphrase under a fresh tenant."""

from __future__ import annotations

import asyncio

from httpx import AsyncClient

from app.state import AppState


async def test_paraphrase_returns_202_and_completes(
    client: AsyncClient, gateway_state: AppState
) -> None:
    response = await client.post("/v1/aegis/simulate/paraphrase")
    assert response.status_code == 202
    run_id = response.json()["run_id"]
    assert run_id
    run = await _wait_run(gateway_state, run_id)
    assert run.status == "completed"
    assert run.count == 2
    assert run.scenario == "paraphrase"


async def test_one_run_per_scenario(
    client: AsyncClient, gateway_state: AppState
) -> None:
    first = await client.post("/v1/aegis/simulate/burst")
    assert first.status_code == 202
    second = await client.post("/v1/aegis/simulate/burst")
    assert second.status_code == 409
    run_id = first.json()["run_id"]
    run = await _wait_run(gateway_state, run_id, limit_s=60.0)
    assert run.status == "completed"
    assert run.count == 50


async def test_unknown_scenario_is_404(client: AsyncClient) -> None:
    response = await client.post("/v1/aegis/simulate/nope")
    assert response.status_code == 404


async def _wait_run(state: AppState, run_id: str, limit_s: float = 30.0):
    deadline = asyncio.get_running_loop().time() + limit_s
    while asyncio.get_running_loop().time() < deadline:
        run = state.simulations.get(run_id)
        if run is not None and run.status != "running":
            return run
        await asyncio.sleep(0.05)
        await state.dispatcher.wait_idle()
    raise AssertionError(f"run {run_id} did not finish")
