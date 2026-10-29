"""In-memory simulation run tracker. One active run per scenario."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from uuid import uuid4


class SimulationBusy(Exception):
    def __init__(self, scenario: str) -> None:
        super().__init__(f"scenario {scenario} is already running")
        self.scenario = scenario


@dataclass(slots=True)
class RunState:
    run_id: str
    scenario: str
    status: str
    count: int = 0
    error: str | None = None


class SimulationRunner:
    """Allows one burst / paraphrase / outage run at a time."""

    def __init__(self) -> None:
        self._running: set[str] = set()
        self.runs: dict[str, RunState] = {}
        self.tasks: dict[str, asyncio.Task[None]] = {}

    def begin(self, scenario: str) -> RunState:
        if scenario in self._running:
            raise SimulationBusy(scenario)
        run_id = uuid4().hex[:16]
        state = RunState(run_id=run_id, scenario=scenario, status="running")
        self._running.add(scenario)
        self.runs[run_id] = state
        return state

    def track(self, run: RunState, task: asyncio.Task[None]) -> None:
        self.tasks[run.run_id] = task

    def finish(self, run: RunState, *, error: str | None = None) -> None:
        run.status = "error" if error else "completed"
        run.error = error
        self._running.discard(run.scenario)
        self.tasks.pop(run.run_id, None)

    def get(self, run_id: str) -> RunState | None:
        return self.runs.get(run_id)
