"""Telemetry events published on `aegis:events` and `/v1/aegis/events`."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.finops import XAegisFinOps

BreakerName = Literal["CLOSED", "OPEN", "HALF_OPEN"]


class RequestCompletedEvent(BaseModel):
    """One finished chat completion, including cache hits."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["request.completed"] = "request.completed"
    request_id: str
    ts: str
    tenant_id: str
    run_id: str | None = None
    scenario: str | None = None
    prompt_preview: str
    finops: XAegisFinOps
    total_ms: float
    status: str


class BreakerStateEvent(BaseModel):
    """Circuit breaker transition."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["breaker.state"] = "breaker.state"
    provider: str
    state: BreakerName
    ts: str


class SimulationStartedEvent(BaseModel):
    """A sandbox scenario has begun."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["simulation.started"] = "simulation.started"
    scenario: str
    run_id: str
    count: int


class SimulationCompletedEvent(BaseModel):
    """A sandbox scenario has finished."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["simulation.completed"] = "simulation.completed"
    scenario: str
    run_id: str
    count: int


TelemetryEvent = Annotated[
    RequestCompletedEvent
    | BreakerStateEvent
    | SimulationStartedEvent
    | SimulationCompletedEvent,
    Field(discriminator="type"),
]
