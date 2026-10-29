"""Golden telemetry events consumed by the control plane in milestone 4."""

from __future__ import annotations

from pathlib import Path

import orjson
from pydantic import TypeAdapter

from app.engine.coerce import object_dict, object_list
from app.schemas.finops import CacheStatus, LatencyBreakdown, Tier, XAegisFinOps
from app.schemas.telemetry import (
    BreakerStateEvent,
    RequestCompletedEvent,
    SimulationCompletedEvent,
    SimulationStartedEvent,
    TelemetryEvent,
)

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "telemetry_events.json"

_ADAPTER: TypeAdapter[TelemetryEvent] = TypeAdapter(TelemetryEvent)


def test_writes_golden_telemetry_events() -> None:
    finops = XAegisFinOps(
        cache_status=CacheStatus.HIT_L2,
        similarity_score=0.964,
        routed_tier=Tier.SIMPLE,
        routed_model="grok-3-mini",
        latency_breakdown_ms=LatencyBreakdown(
            guardrail=0.3,
            l1_lookup=0.6,
            onnx_embed=4.4,
            l2_hnsw=1.7,
            router=0.1,
        ),
        cost_actual_usd=0.0,
        cost_saved_usd=0.0142,
        request_id="chatcmpl-aegis-cache",
        provider="mock-primary",
        gateway_overhead_ms=7.1,
        ttft_ms=7.1,
        tokens_in=12,
        tokens_out=4,
    )
    events = [
        RequestCompletedEvent(
            request_id="chatcmpl-aegis-cache",
            ts="2026-10-27T00:00:00+00:00",
            tenant_id="sandbox",
            run_id="run_paraphrase_1",
            scenario="paraphrase",
            prompt_preview="Translate the English greeting hello into Spanish now.",
            finops=finops,
            total_ms=8.2,
            status="ok",
        ),
        RequestCompletedEvent(
            request_id="chatcmpl-aegis-miss",
            ts="2026-10-27T00:00:01+00:00",
            tenant_id="sandbox",
            prompt_preview="Summarize this Q4 revenue variance in one paragraph.",
            finops=XAegisFinOps(
                cache_status=CacheStatus.MISS,
                similarity_score=None,
                routed_tier=Tier.REASONING,
                routed_model="claude-sonnet-4.5",
                latency_breakdown_ms=LatencyBreakdown(
                    guardrail=0.4,
                    l1_lookup=0.5,
                    onnx_embed=4.5,
                    l2_hnsw=1.8,
                    router=0.2,
                ),
                cost_actual_usd=0.0021,
                cost_saved_usd=0.011,
                request_id="chatcmpl-aegis-miss",
                provider="mock-primary",
                gateway_overhead_ms=8.0,
                ttft_ms=12.0,
                tokens_in=40,
                tokens_out=80,
            ),
            total_ms=120.0,
            status="ok",
        ),
        BreakerStateEvent(
            type="breaker.state",
            provider="mock-primary",
            state="OPEN",
            ts="2026-10-27T00:00:02+00:00",
        ),
        SimulationStartedEvent(
            scenario="outage",
            run_id="run_outage_1",
            count=3,
        ),
        SimulationCompletedEvent(
            scenario="outage",
            run_id="run_outage_1",
            count=3,
        ),
    ]
    dumped = [event.model_dump(mode="json") for event in events]
    for row in dumped:
        _ADAPTER.validate_python(row)
    _FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    _FIXTURE.write_bytes(orjson.dumps(dumped, option=orjson.OPT_INDENT_2) + b"\n")
    loaded = object_list(orjson.loads(_FIXTURE.read_bytes()))
    assert loaded is not None
    assert len(loaded) == 5
    types: set[str] = set()
    for item in loaded:
        mapped = object_dict(item)
        if mapped is None:
            continue
        kind = mapped.get("type")
        if isinstance(kind, str):
            types.add(kind)
    assert types == {
        "request.completed",
        "breaker.state",
        "simulation.started",
        "simulation.completed",
    }
