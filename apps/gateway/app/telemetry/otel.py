"""Optional `gen_ai.*` spans. No-ops unless `OTEL_EXPORTER_OTLP_ENDPOINT` is set."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from app.schemas.finops import XAegisFinOps

logger = logging.getLogger("aegis.gateway.otel")

_state: dict[str, Any] = {"tracer": None, "ready": False}


def emit_gen_ai_span(
    finops: XAegisFinOps,
    *,
    endpoint: str,
    started_at: datetime | None = None,
) -> None:
    """Emit a chat span with an explicit start time when OTLP is configured."""
    if not endpoint:
        return
    tracer = _tracer(endpoint)
    if tracer is None:
        return
    start_ns = None
    if started_at is not None:
        start_ns = int(started_at.timestamp() * 1_000_000_000)
    try:
        span = tracer.start_span(
            "gen_ai.chat",
            start_time=start_ns,
            attributes={
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": finops.routed_model,
                "gen_ai.usage.input_tokens": finops.tokens_in,
                "gen_ai.usage.output_tokens": finops.tokens_out,
                "gen_ai.response.finish_reason": "stop",
                "aegis.cache_status": finops.cache_status.value,
                "aegis.provider": finops.provider or "",
            },
        )
        span.end()
    except Exception:
        logger.exception("failed to emit gen_ai span")


def _tracer(endpoint: str) -> Any:
    if _state["ready"]:
        return _state["tracer"]
    _state["ready"] = True
    try:
        module = __import__("opentelemetry.trace", fromlist=["get_tracer"])
        sdk_trace = __import__("opentelemetry.sdk.trace", fromlist=["TracerProvider"])
        export = __import__(
            "opentelemetry.sdk.trace.export", fromlist=["BatchSpanProcessor"]
        )
        resource_mod = __import__("opentelemetry.sdk.resources", fromlist=["Resource"])
        otlp = __import__(
            "opentelemetry.exporter.otlp.proto.http.trace_exporter",
            fromlist=["OTLPSpanExporter"],
        )
    except ImportError:
        logger.info("opentelemetry is not installed; gen_ai spans disabled")
        return None
    provider = sdk_trace.TracerProvider(
        resource=resource_mod.Resource.create({"service.name": "aegis-gateway"})
    )
    traces_url = endpoint.rstrip("/") + "/v1/traces"
    provider.add_span_processor(
        export.BatchSpanProcessor(otlp.OTLPSpanExporter(endpoint=traces_url))
    )
    module.set_tracer_provider(provider)
    _state["tracer"] = module.get_tracer("aegis.gateway")
    return _state["tracer"]
