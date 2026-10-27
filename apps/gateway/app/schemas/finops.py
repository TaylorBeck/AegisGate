"""FinOps metadata carried on cache hits and completed requests."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class CacheStatus(StrEnum):
    """How the gateway satisfied the request."""

    HIT_L1 = "HIT_L1"
    HIT_L2 = "HIT_L2"
    MISS = "MISS"


class Tier(StrEnum):
    """Zero-shot complexity tier. Order is cheapest to most expensive."""

    SIMPLE = "SIMPLE"
    REASONING = "REASONING"
    AGENTIC_CODE = "AGENTIC_CODE"


class LatencyBreakdown(BaseModel):
    """Per-stage gateway overhead in milliseconds. Skipped stages are null."""

    model_config = ConfigDict(extra="forbid")

    guardrail: float
    l1_lookup: float
    onnx_embed: float | None = None
    l2_hnsw: float | None = None
    router: float | None = None


class FailoverInfo(BaseModel):
    """Upstream provider that was skipped or failed over."""

    model_config = ConfigDict(extra="forbid")

    from_provider: str
    reason: str


class XAegisFinOps(BaseModel):
    """`x_aegis_finops` object. Additive fields default so cache hits can omit them."""

    model_config = ConfigDict(extra="forbid")

    cache_status: CacheStatus
    similarity_score: float | None
    routed_tier: Tier
    routed_model: str
    latency_breakdown_ms: LatencyBreakdown
    cost_actual_usd: float
    cost_saved_usd: float
    request_id: str = ""
    provider: str | None = None
    failover: FailoverInfo | None = None
    gateway_overhead_ms: float = 0.0
    ttft_ms: float | None = None
    tokens_in: int = 0
    tokens_out: int = 0
