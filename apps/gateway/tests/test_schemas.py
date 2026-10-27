"""Wire-shape tests for the OpenAI request and FinOps metadata models."""

from __future__ import annotations

from app.schemas.finops import CacheStatus, LatencyBreakdown, Tier, XAegisFinOps
from app.schemas.openai import ChatCompletionRequest


def test_finops_wire_names() -> None:
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
    )
    dumped = finops.model_dump(mode="json")
    assert dumped["cache_status"] == "HIT_L2"
    assert dumped["similarity_score"] == 0.964
    assert dumped["routed_tier"] == "SIMPLE"
    assert dumped["latency_breakdown_ms"]["onnx_embed"] == 4.4
    assert dumped["cost_actual_usd"] == 0.0
    assert dumped["cost_saved_usd"] == 0.0142


def test_skipped_stages_are_null() -> None:
    breakdown = LatencyBreakdown(guardrail=0.2, l1_lookup=0.4)
    dumped = breakdown.model_dump(mode="json")
    assert dumped["onnx_embed"] is None
    assert dumped["l2_hnsw"] is None
    assert dumped["router"] is None


def test_request_keeps_unknown_openai_fields_and_content_parts() -> None:
    request = ChatCompletionRequest.model_validate(
        {
            "model": "auto",
            "temperature": 0.2,
            "user": "tenant-user",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            ],
        }
    )
    assert request.model_extra is not None
    assert request.model_extra["user"] == "tenant-user"
    content = request.messages[0].content
    assert isinstance(content, list)
    assert content[0].text == "hi"
