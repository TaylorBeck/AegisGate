"""FinOps counterfactual math: cache hit, downgrade spread, equal baseline."""

from __future__ import annotations

from app.catalog import ModelSpec
from app.schemas.finops import Tier
from app.telemetry.finops import compute_costs

_SIMPLE = ModelSpec(
    name="grok-3-mini",
    provider="xai",
    tier=Tier.SIMPLE,
    price_in_per_m=0.30,
    price_out_per_m=0.50,
)
_FRONTIER = ModelSpec(
    name="gpt-5",
    provider="openai",
    tier=Tier.AGENTIC_CODE,
    price_in_per_m=15.0,
    price_out_per_m=75.0,
)


def test_cache_hit_zeros_actual_and_saves_baseline() -> None:
    actual, saved = compute_costs(12, 4, _SIMPLE, _FRONTIER, cache_hit=True)
    assert actual == 0.0
    expected_baseline = (12 / 1_000_000) * 15.0 + (4 / 1_000_000) * 75.0
    assert saved == expected_baseline


def test_downgrade_spread() -> None:
    actual, saved = compute_costs(12, 4, _SIMPLE, _FRONTIER)
    expected_actual = (12 / 1_000_000) * 0.30 + (4 / 1_000_000) * 0.50
    expected_baseline = (12 / 1_000_000) * 15.0 + (4 / 1_000_000) * 75.0
    assert actual == expected_actual
    assert saved == expected_baseline - expected_actual


def test_baseline_equals_actual() -> None:
    actual, saved = compute_costs(100, 20, _FRONTIER, _FRONTIER)
    expected = (100 / 1_000_000) * 15.0 + (20 / 1_000_000) * 75.0
    assert actual == expected
    assert saved == 0.0
