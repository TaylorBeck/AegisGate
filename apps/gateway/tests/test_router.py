"""Anchor classification, margin rule, and session affinity."""

from __future__ import annotations

import orjson

from app.catalog import load_catalog
from app.engine.anchors import ANCHORS
from app.engine.router import AffinityRecord, select_tier
from app.schemas.finops import Tier
from app.state import AppState


def test_each_tier_has_fifty_unique_anchors() -> None:
    seen: set[str] = set()
    for tier in Tier:
        rows = ANCHORS[tier]
        assert len(rows) == 50
        assert len(set(rows)) == 50
        seen.update(rows)
    assert len(seen) == 150


def test_margin_blocks_a_small_escalation() -> None:
    scores = {
        Tier.SIMPLE: 0.90,
        Tier.REASONING: 0.93,
        Tier.AGENTIC_CODE: 0.93,
    }
    assert select_tier(scores, margin=0.03) is Tier.SIMPLE


def test_margin_escalates_only_past_the_selected_tier() -> None:
    reasoning = {
        Tier.SIMPLE: 0.50,
        Tier.REASONING: 0.80,
        Tier.AGENTIC_CODE: 0.82,
    }
    agentic = {
        Tier.SIMPLE: 0.40,
        Tier.REASONING: 0.50,
        Tier.AGENTIC_CODE: 0.90,
    }
    assert select_tier(reasoning, margin=0.03) is Tier.REASONING
    assert select_tier(agentic, margin=0.03) is Tier.AGENTIC_CODE


def test_default_catalog_and_override() -> None:
    catalog = load_catalog("")
    assert catalog.baseline.name == "gpt-5"
    assert [model.provider for model in catalog.models_for(Tier.SIMPLE)] == [
        "xai",
        "google",
    ]
    assert len(catalog.models_for(Tier.REASONING)) == 2
    assert len(catalog.models_for(Tier.AGENTIC_CODE)) == 2

    raw = orjson.dumps(
        {
            "baseline": {
                "name": "frontier",
                "provider": "openai",
                "tier": "AGENTIC_CODE",
                "price_in_per_m": 15,
                "price_out_per_m": 75,
            },
            "tiers": {
                "SIMPLE": [
                    {
                        "name": "custom-small",
                        "provider": "mock",
                        "tier": "SIMPLE",
                        "price_in_per_m": 0.1,
                        "price_out_per_m": 0.2,
                    }
                ],
                "REASONING": [
                    {
                        "name": "custom-mid",
                        "provider": "mock",
                        "tier": "REASONING",
                        "price_in_per_m": 1,
                        "price_out_per_m": 2,
                    }
                ],
                "AGENTIC_CODE": [
                    {
                        "name": "custom-large",
                        "provider": "mock",
                        "tier": "AGENTIC_CODE",
                        "price_in_per_m": 10,
                        "price_out_per_m": 20,
                    }
                ],
            },
        }
    ).decode()
    override = load_catalog(raw)
    assert override.models_for(Tier.SIMPLE)[0].name == "custom-small"
    assert override.baseline.name == "frontier"


async def test_anchor_utterances_classify_to_their_tier(
    gateway_state: AppState,
) -> None:
    for tier in (Tier.SIMPLE, Tier.REASONING, Tier.AGENTIC_CODE):
        for index in (0, 49):
            vector = await gateway_state.embedder.embed(ANCHORS[tier][index])
            classified, scores = gateway_state.router.classify(vector)
            assert classified is tier
            assert set(scores) == set(Tier)


async def test_affinity_rules(gateway_state: AppState) -> None:
    simple = await gateway_state.embedder.embed(ANCHORS[Tier.SIMPLE][0])
    agentic = await gateway_state.embedder.embed(ANCHORS[Tier.AGENTIC_CODE][0])
    sticky_record = AffinityRecord(
        tier=Tier.REASONING,
        provider="anthropic",
        model="claude-sonnet-4.5",
    )

    downgrade = gateway_state.router.resolve(
        simple, message_count=3, affinity=sticky_record
    )
    assert downgrade.sticky is True
    assert downgrade.escalated is False
    assert downgrade.tier is Tier.REASONING
    assert downgrade.model.name == "claude-sonnet-4.5"
    assert downgrade.provider_chain[0] == "anthropic"

    escalated = gateway_state.router.resolve(
        agentic,
        message_count=4,
        affinity=AffinityRecord(tier=Tier.SIMPLE, provider="xai", model="grok-3-mini"),
    )
    assert escalated.escalated is True
    assert escalated.sticky is False
    assert escalated.tier is Tier.AGENTIC_CODE
    assert escalated.provider_chain[0] == "anthropic"

    short = gateway_state.router.resolve(
        simple, message_count=2, affinity=sticky_record
    )
    assert short.sticky is False
    assert short.escalated is False
    assert short.tier is Tier.SIMPLE

    fresh = gateway_state.router.resolve(simple, message_count=5, affinity=None)
    assert fresh.sticky is False
    assert fresh.tier is Tier.SIMPLE

    opened = gateway_state.router.resolve(
        simple,
        message_count=3,
        affinity=sticky_record,
        open_providers=frozenset({"anthropic"}),
    )
    assert opened.tier is Tier.REASONING
    assert opened.sticky is False
    assert opened.provider_chain[0] == "openai"
    assert "anthropic" not in opened.provider_chain
