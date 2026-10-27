"""Model catalog: ordered provider chain per tier and the frontier baseline."""

from __future__ import annotations

from dataclasses import dataclass

import orjson

from app.engine.coerce import object_dict, object_list
from app.schemas.finops import Tier


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One routable model and its public token prices, in USD per million."""

    name: str
    provider: str
    tier: Tier
    price_in_per_m: float
    price_out_per_m: float


@dataclass(frozen=True, slots=True)
class Catalog:
    """Provider chains keyed by tier, plus the frontier baseline model."""

    tiers: dict[Tier, tuple[ModelSpec, ...]]
    baseline: ModelSpec

    def models_for(self, tier: Tier) -> tuple[ModelSpec, ...]:
        models = self.tiers.get(tier)
        if not models:
            raise KeyError(tier)
        return models


def default_catalog() -> Catalog:
    """Built-in chain. Each tier lists the primary provider first."""
    simple = (
        ModelSpec(
            name="grok-3-mini",
            provider="xai",
            tier=Tier.SIMPLE,
            price_in_per_m=0.30,
            price_out_per_m=0.50,
        ),
        ModelSpec(
            name="gemini-2.5-flash",
            provider="google",
            tier=Tier.SIMPLE,
            price_in_per_m=0.15,
            price_out_per_m=0.60,
        ),
    )
    reasoning = (
        ModelSpec(
            name="claude-sonnet-4.5",
            provider="anthropic",
            tier=Tier.REASONING,
            price_in_per_m=3.0,
            price_out_per_m=15.0,
        ),
        ModelSpec(
            name="gpt-5-mini",
            provider="openai",
            tier=Tier.REASONING,
            price_in_per_m=0.40,
            price_out_per_m=1.60,
        ),
    )
    agentic = (
        ModelSpec(
            name="claude-opus-4.5",
            provider="anthropic",
            tier=Tier.AGENTIC_CODE,
            price_in_per_m=15.0,
            price_out_per_m=75.0,
        ),
        ModelSpec(
            name="gpt-5",
            provider="openai",
            tier=Tier.AGENTIC_CODE,
            price_in_per_m=15.0,
            price_out_per_m=75.0,
        ),
    )
    baseline = ModelSpec(
        name="gpt-5",
        provider="openai",
        tier=Tier.AGENTIC_CODE,
        price_in_per_m=15.0,
        price_out_per_m=75.0,
    )
    return Catalog(
        tiers={
            Tier.SIMPLE: simple,
            Tier.REASONING: reasoning,
            Tier.AGENTIC_CODE: agentic,
        },
        baseline=baseline,
    )


def load_catalog(raw: str) -> Catalog:
    """Return the default catalog, or parse an `AEGIS_CATALOG` JSON override."""
    if not raw.strip():
        return default_catalog()
    try:
        parsed: object = orjson.loads(raw)
    except orjson.JSONDecodeError as exc:
        raise ValueError("AEGIS_CATALOG is not valid JSON") from exc
    body = _object(parsed, "catalog")
    baseline = _spec(_required(body, "baseline"))
    tiers_raw = _object(_required(body, "tiers"), "tiers")
    tiers: dict[Tier, tuple[ModelSpec, ...]] = {}
    for tier in (Tier.SIMPLE, Tier.REASONING, Tier.AGENTIC_CODE):
        rows = object_list(tiers_raw.get(tier.value))
        if not rows:
            raise ValueError(f"catalog missing tier {tier.value}")
        specs: list[ModelSpec] = []
        for row in rows:
            spec = _spec(row)
            if spec.tier is not tier:
                raise ValueError("model tier does not match its catalog key")
            specs.append(spec)
        tiers[tier] = tuple(specs)
    return Catalog(tiers=tiers, baseline=baseline)


def _spec(value: object) -> ModelSpec:
    body = _object(value, "model")
    tier_raw = _string(body, "tier")
    try:
        tier = Tier(tier_raw)
    except ValueError as exc:
        raise ValueError(f"unknown tier {tier_raw}") from exc
    return ModelSpec(
        name=_string(body, "name"),
        provider=_string(body, "provider"),
        tier=tier,
        price_in_per_m=_number(body, "price_in_per_m"),
        price_out_per_m=_number(body, "price_out_per_m"),
    )


def _object(value: object, label: str) -> dict[str, object]:
    typed = object_dict(value)
    if typed is None:
        raise ValueError(f"{label} must be an object")
    return typed


def _required(body: dict[str, object], field: str) -> object:
    if field not in body:
        raise ValueError(f"catalog missing {field}")
    return body[field]


def _string(body: dict[str, object], field: str) -> str:
    value = _required(body, field)
    if not isinstance(value, str) or value == "":
        raise ValueError(f"catalog {field} must be a string")
    return value


def _number(body: dict[str, object], field: str) -> float:
    value = _required(body, field)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"catalog {field} must be a number")
    return float(value)
