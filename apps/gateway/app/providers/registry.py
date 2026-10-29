"""Build named providers and circuit breakers from settings and the catalog."""

from __future__ import annotations

from collections.abc import Callable

import httpx

from app.catalog import Catalog
from app.config import Settings
from app.providers.anthropic import AnthropicProvider
from app.providers.base import Provider
from app.providers.breaker import BreakerSet, BreakerState, CircuitBreaker
from app.providers.mock import MockProvider
from app.providers.openai_compat import OpenAICompatProvider
from app.sandbox.chaos import ChaosController

MOCK_PRIMARY = "mock-primary"
MOCK_SECONDARY = "mock-secondary"
MOCK_CHAIN: tuple[str, ...] = (MOCK_PRIMARY, MOCK_SECONDARY)


class ProviderRegistry:
    """Lookup by catalog provider name, plus the sandbox mock chain."""

    def __init__(self, providers: dict[str, Provider]) -> None:
        self._providers = providers

    def get(self, name: str) -> Provider | None:
        return self._providers.get(name)

    def require(self, name: str) -> Provider:
        provider = self._providers.get(name)
        if provider is None:
            raise KeyError(name)
        return provider

    def names(self) -> tuple[str, ...]:
        return tuple(self._providers)


def build_providers(
    settings: Settings,
    http: httpx.AsyncClient,
    chaos: ChaosController,
) -> ProviderRegistry:
    """Construct mock, OpenAI-compat, and Anthropic adapters."""
    providers: dict[str, Provider] = {
        MOCK_PRIMARY: MockProvider(
            MOCK_PRIMARY,
            chaos,
            ttft_ms=settings.mock_ttft_ms,
            tokens_per_sec=settings.mock_tokens_per_sec,
        ),
        MOCK_SECONDARY: MockProvider(
            MOCK_SECONDARY,
            chaos,
            ttft_ms=settings.mock_ttft_ms,
            tokens_per_sec=settings.mock_tokens_per_sec,
        ),
        "openai": OpenAICompatProvider(
            "openai",
            http,
            base_url=settings.openai_compat_base_url,
            api_key=settings.openai_api_key.get_secret_value(),
        ),
        "xai": OpenAICompatProvider(
            "xai",
            http,
            base_url=settings.xai_base_url,
            api_key=_first_secret(
                settings.xai_api_key.get_secret_value(),
                settings.openai_api_key.get_secret_value(),
            ),
        ),
        "google": OpenAICompatProvider(
            "google",
            http,
            base_url=settings.google_base_url,
            api_key=_first_secret(
                settings.google_api_key.get_secret_value(),
                settings.openai_api_key.get_secret_value(),
            ),
        ),
        "anthropic": AnthropicProvider(
            "anthropic",
            http,
            base_url=settings.anthropic_base_url,
            api_key=settings.anthropic_api_key.get_secret_value(),
        ),
    }
    return ProviderRegistry(providers)


def build_breakers(
    registry: ProviderRegistry,
    catalog: Catalog,
    on_transition: Callable[[str, BreakerState], None] | None,
) -> BreakerSet:
    names = set(registry.names())
    for models in catalog.tiers.values():
        for spec in models:
            names.add(spec.provider)
    names.add(catalog.baseline.provider)
    breakers = {
        name: CircuitBreaker(name, on_transition=on_transition)
        for name in sorted(names)
    }
    return BreakerSet(breakers)


def use_mock_providers(
    tenant: str,
    settings: Settings,
    upstream_key: str | None,
) -> bool:
    """Sandbox tenants and demo mode use mocks unless a custom upstream key is set."""
    if upstream_key:
        return False
    if tenant == "sandbox" or tenant.startswith("sandbox_"):
        return True
    return settings.demo_sandbox_mode


def _first_secret(*values: str) -> str:
    for value in values:
        if value:
            return value
    return ""
