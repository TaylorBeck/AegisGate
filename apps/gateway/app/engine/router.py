"""Anchor-matrix complexity router with monotonic session affinity."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np
import orjson
from numpy.typing import NDArray

from app.catalog import Catalog, ModelSpec
from app.engine.anchors import ANCHORS
from app.engine.coerce import object_dict
from app.engine.embedder import LocalVectorEngine
from app.schemas.finops import Tier

_TIER_ORDER: tuple[Tier, ...] = (Tier.SIMPLE, Tier.REASONING, Tier.AGENTIC_CODE)


@dataclass(frozen=True, slots=True)
class AffinityRecord:
    """Sticky provider remembered for a multi-turn conversation."""

    tier: Tier
    provider: str
    model: str


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """Chosen tier, model, and provider chain for a cache miss."""

    tier: Tier
    model: ModelSpec
    provider_chain: tuple[str, ...]
    sticky: bool
    escalated: bool
    scores: dict[Tier, float]


def select_tier(scores: dict[Tier, float], *, margin: float) -> Tier:
    """Stay on the cheaper tier unless a higher one wins by `margin`."""
    chosen = Tier.SIMPLE
    for tier in (Tier.REASONING, Tier.AGENTIC_CODE):
        if scores[tier] > scores[chosen] + margin:
            chosen = tier
    return chosen


def apply_affinity(
    classified: Tier,
    scores: dict[Tier, float],
    *,
    message_count: int,
    affinity: AffinityRecord | None,
    catalog: Catalog,
    open_providers: frozenset[str] = frozenset(),
) -> RouteDecision:
    """Keep the sticky tier unless the turn escalates to `AGENTIC_CODE`.

    Affinity is ignored for two or fewer messages. An open sticky provider
    fails over inside the sticky tier so the caller can overwrite the record.
    """
    tier = classified
    sticky = False
    escalated = False
    preferred: str | None = None
    sticky_record: AffinityRecord | None = None

    if message_count > 2 and affinity is not None:
        if classified == Tier.AGENTIC_CODE and affinity.tier != Tier.AGENTIC_CODE:
            escalated = True
            tier = Tier.AGENTIC_CODE
        elif affinity.provider in open_providers:
            tier = affinity.tier
        else:
            tier = affinity.tier
            sticky = True
            preferred = affinity.provider
            sticky_record = affinity

    models = catalog.models_for(tier)
    chain = _ordered_providers(
        tuple(model.provider for model in models),
        preferred,
        open_providers,
    )
    model = _model_for(
        models, chain[0], sticky_record if chain[0] == preferred else None
    )
    return RouteDecision(
        tier=tier,
        model=model,
        provider_chain=chain,
        sticky=sticky and chain[0] == preferred,
        escalated=escalated,
        scores=scores,
    )


def encode_affinity(record: AffinityRecord) -> bytes:
    return orjson.dumps(
        {
            "model": record.model,
            "provider": record.provider,
            "tier": record.tier.value,
        }
    )


def decode_affinity(payload: bytes | None) -> AffinityRecord | None:
    if not payload:
        return None
    parsed: object = orjson.loads(payload)
    raw = object_dict(parsed)
    if raw is None:
        return None
    tier = raw.get("tier")
    provider = raw.get("provider")
    model = raw.get("model")
    if (
        not isinstance(tier, str)
        or not isinstance(provider, str)
        or not isinstance(model, str)
    ):
        return None
    try:
        parsed = Tier(tier)
    except ValueError:
        return None
    return AffinityRecord(tier=parsed, provider=provider, model=model)


class ComplexityRouter:
    """Dot-product classifier over the pre-embedded anchor matrix."""

    def __init__(self, catalog: Catalog, *, margin: float = 0.03) -> None:
        self._catalog = catalog
        self._margin = margin
        self._matrix: NDArray[np.float32] | None = None
        self._slices: dict[Tier, slice] = {}

    @property
    def anchor_count(self) -> int:
        matrix = self._matrix
        if matrix is None:
            return 0
        return int(matrix.shape[0])

    def build(self, embedder: LocalVectorEngine) -> None:
        """Embed all 150 anchors. Call this on the ONNX thread."""
        texts: list[str] = []
        slices: dict[Tier, slice] = {}
        offset = 0
        for tier in _TIER_ORDER:
            rows = ANCHORS[tier]
            if len(rows) != 50:
                raise RuntimeError(f"{tier.value} anchors must contain 50 utterances")
            slices[tier] = slice(offset, offset + len(rows))
            texts.extend(rows)
            offset += len(rows)
        vectors = embedder.embed_batch_sync(texts)
        if vectors.shape != (offset, embedder.dim):
            raise RuntimeError("anchor matrix has an unexpected shape")
        self._matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        self._slices = slices

    async def build_async(self, embedder: LocalVectorEngine) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(embedder.executor, self.build, embedder)

    def classify(self, vector: NDArray[np.float32]) -> tuple[Tier, dict[Tier, float]]:
        """Per-tier max cosine, then the margin rule. NumPy matmul is the default."""
        matrix = self._matrix
        if matrix is None:
            raise RuntimeError("complexity router anchors are not built")
        flat = np.ascontiguousarray(vector, dtype=np.float32).reshape(-1)
        if flat.shape != (LocalVectorEngine.dim,):
            raise ValueError("query vector must be 384-dimensional")
        sims: NDArray[np.float32] = np.asarray(matrix @ flat, dtype=np.float32)
        scores: dict[Tier, float] = {}
        for tier, span in self._slices.items():
            scores[tier] = float(np.max(sims[span]))
        return select_tier(scores, margin=self._margin), scores

    def resolve(
        self,
        vector: NDArray[np.float32],
        *,
        message_count: int,
        affinity: AffinityRecord | None,
        open_providers: frozenset[str] = frozenset(),
    ) -> RouteDecision:
        classified, scores = self.classify(vector)
        return apply_affinity(
            classified,
            scores,
            message_count=message_count,
            affinity=affinity,
            catalog=self._catalog,
            open_providers=open_providers,
        )


def _ordered_providers(
    providers: tuple[str, ...],
    preferred: str | None,
    open_providers: frozenset[str],
) -> tuple[str, ...]:
    available = tuple(
        provider for provider in providers if provider not in open_providers
    )
    if not available:
        available = providers
    if preferred is None or preferred not in available:
        return available
    rest = tuple(provider for provider in available if provider != preferred)
    return (preferred, *rest)


def _model_for(
    models: tuple[ModelSpec, ...],
    provider: str,
    affinity: AffinityRecord | None,
) -> ModelSpec:
    if affinity is not None and affinity.provider == provider:
        for model in models:
            if model.name == affinity.model and model.provider == provider:
                return model
        fallback = next(
            (model for model in models if model.provider == provider), models[0]
        )
        return ModelSpec(
            name=affinity.model,
            provider=provider,
            tier=affinity.tier,
            price_in_per_m=fallback.price_in_per_m,
            price_out_per_m=fallback.price_out_per_m,
        )
    for model in models:
        if model.provider == provider:
            return model
    return models[0]
