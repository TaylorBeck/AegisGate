"""Counterfactual FinOps cost math.

actual_usd = tokens_in/1e6 * actual.price_in + tokens_out/1e6 * actual.price_out
baseline_usd uses the frontier catalog model.
saved_usd = max(0, baseline_usd - actual_usd)
Cache hits force actual_usd to 0 and keep the stored token counts.
"""

from __future__ import annotations

from app.catalog import ModelSpec


def compute_costs(
    tokens_in: int,
    tokens_out: int,
    actual: ModelSpec | None,
    baseline: ModelSpec,
    *,
    cache_hit: bool = False,
) -> tuple[float, float]:
    """Return `(actual_usd, saved_usd)`."""
    baseline_usd = _cost(tokens_in, tokens_out, baseline)
    if cache_hit or actual is None:
        return 0.0, baseline_usd
    actual_usd = _cost(tokens_in, tokens_out, actual)
    saved = baseline_usd - actual_usd
    if saved < 0.0:
        saved = 0.0
    return actual_usd, saved


def _cost(tokens_in: int, tokens_out: int, spec: ModelSpec) -> float:
    return (tokens_in / 1_000_000.0) * spec.price_in_per_m + (
        tokens_out / 1_000_000.0
    ) * spec.price_out_per_m
