"""Nanosecond stage timer for the hot-path latency breakdown."""

from __future__ import annotations

import time

from app.schemas.finops import LatencyBreakdown


class StageTimer:
    """Records stage durations with `perf_counter_ns` and builds a breakdown.

    Stages that never start stay null so a cache hit can omit embed, HNSW, and routing.
    """

    def __init__(self) -> None:
        self._elapsed: dict[str, float] = {}
        self._open: dict[str, int] = {}

    def start(self, stage: str) -> None:
        self._open[stage] = time.perf_counter_ns()

    def stop(self, stage: str) -> float:
        started = self._open.pop(stage)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        self._elapsed[stage] = elapsed_ms
        return elapsed_ms

    def breakdown(self) -> LatencyBreakdown:
        return LatencyBreakdown(
            guardrail=self._elapsed.get("guardrail", 0.0),
            l1_lookup=self._elapsed.get("l1_lookup", 0.0),
            onnx_embed=self._elapsed.get("onnx_embed"),
            l2_hnsw=self._elapsed.get("l2_hnsw"),
            router=self._elapsed.get("router"),
        )
