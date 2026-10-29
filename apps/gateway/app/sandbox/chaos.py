"""Fault injection for the sandbox mock providers only."""

from __future__ import annotations


class ChaosController:
    """Inject 429/503 responses (and optional stalls) into named mock providers."""

    def __init__(self) -> None:
        self._faults: dict[str, tuple[int, int]] = {}
        self._stalls: dict[str, tuple[int, float]] = {}

    def inject(self, provider: str, status: int, count: int) -> None:
        if count <= 0:
            self._faults.pop(provider, None)
            return
        self._faults[provider] = (status, count)

    def stall(self, provider: str, after_chunks: int, sleep_s: float) -> None:
        self._stalls[provider] = (after_chunks, sleep_s)

    def clear(self, provider: str | None = None) -> None:
        if provider is None:
            self._faults.clear()
            self._stalls.clear()
            return
        self._faults.pop(provider, None)
        self._stalls.pop(provider, None)

    def fault_for(self, provider: str) -> int | None:
        item = self._faults.get(provider)
        if item is None:
            return None
        status, count = item
        if count <= 1:
            self._faults.pop(provider, None)
        else:
            self._faults[provider] = (status, count - 1)
        return status

    def stall_for(self, provider: str) -> tuple[int, float] | None:
        return self._stalls.get(provider)
