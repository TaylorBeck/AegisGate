"""Circuit breaker with consecutive-failure and rolling-window rules."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from enum import StrEnum

BreakerClock = Callable[[], float]


class BreakerState(StrEnum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """Opens after 3 consecutive failures or a 50% fail rate over 10+ calls / 30s.

    HALF_OPEN after 10s allows one probe. The probe closes or reopens the breaker.
    `clock` is injectable so tests can drive transitions without sleeping.
    """

    def __init__(
        self,
        name: str,
        *,
        clock: BreakerClock | None = None,
        on_transition: Callable[[str, BreakerState], None] | None = None,
        consecutive_limit: int = 3,
        window_seconds: float = 30.0,
        min_window_calls: int = 10,
        failure_rate: float = 0.5,
        open_seconds: float = 10.0,
    ) -> None:
        self.name = name
        self._clock = clock if clock is not None else time.monotonic
        self._on_transition = on_transition
        self._consecutive_limit = consecutive_limit
        self._window_seconds = window_seconds
        self._min_window_calls = min_window_calls
        self._failure_rate = failure_rate
        self._open_seconds = open_seconds
        self._state = BreakerState.CLOSED
        self._consecutive_failures = 0
        self._opened_at = 0.0
        self._probe_outstanding = False
        self._window: deque[tuple[float, bool]] = deque()

    @property
    def state(self) -> BreakerState:
        self._maybe_half_open()
        return self._state

    def allow(self) -> bool:
        """Return True when a call may be attempted."""
        self._maybe_half_open()
        if self._state is BreakerState.OPEN:
            return False
        if self._state is BreakerState.HALF_OPEN:
            if self._probe_outstanding:
                return False
            self._probe_outstanding = True
            return True
        return True

    def record_success(self) -> None:
        self._note(success=True)
        self._consecutive_failures = 0
        self._probe_outstanding = False
        if self._state is BreakerState.HALF_OPEN:
            self._window.clear()
            self._set(BreakerState.CLOSED)

    def record_failure(self) -> None:
        self._note(success=False)
        self._consecutive_failures += 1
        self._probe_outstanding = False
        if self._state is BreakerState.HALF_OPEN:
            self._set(BreakerState.OPEN)
            return
        if self._state is BreakerState.OPEN:
            return
        if self._consecutive_failures >= self._consecutive_limit:
            self._set(BreakerState.OPEN)
            return
        if self._window_tripped():
            self._set(BreakerState.OPEN)

    def _maybe_half_open(self) -> None:
        if self._state is not BreakerState.OPEN:
            return
        if self._clock() - self._opened_at >= self._open_seconds:
            self._set(BreakerState.HALF_OPEN)

    def _window_tripped(self) -> bool:
        now = self._clock()
        self._prune(now)
        total = len(self._window)
        if total < self._min_window_calls:
            return False
        failures = sum(1 for _ts, ok in self._window if not ok)
        return failures / total >= self._failure_rate

    def _note(self, *, success: bool) -> None:
        now = self._clock()
        self._window.append((now, success))
        self._prune(now)

    def _prune(self, now: float) -> None:
        cutoff = now - self._window_seconds
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()

    def _set(self, state: BreakerState) -> None:
        if state is self._state:
            return
        self._state = state
        if state is BreakerState.OPEN:
            self._opened_at = self._clock()
            self._probe_outstanding = False
        if state is BreakerState.HALF_OPEN:
            self._probe_outstanding = False
        if self._on_transition is not None:
            self._on_transition(self.name, state)


class BreakerSet:
    """Named breakers plus the set of currently OPEN providers."""

    def __init__(self, breakers: dict[str, CircuitBreaker]) -> None:
        self._breakers = breakers

    def get(self, name: str) -> CircuitBreaker | None:
        return self._breakers.get(name)

    def require(self, name: str) -> CircuitBreaker:
        breaker = self._breakers.get(name)
        if breaker is None:
            raise KeyError(name)
        return breaker

    def open_names(self) -> frozenset[str]:
        return frozenset(
            name
            for name, breaker in self._breakers.items()
            if breaker.state is BreakerState.OPEN
        )
