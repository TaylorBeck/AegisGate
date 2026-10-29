"""Circuit breaker transitions under an injectable clock."""

from __future__ import annotations

from app.providers.breaker import BreakerState, CircuitBreaker


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def test_three_consecutive_failures_open() -> None:
    clock = FakeClock()
    seen: list[BreakerState] = []
    breaker = CircuitBreaker(
        "p", clock=clock, on_transition=lambda _n, s: seen.append(s)
    )
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state is BreakerState.CLOSED
    breaker.record_failure()
    assert breaker.state is BreakerState.OPEN
    assert seen == [BreakerState.OPEN]
    assert breaker.allow() is False


def test_success_resets_consecutive_failures() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker("p", clock=clock)
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state is BreakerState.CLOSED


def test_window_failure_rate_opens() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker("p", clock=clock)
    # S F repeated so consecutive failures stay at 1. The 10th call is a
    # failure that brings the window to 50% of 10.
    for index in range(10):
        if index % 2 == 0:
            breaker.record_success()
        else:
            breaker.record_failure()
        clock.advance(0.2)
    assert breaker.state is BreakerState.OPEN


def test_half_open_probe_success_closes() -> None:
    clock = FakeClock()
    seen: list[BreakerState] = []
    breaker = CircuitBreaker(
        "p", clock=clock, on_transition=lambda _n, s: seen.append(s)
    )
    for _ in range(3):
        breaker.record_failure()
    assert breaker.state is BreakerState.OPEN
    clock.advance(10.0)
    assert breaker.state is BreakerState.HALF_OPEN
    assert breaker.allow() is True
    assert breaker.allow() is False
    breaker.record_success()
    assert breaker.state is BreakerState.CLOSED
    assert BreakerState.HALF_OPEN in seen
    assert seen[-1] is BreakerState.CLOSED


def test_half_open_probe_failure_reopens() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker("p", clock=clock)
    for _ in range(3):
        breaker.record_failure()
    clock.advance(10.0)
    assert breaker.allow() is True
    breaker.record_failure()
    assert breaker.state is BreakerState.OPEN
    assert breaker.allow() is False
