"""Tests for the Breeze rate-limit primitives + fetch-planning helpers (no live session needed)."""

from datetime import date

import pytest

from volarb.adapters.breeze.history import atm_strike_band, chunk_date_ranges
from volarb.adapters.breeze.ratelimit import (
    DailyBudgetExceeded,
    RateLimiter,
    classify_response,
)


def test_classify_response():
    assert classify_response({"Success": [{"x": 1}], "Status": 200}) == "ok"
    assert classify_response({"Success": [], "Status": 200}) == "empty"
    assert (
        classify_response({"Status": 5, "Error": "Limit exceed: API call per day: "}) == "rate_day"
    )
    assert (
        classify_response({"Status": 5, "Error": "Limit exceed: API call per minute"})
        == "rate_minute"
    )


def test_rate_limiter_paces_to_min_interval():
    slept = []
    t = [0.0]  # frozen clock -> 2nd call must wait the full interval
    rl = RateLimiter(min_interval=1.0, sleep=slept.append, clock=lambda: t[0])
    rl.acquire()  # first call: no wait
    rl.acquire()  # clock frozen -> sleeps ~1.0s
    assert slept and slept[-1] == pytest.approx(1.0)


def test_rate_limiter_daily_budget():
    rl = RateLimiter(min_interval=0.0, daily_budget=3, sleep=lambda s: None, clock=lambda: 0.0)
    for _ in range(3):
        rl.acquire()
    with pytest.raises(DailyBudgetExceeded):
        rl.acquire()
    assert rl.calls_made == 3


def test_chunk_date_ranges_1min_two_day_spans():
    chunks = chunk_date_ranges(date(2026, 1, 1), date(2026, 1, 10), "1minute")
    assert chunks[0] == (date(2026, 1, 1), date(2026, 1, 2))
    assert chunks[-1][1] == date(2026, 1, 10)
    assert len(chunks) == 5


def test_atm_strike_band():
    assert atm_strike_band(24717, n=2, step=50) == [24600, 24650, 24700, 24750, 24800]
