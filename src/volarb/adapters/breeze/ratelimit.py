"""Breeze REST rate-limit primitives (verified 2026: 100 calls/min, 5,000/day, GLOBAL counter).

Pace WELL under the cap -- real enforcement is stricter/buggier than published (a user hit the daily
error at ~170 calls). Steady pacing = 1.0s/call (~60/min); stop at ~4,500/day for headroom.
Rate rejection is NOT HTTP 429 -- it is a JSON body {"Status": 5, "Error": "Limit exceed: ..."}.
"""
from __future__ import annotations

import time

DEFAULT_MIN_INTERVAL = 1.0    # seconds between calls (~60/min, under the 100/min cap)
DEFAULT_DAILY_BUDGET = 4500   # stop here (10% under the 5,000/day cap)


class DailyBudgetExceeded(Exception):
    """Raised when the per-day call budget is spent -> checkpoint and resume next day."""


class RateLimiter:
    """Min-interval pacing + a per-day call budget. sleep/clock are injectable for testing."""

    def __init__(self, min_interval=DEFAULT_MIN_INTERVAL, daily_budget=DEFAULT_DAILY_BUDGET,
                 sleep=time.sleep, clock=time.monotonic):
        self.min_interval = min_interval
        self.daily_budget = daily_budget
        self._sleep = sleep
        self._clock = clock
        self._last = None
        self.calls_made = 0

    def acquire(self) -> None:
        """Pace one call; raise DailyBudgetExceeded once the daily budget is spent."""
        if self.calls_made >= self.daily_budget:
            raise DailyBudgetExceeded(self.calls_made)
        if self._last is not None:
            wait = self.min_interval - (self._clock() - self._last)
            if wait > 0:
                self._sleep(wait)
        self._last = self._clock()
        self.calls_made += 1


def classify_response(resp: dict) -> str:
    """Classify a Breeze reply: 'ok' | 'empty' | 'rate_minute' | 'rate_day' | 'error'.

    Rate limits arrive in the BODY (Status 5 / "Limit exceed"), not as HTTP 429. An empty Success
    can be a transient glitch -> retry a couple times before accepting "no data".
    """
    status = resp.get("Status")
    err = (resp.get("Error") or "")
    if status == 5 or "limit exceed" in err.lower():
        return "rate_day" if "per day" in err.lower() else "rate_minute"
    rows = resp.get("Success") or resp.get("data") or []
    if rows:
        return "ok"
    if status in (200, None):
        return "empty"
    return "error"
