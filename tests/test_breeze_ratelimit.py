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


# --- Coverage additions from the independent verification pass ---------------------------------
# The original suite pins classify's 4 happy branches and frozen-clock full-interval pacing.
# Below we pin the pacing SUBTRACTION math, the wait<=0 / first-call no-sleep paths, the budget
# boundary edges, and every remaining classify branch (precedence, alt keys, case, None rows).


def _advancing_limiter(min_interval, *, daily_budget=4500):
    """RateLimiter on a controllable fake clock; returns (rl, slept, t) where t[0] is 'now'."""
    slept: list[float] = []
    t = [0.0]
    rl = RateLimiter(
        min_interval=min_interval,
        daily_budget=daily_budget,
        sleep=slept.append,
        clock=lambda: t[0],
    )
    return rl, slept, t


def test_acquire_first_call_never_sleeps():
    # _last is None on the very first call -> the pacing block is skipped entirely.
    rl, slept, _ = _advancing_limiter(1.0)
    rl.acquire()
    assert slept == []


def test_acquire_waits_only_the_remaining_interval():
    # 0.3s already elapsed externally -> must sleep exactly min_interval - 0.3 = 0.7s, not 1.0.
    rl, slept, t = _advancing_limiter(1.0)
    rl.acquire()  # _last := 0.0
    t[0] = 0.3
    rl.acquire()
    assert slept == [pytest.approx(0.7)]


def test_acquire_no_sleep_when_interval_already_elapsed():
    # wait = 1.0 - 5.0 < 0 -> the `if wait > 0` guard skips sleep entirely.
    rl, slept, t = _advancing_limiter(1.0)
    rl.acquire()
    t[0] = 5.0
    rl.acquire()
    assert slept == []


def test_acquire_exact_interval_boundary_no_sleep():
    # wait == 0 is NOT > 0 -> no sleep at the exact boundary (steady-state cadence).
    rl, slept, t = _advancing_limiter(1.0)
    rl.acquire()
    t[0] = 1.0
    rl.acquire()
    assert slept == []


def test_last_timestamp_recorded_after_sleep_so_cadence_is_steady():
    # _last is set from clock() AFTER sleeping; with an advancing fake clock each subsequent
    # frozen-between-calls acquire pays the full interval again (no drift / no double-charge).
    slept: list[float] = []
    t = [0.0]

    def sleeper(s):
        slept.append(s)
        t[0] += s  # a real sleep advances wall time

    rl = RateLimiter(min_interval=1.0, sleep=sleeper, clock=lambda: t[0])
    rl.acquire()  # t=0, no sleep, _last=0
    rl.acquire()  # frozen -> sleep 1.0, t=1, _last=1
    rl.acquire()  # frozen -> sleep 1.0, t=2, _last=2
    assert slept == [pytest.approx(1.0), pytest.approx(1.0)]


def test_budget_zero_raises_immediately():
    rl = RateLimiter(daily_budget=0, sleep=lambda s: None, clock=lambda: 0.0)
    with pytest.raises(DailyBudgetExceeded):
        rl.acquire()
    assert rl.calls_made == 0


def test_budget_exception_carries_calls_made():
    rl = RateLimiter(min_interval=0.0, daily_budget=2, sleep=lambda s: None, clock=lambda: 0.0)
    rl.acquire()
    rl.acquire()
    with pytest.raises(DailyBudgetExceeded) as exc:
        rl.acquire()
    assert exc.value.args == (2,)  # reports the spent count for the resume checkpoint


def test_budget_overflow_does_not_advance_counter():
    # A rejected acquire must not consume budget (so resume-next-day math stays exact).
    rl = RateLimiter(min_interval=0.0, daily_budget=1, sleep=lambda s: None, clock=lambda: 0.0)
    rl.acquire()
    for _ in range(3):
        with pytest.raises(DailyBudgetExceeded):
            rl.acquire()
    assert rl.calls_made == 1


def test_classify_status5_generic_defaults_to_rate_minute():
    # Status 5 + "limit exceed" but NO "per day" -> the per-minute default branch.
    assert classify_response({"Status": 5, "Error": "Limit exceed: blah"}) == "rate_minute"


def test_classify_rate_detected_from_error_without_status5():
    # "limit exceed" in the body alone (Status not 5) still classifies as a rate hit.
    assert classify_response({"Error": "Limit exceed: API call per day"}) == "rate_day"
    assert classify_response({"Error": "limit exceed soon"}) == "rate_minute"


def test_classify_rate_takes_precedence_over_present_rows():
    # A rate body that also carries stale Success rows must NOT be read as 'ok'.
    resp = {"Status": 5, "Success": [{"x": 1}], "Error": "Limit exceed per minute"}
    assert classify_response(resp) == "rate_minute"


def test_classify_is_case_insensitive():
    assert classify_response({"Status": 5, "Error": "LIMIT EXCEED: API CALL PER DAY"}) == "rate_day"
    assert classify_response({"Error": "Limit Exceed"}) == "rate_minute"


def test_classify_reads_alternate_data_key():
    # Rows may arrive under 'data' instead of 'Success'.
    assert classify_response({"data": [{"a": 1}], "Status": 200}) == "ok"


def test_classify_ok_overrides_non200_status_when_rows_present():
    # Rows present short-circuit to 'ok' before the status check is consulted.
    assert classify_response({"Status": 500, "Success": [{"a": 1}]}) == "ok"


def test_classify_error_branch_non200_empty():
    # Non-rate, non-200, no rows -> 'error' (server fault, not just "no data").
    assert classify_response({"Status": 500}) == "error"
    assert classify_response({"Status": 503, "Success": []}) == "error"


def test_classify_empty_when_status_none_or_200():
    # 200-with-no-rows and status-absent both mean "no data" (retryable), not error.
    assert classify_response({"Success": [], "Status": 200}) == "empty"
    assert classify_response({}) == "empty"


def test_classify_none_rows_do_not_crash():
    # Success/data explicitly None must hit the `or []` guard, not blow up on truthiness.
    assert classify_response({"Success": None, "Status": 200}) == "empty"
    assert classify_response({"data": None, "Status": 200}) == "empty"
