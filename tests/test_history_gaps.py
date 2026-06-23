from datetime import date, timedelta
import math

import pytest

from volarb.adapters.breeze.history import (
    DAYS_PER_CALL,
    atm_strike_band,
    candles_to_frame,
    chunk_date_ranges,
    ist_window,
)


# ---------- chunk_date_ranges ----------
def test_chunk_contiguous_covering_nonoverlapping_bounded():
    """Chunks tile [start, end] exactly: cover endpoints, no gap, no overlap, span within cap."""
    for interval, span in DAYS_PER_CALL.items():
        s, e = date(2024, 1, 1), date(2026, 12, 31)
        ch = chunk_date_ranges(s, e, interval)
        assert ch[0][0] == s and ch[-1][1] == e
        for a, b in ch:
            assert a <= b and (b - a).days + 1 <= span
        for (_, b), (c, _) in zip(ch, ch[1:]):
            assert c == b + timedelta(days=1)  # day after previous end -> no gap/overlap


def test_chunk_exact_multiple_vs_partial_tail():
    # span=2, 4 days -> exactly 2 full chunks (boundary lands on a chunk edge)
    assert chunk_date_ranges(date(2026, 1, 1), date(2026, 1, 4), "1minute") == [
        (date(2026, 1, 1), date(2026, 1, 2)),
        (date(2026, 1, 3), date(2026, 1, 4)),
    ]
    # 5 days -> partial single-day tail chunk
    assert chunk_date_ranges(date(2026, 1, 1), date(2026, 1, 5), "1minute")[-1] == (
        date(2026, 1, 5),
        date(2026, 1, 5),
    )


def test_chunk_single_day():
    assert chunk_date_ranges(date(2026, 1, 1), date(2026, 1, 1), "1minute") == [
        (date(2026, 1, 1), date(2026, 1, 1))
    ]


def test_chunk_start_after_end_is_empty():
    assert chunk_date_ranges(date(2026, 1, 5), date(2026, 1, 1), "1minute") == []


def test_chunk_unsupported_interval_raises():
    with pytest.raises(ValueError):
        chunk_date_ranges(date(2026, 1, 1), date(2026, 1, 2), "15minute")


def test_chunk_span_stays_under_1000_candle_cap():
    # ~375 / 75 / 13 candles per NSE session for 1/5/30-minute bars
    for interval, per_session in {"1minute": 375, "5minute": 75, "30minute": 13}.items():
        assert DAYS_PER_CALL[interval] * per_session <= 1000


# ---------- atm_strike_band ----------
def test_atm_band_width_symmetry_uniform_step():
    b = atm_strike_band(24023, n=3, step=50)
    assert len(b) == 2 * 3 + 1 and b[3] == 24000  # ATM centered
    assert all(b[i + 1] - b[i] == 50 for i in range(len(b) - 1))
    assert b[0] == 24000 - 3 * 50 and b[-1] == 24000 + 3 * 50


def test_atm_band_n_zero_is_single_atm():
    assert atm_strike_band(24023, n=0) == [24000]


def test_atm_rounds_to_nearest_step_either_side_of_midpoint():
    assert atm_strike_band(24023, n=0) == [24000]  # below midpoint -> down
    assert atm_strike_band(24026, n=0) == [24050]  # above midpoint -> up


def test_atm_half_step_uses_bankers_rounding():
    # exact x.5*step: Python round() is round-half-to-EVEN, not half-up. Pin it.
    assert atm_strike_band(24025, n=0) == [24000]
    assert atm_strike_band(24075, n=0) == [24100]
    assert atm_strike_band(24125, n=0) == [24100]


def test_atm_custom_step():
    assert atm_strike_band(24049, n=1, step=100) == [23900, 24000, 24100]


# ---------- candles_to_frame edge cases ----------
def test_candles_dedup_preserves_stable_order():
    # adjacent chunk requests can repeat a boundary bar -> dedup-by-sort must be STABLE
    pl = {
        "Success": [
            {
                "datetime": "2026-02-03 09:20:00",
                "open": 1,
                "high": 1,
                "low": 1,
                "close": 1,
                "volume": 1,
            },
            {
                "datetime": "2026-02-03 09:20:00",
                "open": 2,
                "high": 2,
                "low": 2,
                "close": 2,
                "volume": 2,
            },
        ]
    }
    df = candles_to_frame(pl)
    assert len(df) == 2 and df["close"].tolist() == [1, 2]
    assert df.index.is_monotonic_increasing


def test_candles_data_key_fallback():
    pl = {
        "data": [
            {
                "datetime": "2026-02-03 09:20:00",
                "open": 1,
                "high": 1,
                "low": 1,
                "close": 1,
                "volume": 1,
            }
        ]
    }
    assert len(candles_to_frame(pl)) == 1


def test_candles_none_or_empty_payload_is_empty():
    assert candles_to_frame({"Success": None, "data": None}).empty
    assert candles_to_frame({}).empty


def test_candles_coerces_strings_and_bad_values_to_nan():
    pl = {
        "Success": [
            {
                "datetime": "2026-02-03 09:20:00",
                "open": "100.5",
                "high": "x",
                "low": "99",
                "close": "101",
                "volume": "5",
            }
        ]
    }
    df = candles_to_frame(pl)
    assert df["open"].iloc[0] == 100.5
    assert math.isnan(df["high"].iloc[0])  # unparseable -> NaN, not raise
    assert df["volume"].iloc[0] == 5


def test_candles_drops_metadata_and_canonical_oi_order():
    # OI key appears FIRST and metadata cols present -> output is canonical OHLCV+OI, metadata gone
    pl = {
        "Success": [
            {
                "open_interest": 5,
                "stock_code": "NIFTY",
                "datetime": "2026-02-03 09:20:00",
                "volume": 1,
                "close": 1,
                "low": 1,
                "high": 1,
                "open": 1,
                "right": "Call",
            }
        ]
    }
    assert list(candles_to_frame(pl).columns) == [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "open_interest",
    ]


def test_candles_missing_volume_column():
    pl = {
        "Success": [{"datetime": "2026-02-03 09:20:00", "open": 1, "high": 1, "low": 1, "close": 1}]
    }
    assert list(candles_to_frame(pl).columns) == ["open", "high", "low", "close"]


# ---------- ist_window ----------
def test_ist_window_custom_hours():
    f, t = ist_window(date(2026, 6, 19), "10:00", "14:00")
    assert f == "2026-06-19T10:00:00.000Z" and t == "2026-06-19T14:00:00.000Z"
