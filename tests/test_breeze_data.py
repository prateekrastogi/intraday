"""Unit tests for the Breeze data layer helpers that need NO live session (pure functions)."""

from volarb.adapters.breeze.auth import login_url
from volarb.adapters.breeze.history import candles_to_frame


def test_login_url_encodes_key():
    url = login_url("abc+def=ghi")
    assert url.startswith("https://api.icicidirect.com/apiuser/login?api_key=")
    assert "abc%2Bdef%3Dghi" in url  # +, = must be URL-encoded


def test_candles_to_frame_normalizes_and_sorts():
    payload = {
        "Success": [
            {
                "datetime": "2026-02-03 09:21:00",
                "open": 100.5,
                "high": 102.0,
                "low": 100.0,
                "close": 101.5,
                "volume": 900,
            },
            {
                "datetime": "2026-02-03 09:20:00",
                "open": 100.0,
                "high": 101.0,
                "low": 99.5,
                "close": 100.5,
                "volume": 1200,
            },
        ],
        "Status": 200,
        "Error": None,
    }
    df = candles_to_frame(payload)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2
    assert df.index.is_monotonic_increasing  # sorted ascending by datetime
    assert df.iloc[0]["close"] == 100.5  # the 09:20 row sorts first


def test_candles_to_frame_empty():
    assert candles_to_frame({"Success": []}).empty


def test_candles_to_frame_preserves_open_interest_for_options():
    # real options-row shape (verified on live Breeze 2026-06): carries open_interest + extra metadata
    payload = {
        "Success": [
            {
                "datetime": "2026-06-22 09:16:00",
                "open": 145.6,
                "high": 157.9,
                "low": 142.75,
                "close": 157.9,
                "volume": 2895555,
                "open_interest": 7138365,
                "stock_code": "NIFTY",
                "right": "Call",
                "strike_price": 24000.0,
                "expiry_date": "23-JUN-2026",
            },
            {
                "datetime": "2026-06-22 09:15:00",
                "open": 147.35,
                "high": 147.35,
                "low": 130.2,
                "close": 142.25,
                "volume": 3101475,
                "open_interest": 8956025,
                "stock_code": "NIFTY",
                "right": "Call",
                "strike_price": 24000.0,
                "expiry_date": "23-JUN-2026",
            },
        ],
        "Status": 200,
        "Error": None,
    }
    df = candles_to_frame(payload)
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "open_interest"]
    assert df.index.is_monotonic_increasing
    assert df.iloc[0]["open_interest"] == 8956025  # the 09:15 row sorts first, OI kept as number
    assert df.iloc[0]["close"] == 142.25


def test_ist_window_builds_full_session():
    from datetime import date
    from volarb.adapters.breeze.history import ist_window

    f, t = ist_window(date(2026, 6, 19))
    assert f == "2026-06-19T09:15:00.000Z"  # IST literal (Breeze ignores the Z)
    assert t == "2026-06-19T15:30:00.000Z"
