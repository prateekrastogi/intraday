"""Unit tests for the Breeze data layer helpers that need NO live session (pure functions)."""
from volarb.adapters.breeze.auth import login_url
from volarb.adapters.breeze.history import candles_to_frame


def test_login_url_encodes_key():
    url = login_url("abc+def=ghi")
    assert url.startswith("https://api.icicidirect.com/apiuser/login?api_key=")
    assert "abc%2Bdef%3Dghi" in url            # +, = must be URL-encoded


def test_candles_to_frame_normalizes_and_sorts():
    payload = {"Success": [
        {"datetime": "2026-02-03 09:21:00", "open": 100.5, "high": 102.0, "low": 100.0, "close": 101.5, "volume": 900},
        {"datetime": "2026-02-03 09:20:00", "open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5, "volume": 1200},
    ], "Status": 200, "Error": None}
    df = candles_to_frame(payload)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2
    assert df.index.is_monotonic_increasing    # sorted ascending by datetime
    assert df.iloc[0]["close"] == 100.5        # the 09:20 row sorts first


def test_candles_to_frame_empty():
    assert candles_to_frame({"Success": []}).empty
