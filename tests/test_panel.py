"""Synthetic-parquet tests for the data panel (the lake -> session-aligned frames layer).

Writes tiny parquet files matching the lake layout into tmp_path; no real lake/creds/network.
"""

from __future__ import annotations

import datetime as dt
import math

import pandas as pd
import pytest

from volarb.data import panel


def _session_df(day: dt.date, close: float, n: int = 6, step: float = 0.0, oi: int = 1000):
    idx = pd.date_range(dt.datetime.combine(day, dt.time(9, 15)), periods=n, freq="1min")
    closes = [close + i * step for i in range(n)]
    return pd.DataFrame(
        {
            "datetime": idx,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": 100,
            "open_interest": oi,
        }
    )


def _write(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def test_trading_days_and_within_session_return(tmp_path):
    day = dt.date(2026, 6, 16)
    _write(tmp_path / "futures" / f"{day}.parquet", _session_df(day, 24000.0, n=6, step=2.0))
    assert panel.trading_days(tmp_path) == [day]
    u = panel.load_underlying(tmp_path, day)
    lr = u["log_ret"].to_list()
    assert lr[0] is None  # no return across the overnight gap / futures roll
    assert lr[1] is not None and lr[5] is not None
    assert u.height == panel.SESSION_MINUTES  # reindexed onto the full 375-min grid


def test_load_vix(tmp_path):
    day = dt.date(2026, 6, 16)
    _write(tmp_path / "vix" / f"{day}.parquet", _session_df(day, 13.5, n=6))
    v = panel.load_vix(tmp_path, day)
    assert "vix" in v.columns
    assert v.filter(v["vix"].is_not_null())["vix"].to_list()[0] == pytest.approx(13.5)


def _write_chain(tmp_path, day, expiry, forward=24000.0, premia=None):
    _write(tmp_path / "futures" / f"{day}.parquet", _session_df(day, forward, n=6))
    premia = premia or {23950: (118.0, 70.0), 24000: (95.0, 95.0), 24050: (74.0, 120.0)}
    for k, (ce, pe) in premia.items():
        base = tmp_path / "options" / f"expiry={expiry}"
        _write(base / "right=CE" / f"strike={k}" / f"{day}.parquet", _session_df(day, ce, n=6))
        _write(base / "right=PE" / f"strike={k}" / f"{day}.parquet", _session_df(day, pe, n=6))


def test_atm_iv_series_inverts_to_sane_iv(tmp_path):
    day, expiry = dt.date(2026, 6, 16), dt.date(2026, 6, 18)
    _write_chain(tmp_path, day, expiry)
    s = panel.atm_iv_series(tmp_path, day, expiry)
    assert s.height > 0 and "atm_iv" in s.columns
    vals = [x for x in s["atm_iv"].to_list() if x is not None and not math.isnan(x)]
    assert vals and all(0.01 < x < 1.0 for x in vals)


def test_term_slope_two_expiries(tmp_path):
    day, e1, e2 = dt.date(2026, 6, 16), dt.date(2026, 6, 18), dt.date(2026, 6, 25)
    _write_chain(tmp_path, day, e1)
    _write_chain(
        tmp_path,
        day,
        e2,
        premia={23950: (160.0, 110.0), 24000: (140.0, 140.0), 24050: (118.0, 165.0)},
    )
    slope = panel.term_slope(tmp_path, day, e1, e2)
    assert "slope" in slope.columns and slope.height > 0
