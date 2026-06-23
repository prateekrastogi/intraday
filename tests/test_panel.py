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


# ----------------------------------------------------------------------------
# Gap-filling coverage from the independent math-verification pass (panel module).
# These pin the LOAD-BEARING panel properties the originals only sanity-checked:
# exact ATM-IV round-trip, term-slope SIGN, holiday/empty-day modes, OI-vs-price
# ffill on the full 375 grid, and missing-strike-file robustness. Synthetic only.
# ----------------------------------------------------------------------------

import math as _math  # noqa: E402

from volarb.pricing.greeks import DEFAULT_R, years_to_expiry  # noqa: E402


def _b76(F, K, T, sigma, r, is_call):
    """Black-76 forward-option price (b=0, s=forward) -- the convention greeks.py inverts."""
    df = _math.exp(-r * T)
    d1 = (_math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * _math.sqrt(T))
    d2 = d1 - sigma * _math.sqrt(T)
    ncdf = lambda x: 0.5 * (1.0 + _math.erf(x / _math.sqrt(2)))  # noqa: E731
    if is_call:
        return df * (F * ncdf(d1) - K * ncdf(d2))
    return df * (K * ncdf(-d2) - F * ncdf(-d1))


def _write_planted_chain(tmp_path, day, expiry, forward, sigma, strikes, r=DEFAULT_R, n=6):
    """Write a futures session + a CE/PE chain whose closes are Black-76 prices of a
    KNOWN flat vol ``sigma`` (per-minute, at the true minutes-to-15:30). Lets the test
    invert the panel's ATM-IV back to ``sigma`` exactly."""
    idx = pd.date_range(dt.datetime.combine(day, dt.time(9, 15)), periods=n, freq="1min")
    _write(
        tmp_path / "futures" / f"{day}.parquet",
        pd.DataFrame(
            {
                "datetime": idx,
                "open": forward,
                "high": forward,
                "low": forward,
                "close": forward,
                "volume": 100,
                "open_interest": 1000,
            }
        ),
    )
    expc = dt.datetime.combine(expiry, dt.time(15, 30))
    base = tmp_path / "options" / f"expiry={expiry}"
    for K in strikes:
        for right, is_call in (("CE", True), ("PE", False)):
            closes = []
            for m in range(n):
                t = dt.datetime.combine(day, dt.time(9, 15)) + dt.timedelta(minutes=m)
                mins = (expc - t).total_seconds() / 60.0
                closes.append(_b76(forward, K, years_to_expiry(mins), sigma, r, is_call))
            _write(
                base / f"right={right}" / f"strike={K}" / f"{day}.parquet",
                pd.DataFrame(
                    {
                        "datetime": idx,
                        "open": closes,
                        "high": closes,
                        "low": closes,
                        "close": closes,
                        "volume": 100,
                        "open_interest": 1000,
                    }
                ),
            )


def test_atm_iv_round_trips_to_planted_vol(tmp_path):
    """The inversion is exact, not merely "in (0.01,1.0)": planting sigma=0.18 and pricing
    the chain at that vol must recover ~0.18 every minute (vega-weighted single-vol invert)."""
    day, expiry = dt.date(2026, 6, 16), dt.date(2026, 6, 18)
    _write_planted_chain(tmp_path, day, expiry, 24000.0, 0.18, [23950, 24000, 24050])
    s = panel.atm_iv_series(tmp_path, day, expiry)
    vals = [x for x in s["atm_iv"].to_list() if x is not None]
    assert vals
    for v in vals:
        assert v == pytest.approx(0.18, abs=1e-3)  # round-trip, not just sane-range


def test_term_slope_sign_inverted_curve_is_negative(tmp_path):
    """Docstring/Johnson-2017: negative slope == inverted (front IV > next). Plant front
    vol 0.30 > next 0.18 and assert EVERY minute's slope ~ (0.18-0.30) = -0.12 < 0.
    Original test only checked height>0 -- the sign is the whole signal."""
    day, e1, e2 = dt.date(2026, 6, 16), dt.date(2026, 6, 18), dt.date(2026, 6, 25)
    _write_planted_chain(tmp_path, day, e1, 24000.0, 0.30, [23950, 24000, 24050])
    _write_planted_chain(tmp_path, day, e2, 24000.0, 0.18, [23950, 24000, 24050])
    slopes = panel.term_slope(tmp_path, day, e1, e2)["slope"].to_list()
    assert slopes
    for sl in slopes:
        assert sl < 0.0
        assert sl == pytest.approx(-0.12, abs=2e-3)


def test_term_slope_sign_contango_is_positive(tmp_path):
    """Mirror: next IV > front -> positive slope (normal/contango term structure)."""
    day, e1, e2 = dt.date(2026, 6, 16), dt.date(2026, 6, 18), dt.date(2026, 6, 25)
    _write_planted_chain(tmp_path, day, e1, 24000.0, 0.16, [23950, 24000, 24050])
    _write_planted_chain(tmp_path, day, e2, 24000.0, 0.22, [23950, 24000, 24050])
    slopes = panel.term_slope(tmp_path, day, e1, e2)["slope"].to_list()
    assert slopes and all(sl > 0.0 for sl in slopes)
    assert slopes[0] == pytest.approx(0.06, abs=2e-3)


def test_holiday_missing_futures_file_raises(tmp_path):
    """A day with NO futures parquet (holiday / not-yet-backfilled) is a hard error in
    load_underlying/atm_iv_series, NOT a silent empty frame -- pin that contract so a
    caller knows to gate on trading_days() first."""
    day, expiry = dt.date(2026, 6, 16), dt.date(2026, 6, 18)
    with pytest.raises(FileNotFoundError):
        panel.load_underlying(tmp_path, day)
    with pytest.raises(FileNotFoundError):
        panel.atm_iv_series(tmp_path, day, expiry)


def test_empty_option_chain_returns_typed_empty_frame(tmp_path):
    """Futures present but the expiry has NO option files: atm_iv_series degrades to an
    empty frame with the right schema (not a crash), and term_slope's inner-join is empty."""
    day, e1, e2 = dt.date(2026, 6, 16), dt.date(2026, 6, 18), dt.date(2026, 6, 25)
    _write(tmp_path / "futures" / f"{day}.parquet", _session_df(day, 24000.0, n=6))
    s = panel.atm_iv_series(tmp_path, day, e1)
    assert s.height == 0 and s.columns == ["datetime", "atm_iv"]
    # e1 priced, e2 absent -> inner join yields zero rows, still typed.
    _write_chain(tmp_path, day, e1)
    sl = panel.term_slope(tmp_path, day, e1, e2)
    assert sl.height == 0 and sl.columns == ["datetime", "slope"]


def test_missing_one_strike_file_still_prices(tmp_path):
    """Robustness: if a single strike's parquet is absent, the minute still prices off the
    remaining >=2 quotes rather than dropping the whole minute or erroring."""
    day, expiry = dt.date(2026, 6, 16), dt.date(2026, 6, 18)
    _write_chain(tmp_path, day, expiry)
    missing = (
        tmp_path / "options" / f"expiry={expiry}" / "right=CE" / "strike=24000" / f"{day}.parquet"
    )
    missing.unlink()
    s = panel.atm_iv_series(tmp_path, day, expiry)
    vals = [x for x in s["atm_iv"].to_list() if x is not None and not _math.isnan(x)]
    assert vals and all(0.01 < x < 1.0 for x in vals)


def test_full_grid_price_null_but_oi_ffilled(tmp_path):
    """Beyond the 6 written bars the canonical 375-grid tail must keep prices NULL (no
    blanket ffill -> no fabricated zero returns / RV bias) while OI ffills (stock, not flow)."""
    day = dt.date(2026, 6, 16)
    idx = pd.date_range(dt.datetime.combine(day, dt.time(9, 15)), periods=6, freq="1min")
    df = pd.DataFrame(
        {
            "datetime": idx,
            "open": 24000.0,
            "high": 24000.0,
            "low": 24000.0,
            "close": 24000.0,
            "volume": 100,
            "open_interest": [1000, 1001, 1002, 1003, 1004, 1005],
        }
    )
    _write(tmp_path / "futures" / f"{day}.parquet", df)
    u = panel.load_underlying(tmp_path, day)
    assert u.height == panel.SESSION_MINUTES
    close, lr, oi = u["close"].to_list(), u["log_ret"].to_list(), u["open_interest"].to_list()
    assert close[100] is None and lr[100] is None  # tail: no fabricated price/return
    assert oi[100] == 1005  # OI carried forward from the last real bar
    assert oi[5] == 1005 and oi[0] == 1000  # within the real bars OI is exact


def test_load_vix_reindexed_to_grid_not_ffilled(tmp_path):
    """VIX is reindexed onto the 375-grid and left null past the written bars (it is a
    price-like level, ffilling would fabricate a flat tail)."""
    day = dt.date(2026, 6, 16)
    _write(tmp_path / "vix" / f"{day}.parquet", _session_df(day, 13.5, n=6))
    v = panel.load_vix(tmp_path, day)
    assert v.height == panel.SESSION_MINUTES and v.columns == ["datetime", "vix"]
    vix = v["vix"].to_list()
    assert vix[0] == pytest.approx(13.5)
    assert vix[100] is None  # not ffilled past the real bars
