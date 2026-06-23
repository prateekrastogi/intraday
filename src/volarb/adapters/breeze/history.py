"""Breeze historical data -> normalized OHLCV + fetch-planning helpers (the backtest-data half).

Pulls NIFTY-futures underlying + per-strike option OHLCV (incl. EXPIRED) via get_historical_data_v2.
Verified Breeze limits (2026): 100 calls/min, 5,000/day (GLOBAL); 1,000 candles/request -- so we
CHUNK date ranges per interval (chunk_date_ranges) and fetch only the ATM +/- N strike band
(atm_strike_band). Writing into the Nautilus ParquetDataCatalog comes when we wire the backtester.

NOTE: validate candles_to_frame() against a real get_historical_data_v2 reply once a session exists.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

_OHLC = ["open", "high", "low", "close", "volume"]

# 1000-candle/request cap -> conservative calendar-days per call by interval (NSE ~375 min/session)
DAYS_PER_CALL = {"1minute": 2, "5minute": 13, "30minute": 76, "1day": 1000}


def candles_to_frame(payload: dict) -> pd.DataFrame:
    """Normalize a get_historical_data_v2 response -> OHLCV DataFrame indexed by datetime (sorted)."""
    rows = payload.get("Success") or payload.get("data") or []
    if not rows:
        return pd.DataFrame(columns=_OHLC)
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    for c in _OHLC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.set_index("datetime").sort_index()
    return df[[c for c in _OHLC if c in df.columns]]


def fetch_ohlc(bz, *, interval, from_date, to_date, stock_code, exchange_code,
               product_type="", expiry_date="", right="", strike_price="") -> pd.DataFrame:
    """Thin wrapper over BreezeConnect.get_historical_data_v2 -> normalized OHLCV frame."""
    resp = bz.get_historical_data_v2(
        interval=interval, from_date=from_date, to_date=to_date,
        stock_code=stock_code, exchange_code=exchange_code, product_type=product_type,
        expiry_date=expiry_date, right=right, strike_price=strike_price,
    )
    return candles_to_frame(resp)


def chunk_date_ranges(start: date, end: date, interval: str) -> list[tuple[date, date]]:
    """Split [start, end] into chunks that stay under the 1000-candle/request cap for `interval`."""
    span = DAYS_PER_CALL.get(interval)
    if span is None:
        raise ValueError(f"unsupported interval for chunking: {interval}")
    out, cur = [], start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=span - 1), end)
        out.append((cur, chunk_end))
        cur = chunk_end + timedelta(days=1)
    return out


def atm_strike_band(spot: float, n: int = 3, step: int = 50) -> list[int]:
    """ATM +/- n strikes at `step` spacing -- the minimal universe for a straddle/strangle."""
    atm = round(spot / step) * step
    return [atm + i * step for i in range(-n, n + 1)]
