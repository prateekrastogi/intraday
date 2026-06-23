"""Breeze historical data -> normalized OHLCV(+OI) + fetch-planning helpers (the backtest-data half).

Pulls NIFTY-futures underlying + per-strike option OHLCV (incl. EXPIRED) via get_historical_data_v2.
Verified Breeze limits (2026): 100 calls/min, 5,000/day (GLOBAL); 1,000 candles/request -- so we
CHUNK date ranges per interval (chunk_date_ranges) and fetch only the ATM +/- N strike band
(atm_strike_band). Writing into the Nautilus ParquetDataCatalog comes when we wire the backtester.

TIMEZONE (verified on live data 2026-06): get_historical_data_v2 treats from_date/to_date as IST
wall-clock and IGNORES the trailing 'Z' -- pass IST literal times (use ist_window). A full NSE
session 09:15->15:30 returns 375 one-minute candles; sending UTC silently CLIPS the window.

Options rows additionally carry open_interest (index rows do not) -- candles_to_frame preserves it.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

_OHLC = ["open", "high", "low", "close", "volume"]
_OPTIONAL = ["open_interest"]  # present on options/futures rows, absent on the index
_NUMERIC = _OHLC + _OPTIONAL

# 1000-candle/request cap -> conservative calendar-days per call by interval (NSE ~375 min/session)
DAYS_PER_CALL = {"1minute": 2, "5minute": 13, "30minute": 76, "1day": 1000}


def candles_to_frame(payload: dict) -> pd.DataFrame:
    """Normalize a get_historical_data_v2 reply -> OHLCV(+open_interest) frame, datetime-indexed/sorted."""
    rows = payload.get("Success") or payload.get("data") or []
    if not rows:
        return pd.DataFrame(columns=_OHLC)
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    for c in _NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.set_index("datetime").sort_index()
    keep = [c for c in _OHLC if c in df.columns] + [c for c in _OPTIONAL if c in df.columns]
    return df[keep]


def ist_window(d: date, start_hm: str = "09:15", end_hm: str = "15:30") -> tuple[str, str]:
    """(from_date, to_date) ISO strings for an NSE session on `d`, as IST wall-clock.

    Breeze treats these as IST and ignores the 'Z'; 09:15->15:30 yields the full 375-candle session.
    """
    iso = d.isoformat()
    return (f"{iso}T{start_hm}:00.000Z", f"{iso}T{end_hm}:00.000Z")


def fetch_ohlc(
    bz,
    *,
    interval,
    from_date,
    to_date,
    stock_code,
    exchange_code,
    product_type="",
    expiry_date="",
    right="",
    strike_price="",
) -> pd.DataFrame:
    """Thin wrapper over BreezeConnect.get_historical_data_v2 -> normalized OHLCV(+OI) frame."""
    resp = bz.get_historical_data_v2(
        interval=interval,
        from_date=from_date,
        to_date=to_date,
        stock_code=stock_code,
        exchange_code=exchange_code,
        product_type=product_type,
        expiry_date=expiry_date,
        right=right,
        strike_price=strike_price,
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
