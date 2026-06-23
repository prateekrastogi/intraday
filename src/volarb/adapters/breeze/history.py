"""Breeze historical data -> normalized OHLCV (the backtest-data half of the data layer).

Pulls NIFTY-futures underlying + per-strike option OHLCV (incl. EXPIRED) via get_historical_data_v2,
returning tidy DataFrames. Writing these into the Nautilus ParquetDataCatalog is added when we wire
the backtester (Step 5); the live LiveMarketDataClient is a separate piece (data.py).

NOTE: the exact response keys below are the documented shape; VALIDATE candles_to_frame() against a
real get_historical_data_v2 response once a Breeze session is available, then lock it with a sample.
For options: exchange_code="NFO", product_type="options", stock_code="NIFTY", right="call"/"put",
expiry_date + strike_price set; for the index/future use the appropriate exchange/product codes.
"""
from __future__ import annotations

import pandas as pd

_OHLC = ["open", "high", "low", "close", "volume"]


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
