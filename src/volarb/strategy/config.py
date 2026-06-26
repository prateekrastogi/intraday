"""Instrument configs for the directional ITM opening-range-breakout strategy.

ONE strategy (strategy/orb.py), parametrised per instrument so the same code runs NIFTY-weekly equity options
and MCX Crude commodity options. The opening-range window is scaled to ~8% of the session (NIFTY 30/375;
Crude 70/~895), matching the fraction that works on equities -- a longer OR on the 14.9h MCX session avoids
the over-firing a fixed 30-min window causes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time


@dataclass(frozen=True)
class Inst:
    name: str
    root: str               # marketdata lake: {root}/underlying/{date}.parquet + {root}/options/expiry=*/right=*/strike=*/{date}.parquet
    session_start: time
    session_end: time
    or_end: time            # opening-range window end (~session_start + 8% of session)
    or_minutes: int         # positional OR fallback for files without a datetime column (e.g. nifty3y)
    min_bars: int           # skip thin/illiquid days
    step: int               # option strike interval
    expiry: str             # "weekly" | "monthly"
    moneyness: float        # ITM fraction (~delta 0.65); 0.014 = 1.4% in-the-money
    hold: int               # hold horizon in TRADING days
    stop: float             # intraday stop fraction of premium
    spread: float           # round-trip cost charged per trade (fraction of premium)


# ~8% opening-range fraction; ITM delta~0.65; cut-loss-fast -12%; ride 5 trading days (the validated config).
NIFTY_WEEKLY = Inst(
    name="NIFTY-weekly", root="/home/ubuntu/marketdata/nifty3y",
    session_start=time(9, 15), session_end=time(15, 29), or_end=time(9, 45), or_minutes=30,
    min_bars=60, step=50, expiry="weekly", moneyness=0.014, hold=5, stop=0.12, spread=0.01,
)
CRUDE = Inst(
    name="CRUDE", root="/home/ubuntu/marketdata/commodity/CRUDE",
    session_start=time(9, 0), session_end=time(23, 54), or_end=time(10, 10), or_minutes=70,
    min_bars=120, step=50, expiry="monthly", moneyness=0.014, hold=5, stop=0.12, spread=0.01,
)

INSTRUMENTS = {i.name: i for i in (NIFTY_WEEKLY, CRUDE)}
