"""NSE options fee + charge model (IMPLEMENTATION.md sec.4) -- pure, deterministic, golden-tested.

The core math lives here so backtest, paper, and live all charge identically; a Nautilus FeeModel
wraps this at integration. Rates are DATE-AWARE (STT on options stepped up Oct-2024 and Apr-2026).
VERIFY against a real broker contract note before trusting live P&L -- statutory rates change.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

# Rates as a fraction of premium turnover unless noted. VERIFY vs current NSE/SEBI circulars.
EXCHANGE_TXN_RATE = 0.0003553  # NSE F&O options exchange txn charge (~0.03553% of premium)
SEBI_RATE = 0.000001  # SEBI turnover fee 0.0001% (Rs 10 / crore)
STAMP_BUY_RATE = 0.00003  # stamp duty 0.003% on the BUY premium only
GST_RATE = 0.18  # 18% GST on (brokerage + exchange + SEBI)
STT_EXERCISE_RATE = 0.00125  # STT 0.125% of intrinsic on ITM exercise
DEFAULT_BROKERAGE = 20.0  # Rs per executed order (flat-fee discount broker)


def stt_sell_rate(d: date) -> float:
    """STT on the SELL of options premium -- date-aware (it stepped up over time)."""
    if d >= date(2026, 4, 1):
        return 0.0015  # 0.15% from 1 Apr 2026
    if d >= date(2024, 10, 1):
        return 0.0010  # 0.10% from 1 Oct 2024
    return 0.000625  # 0.0625% before 1 Oct 2024


@dataclass(frozen=True)
class Charges:
    """Per-order NSE charge breakdown, in rupees."""

    stt: float
    exchange: float
    sebi: float
    stamp: float
    brokerage: float
    gst: float

    @property
    def total(self) -> float:
        return self.stt + self.exchange + self.sebi + self.stamp + self.brokerage + self.gst


def option_order_charges(
    side: str, premium: float, qty: int, trade_date: date, brokerage: float = DEFAULT_BROKERAGE
) -> Charges:
    """All NSE charges (Rs) for ONE option order.

    side       : "BUY" or "SELL"
    premium    : option price per unit
    qty        : total units = lots * lot_size
    trade_date : used for the date-aware STT rate
    """
    side = side.upper()
    if side not in ("BUY", "SELL"):
        raise ValueError("side must be BUY or SELL")
    turnover = premium * qty
    stt = turnover * stt_sell_rate(trade_date) if side == "SELL" else 0.0
    exch = turnover * EXCHANGE_TXN_RATE
    sebi = turnover * SEBI_RATE
    stamp = turnover * STAMP_BUY_RATE if side == "BUY" else 0.0
    gst = (brokerage + exch + sebi) * GST_RATE
    return Charges(stt=stt, exchange=exch, sebi=sebi, stamp=stamp, brokerage=brokerage, gst=gst)


def round_trip_charges(
    premium_in: float,
    premium_out: float,
    qty: int,
    trade_date: date,
    brokerage: float = DEFAULT_BROKERAGE,
) -> Charges:
    """Entry BUY + exit SELL of one leg (the long-vol round trip)."""
    b = option_order_charges("BUY", premium_in, qty, trade_date, brokerage)
    s = option_order_charges("SELL", premium_out, qty, trade_date, brokerage)
    return Charges(
        stt=b.stt + s.stt,
        exchange=b.exchange + s.exchange,
        sebi=b.sebi + s.sebi,
        stamp=b.stamp + s.stamp,
        brokerage=b.brokerage + s.brokerage,
        gst=b.gst + s.gst,
    )


def exercise_stt(intrinsic: float, qty: int) -> float:
    """STT on ITM option exercise = 0.125% of intrinsic value. Square off to avoid it."""
    return intrinsic * qty * STT_EXERCISE_RATE


def structure_round_trip_charges(
    legs,
    qty: int,
    trade_date: date,
    brokerage: float = DEFAULT_BROKERAGE,
) -> Charges:
    """Round-trip charges for a MULTI-LEG structure (straddle / strangle): sum of per-leg round trips.

    ``legs`` is an iterable of ``(premium_in, premium_out)`` (uses scalar ``qty``) OR ``(premium_in, premium_out, leg_qty)`` (per-leg qty, for unequal-ratio backspreads), one per leg -- a NIFTY straddle is
    ``[(call_in, call_out), (put_in, put_out)]``, i.e. 4 executed orders total, so brokerage is
    ``4 * DEFAULT_BROKERAGE`` (Rs80) and every leg pays its own STT / exchange / SEBI / GST on its
    own premium. Use this (not the single-leg ``round_trip_charges``) for the strategy's true
    structure cost -- otherwise the backtest under-charges a straddle by ~half.
    """
    legs = list(legs)
    if not legs:
        raise ValueError("a structure needs at least one leg")
    per_leg = []
    for leg in legs:
        leg_qty = leg[2] if len(leg) > 2 else qty  # (premium_in, premium_out[, leg_qty]); unequal ratio legs
        per_leg.append(round_trip_charges(leg[0], leg[1], leg_qty, trade_date, brokerage))
    return Charges(
        stt=sum(c.stt for c in per_leg),
        exchange=sum(c.exchange for c in per_leg),
        sebi=sum(c.sebi for c in per_leg),
        stamp=sum(c.stamp for c in per_leg),
        brokerage=sum(c.brokerage for c in per_leg),
        gst=sum(c.gst for c in per_leg),
    )


def structure_spread_cost(legs, qty, spread_in_frac, spread_out_frac=None):
    """Half bid-ask spread cost (Rs) for crossing on entry + exit of a multi-leg structure.

    ``legs``: iterable of (premium_in, premium_out) per leg. You buy at ~mid + half-spread on entry and
    sell at ~mid - half-spread on exit, so the round-trip spread paid per leg is approximately
    (spread_in/2)*premium_in + (spread_out/2)*premium_out, times ``qty``. ``spread_*_frac`` are the FULL
    bid-ask spreads as a fraction of premium (e.g. 0.006 = the measured ~0.6% ATM open spread). This is
    SEPARATE from :func:`structure_round_trip_charges` (statutory fees) and added to it by the backtest.
    """
    legs = list(legs)
    if spread_out_frac is None:
        spread_out_frac = spread_in_frac
    if not legs:
        raise ValueError("legs must be non-empty (consistency with structure_round_trip_charges)")
    if qty <= 0 or spread_in_frac < 0 or spread_out_frac < 0:
        raise ValueError("qty must be > 0 and spreads must be non-negative")
    total = 0.0
    for leg in legs:
        leg_qty = leg[2] if len(leg) > 2 else qty  # per-leg qty for unequal ratio backspreads
        total += ((spread_in_frac / 2.0) * leg[0] + (spread_out_frac / 2.0) * leg[1]) * leg_qty
    return float(total)
