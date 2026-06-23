"""Golden-file tests for the NSE options cost model. Exact Rs values are hand-computed.
Example trade: 1 NIFTY lot (75 units) of an option at premium 100 -> turnover Rs 7,500."""
from datetime import date

import pytest

from volarb.costs.nse_fees import (
    exercise_stt,
    option_order_charges,
    round_trip_charges,
    stt_sell_rate,
)

QTY = 75          # NIFTY lot size (current; illustrative -- lot size is date-varying and lives in the instrument layer, NOT the cost model, which is qty-based)
PREMIUM = 100.0   # per unit -> turnover = 7500


def test_stt_rate_is_date_aware():
    assert stt_sell_rate(date(2024, 1, 1)) == 0.000625    # pre Oct-2024
    assert stt_sell_rate(date(2025, 1, 1)) == 0.0010      # Oct-2024 .. Mar-2026
    assert stt_sell_rate(date(2026, 6, 22)) == 0.0015     # from Apr-2026


def test_sell_order_apr2026():
    c = option_order_charges("SELL", PREMIUM, QTY, date(2026, 6, 22))
    assert c.stt == pytest.approx(11.25)          # 7500 * 0.0015
    assert c.exchange == pytest.approx(2.66475)   # 7500 * 0.0003553
    assert c.sebi == pytest.approx(0.0075)        # 7500 * 0.000001
    assert c.stamp == 0.0                         # no stamp on sell
    assert c.brokerage == 20.0
    assert c.gst == pytest.approx(4.081005)       # 0.18 * (20 + 2.66475 + 0.0075)
    assert c.total == pytest.approx(38.003255)


def test_buy_order_apr2026():
    c = option_order_charges("BUY", PREMIUM, QTY, date(2026, 6, 22))
    assert c.stt == 0.0                           # no STT on buy
    assert c.stamp == pytest.approx(0.225)        # 7500 * 0.00003
    assert c.total == pytest.approx(26.978255)


def test_sell_stt_lower_before_apr2026():
    c = option_order_charges("SELL", PREMIUM, QTY, date(2025, 1, 1))
    assert c.stt == pytest.approx(7.50)           # 7500 * 0.0010
    assert c.total == pytest.approx(34.253255)


def test_round_trip_is_about_point_eight_percent():
    rt = round_trip_charges(PREMIUM, PREMIUM, QTY, date(2026, 6, 22))
    turnover = PREMIUM * QTY
    assert rt.total == pytest.approx(64.98151)    # 26.978255 + 38.003255
    assert 0.008 < rt.total / turnover < 0.010    # ~0.87% of premium (IMPLEMENTATION.md sec.4)


def test_itm_exercise_stt():
    assert exercise_stt(50.0, QTY) == pytest.approx(4.6875)   # 3750 * 0.00125
