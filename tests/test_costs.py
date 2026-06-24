"""Golden-file tests for the NSE options cost model. Exact Rs values are hand-computed.
Example trade: 1 NIFTY lot (75 units) of an option at premium 100 -> turnover Rs 7,500."""

from datetime import date

import pytest

from volarb.costs.nse_fees import (
    exercise_stt,
    option_order_charges,
    round_trip_charges,
    stt_sell_rate,
    structure_round_trip_charges,
)

QTY = 75  # NIFTY lot size (current; illustrative -- lot size is date-varying and lives in the instrument layer, NOT the cost model, which is qty-based)
PREMIUM = 100.0  # per unit -> turnover = 7500


def test_stt_rate_is_date_aware():
    assert stt_sell_rate(date(2024, 1, 1)) == 0.000625  # pre Oct-2024
    assert stt_sell_rate(date(2025, 1, 1)) == 0.0010  # Oct-2024 .. Mar-2026
    assert stt_sell_rate(date(2026, 6, 22)) == 0.0015  # from Apr-2026


def test_sell_order_apr2026():
    c = option_order_charges("SELL", PREMIUM, QTY, date(2026, 6, 22))
    assert c.stt == pytest.approx(11.25)  # 7500 * 0.0015
    assert c.exchange == pytest.approx(2.66475)  # 7500 * 0.0003553
    assert c.sebi == pytest.approx(0.0075)  # 7500 * 0.000001
    assert c.stamp == 0.0  # no stamp on sell
    assert c.brokerage == 20.0
    assert c.gst == pytest.approx(4.081005)  # 0.18 * (20 + 2.66475 + 0.0075)
    assert c.total == pytest.approx(38.003255)


def test_buy_order_apr2026():
    c = option_order_charges("BUY", PREMIUM, QTY, date(2026, 6, 22))
    assert c.stt == 0.0  # no STT on buy
    assert c.stamp == pytest.approx(0.225)  # 7500 * 0.00003
    assert c.total == pytest.approx(26.978255)


def test_sell_stt_lower_before_apr2026():
    c = option_order_charges("SELL", PREMIUM, QTY, date(2025, 1, 1))
    assert c.stt == pytest.approx(7.50)  # 7500 * 0.0010
    assert c.total == pytest.approx(34.253255)


def test_round_trip_is_about_point_eight_percent():
    rt = round_trip_charges(PREMIUM, PREMIUM, QTY, date(2026, 6, 22))
    turnover = PREMIUM * QTY
    assert rt.total == pytest.approx(64.98151)  # 26.978255 + 38.003255
    assert 0.008 < rt.total / turnover < 0.010  # ~0.87% of premium (IMPLEMENTATION.md sec.4)


def test_itm_exercise_stt():
    assert exercise_stt(50.0, QTY) == pytest.approx(4.6875)  # 3750 * 0.00125


# ---- gap-filling coverage from the independent math-verification pass ----


def test_pre_oct2024_stt_rung_is_charged():
    # The golden tests pin only the two newer rungs end-to-end; the 0.0625% rung was never
    # exercised through option_order_charges. Pin it so a ladder regression cannot hide here.
    c = option_order_charges("SELL", PREMIUM, QTY, date(2024, 1, 1))
    assert c.stt == pytest.approx(7500 * 0.000625)  # 4.6875
    assert c.total == pytest.approx(4.6875 + 2.66475 + 0.0075 + 0.0 + 20.0 + 4.081005)


def test_gst_base_excludes_stt_and_stamp():
    # GST is 18% on (brokerage + exchange + SEBI) ONLY. STT (sell) and stamp (buy) must NOT
    # enter the base, so BUY and SELL GST are identical despite very different STT/stamp.
    buy = option_order_charges("BUY", PREMIUM, QTY, date(2026, 6, 22))
    sell = option_order_charges("SELL", PREMIUM, QTY, date(2026, 6, 22))
    assert buy.gst == pytest.approx(sell.gst)
    assert sell.gst == pytest.approx(0.18 * (20.0 + sell.exchange + sell.sebi))
    # explicit: removing STT/stamp from the picture would not change GST
    assert buy.gst == pytest.approx(0.18 * (buy.brokerage + buy.exchange + buy.sebi))


def test_custom_brokerage_flows_into_gst_base():
    # Brokerage is configurable (and is part of the GST base) -- a zero-brokerage broker still
    # pays GST on exchange + SEBI only, with no flat Rs20 floor sneaking in.
    c = option_order_charges("SELL", PREMIUM, QTY, date(2026, 6, 22), brokerage=0.0)
    assert c.brokerage == 0.0
    assert c.gst == pytest.approx(0.18 * (c.exchange + c.sebi))
    c2 = option_order_charges("SELL", PREMIUM, QTY, date(2026, 6, 22), brokerage=50.0)
    assert c2.gst == pytest.approx(0.18 * (50.0 + c2.exchange + c2.sebi))


def test_buy_has_no_stt_sell_has_no_stamp():
    buy = option_order_charges("BUY", PREMIUM, QTY, date(2026, 6, 22))
    sell = option_order_charges("SELL", PREMIUM, QTY, date(2026, 6, 22))
    assert buy.stt == 0.0 and sell.stamp == 0.0
    assert buy.stamp > 0.0 and sell.stt > 0.0  # the other tax IS present on each side


def test_total_is_exact_field_sum():
    c = option_order_charges("SELL", PREMIUM, QTY, date(2026, 6, 22))
    assert c.total == pytest.approx(c.stt + c.exchange + c.sebi + c.stamp + c.brokerage + c.gst)


def test_invalid_side_raises():
    with pytest.raises(ValueError):
        option_order_charges("HOLD", PREMIUM, QTY, date(2026, 6, 22))


def test_side_is_case_insensitive():
    lo = option_order_charges("sell", PREMIUM, QTY, date(2026, 6, 22))
    hi = option_order_charges("SELL", PREMIUM, QTY, date(2026, 6, 22))
    assert lo.total == pytest.approx(hi.total)


def test_charges_linear_in_qty_except_flat_brokerage():
    c1 = option_order_charges("SELL", PREMIUM, QTY, date(2026, 6, 22))
    c2 = option_order_charges("SELL", PREMIUM, 3 * QTY, date(2026, 6, 22))
    for f in ("stt", "exchange", "sebi"):
        assert getattr(c2, f) == pytest.approx(3 * getattr(c1, f))
    assert c2.brokerage == c1.brokerage == 20.0  # flat, not scaled
    # GST scales the variable part but keeps the flat brokerage component constant
    assert c2.gst == pytest.approx(0.18 * (20.0 + c2.exchange + c2.sebi))


def test_exercise_stt_linear_and_zero_at_otm():
    assert exercise_stt(0.0, QTY) == 0.0  # OTM expiry: no intrinsic -> no exercise STT
    assert exercise_stt(100.0, QTY) == pytest.approx(2 * exercise_stt(50.0, QTY))


def test_round_trip_is_buy_plus_sell_field_by_field():
    d = date(2026, 6, 22)
    rt = round_trip_charges(120.0, 130.0, QTY, d)
    b = option_order_charges("BUY", 120.0, QTY, d)
    s = option_order_charges("SELL", 130.0, QTY, d)
    for f in ("stt", "exchange", "sebi", "stamp", "brokerage", "gst"):
        assert getattr(rt, f) == pytest.approx(getattr(b, f) + getattr(s, f))
    assert rt.brokerage == pytest.approx(40.0)  # 2 orders


def test_round_trip_uses_trade_date_stt():
    # Entry/exit STT both follow the single trade_date rung (pre- vs post-Apr-2026 differ).
    qty, prem = QTY, 100.0
    old = round_trip_charges(prem, prem, qty, date(2025, 1, 1))
    new = round_trip_charges(prem, prem, qty, date(2026, 6, 22))
    assert new.stt > old.stt  # 0.0015 vs 0.0010 rung
    assert old.stt == pytest.approx(7500 * 0.0010)  # only sell leg pays STT


def test_structure_straddle_is_four_orders_and_leg_sum():
    d = date(2026, 6, 22)
    legs = [(150.0, 155.0), (140.0, 138.0)]  # call leg, put leg (asymmetric exits)
    struct = structure_round_trip_charges(legs, QTY, d)
    call = round_trip_charges(150.0, 155.0, QTY, d)
    put = round_trip_charges(140.0, 138.0, QTY, d)
    assert struct.brokerage == pytest.approx(80.0)  # 4 orders x Rs20
    for f in ("stt", "exchange", "sebi", "stamp", "gst", "total"):
        assert getattr(struct, f) == pytest.approx(getattr(call, f) + getattr(put, f))


def test_structure_single_leg_equals_round_trip():
    d = date(2026, 6, 22)
    one = structure_round_trip_charges([(100.0, 110.0)], QTY, d)
    rt = round_trip_charges(100.0, 110.0, QTY, d)
    assert one.total == pytest.approx(rt.total)


def test_structure_empty_legs_raises():
    with pytest.raises(ValueError):
        structure_round_trip_charges([], QTY, date(2026, 6, 22))


def test_structure_round_trip_per_leg_qty_backspread():
    """1x2 backspread: the long leg (2 lots) must be charged on its own larger qty (per-leg qty)."""
    import datetime as dt

    from volarb.costs.nse_fees import structure_round_trip_charges

    d = dt.date(2025, 1, 15)
    lot = 75
    # short 1 lot near-money, long 2 lots OTM (unequal ratio)
    legs_ratio = [(50.0, 90.0, 2 * lot), (140.0, 110.0, 1 * lot)]   # (in, out, leg_qty)
    legs_equal = [(50.0, 90.0), (140.0, 110.0)]                      # scalar qty path (both = lot)
    c_ratio = structure_round_trip_charges(legs_ratio, lot, d).total
    c_equal = structure_round_trip_charges(legs_equal, lot, d).total
    assert c_ratio > c_equal > 0                                     # doubling the long leg costs more
    # backward-compat: a 2-tuple leg with leg_qty==scalar qty matches the scalar path
    same = structure_round_trip_charges([(50.0, 90.0, lot), (140.0, 110.0, lot)], lot, d).total
    assert same == c_equal
