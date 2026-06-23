"""IV/greeks wrapper tests -- exercise the real Nautilus Black-76 inverter."""

import math

from volarb.pricing.greeks import atm_iv, iv_greeks, years_to_expiry


def test_years_to_expiry_floors_positive():
    assert years_to_expiry(0) > 0
    assert math.isclose(years_to_expiry(2 * 24 * 60), 2 / 365, rel_tol=1e-9)  # 2 days


def test_iv_greeks_atm_weekly_is_sane():
    g = iv_greeks(price=150.0, forward=24115.0, strike=24100.0, t_years=2 / 365, is_call=True)
    assert g is not None
    assert 0.05 < g["vol"] < 0.60  # plausible NIFTY weekly IV
    assert 0.3 < g["delta"] < 0.7  # ~ATM call
    assert g["vega"] > 0


def test_iv_greeks_rejects_below_intrinsic():
    assert (
        iv_greeks(price=50.0, forward=24300.0, strike=24100.0, t_years=2 / 365, is_call=True)
        is None
    )
    assert (
        iv_greeks(price=100.0, forward=24100.0, strike=24100.0, t_years=0.0, is_call=True) is None
    )


def test_atm_iv_interpolates_to_forward():
    quotes = [(24000, 165.0, True), (24050, 138.0, True), (24100, 113.0, True)]
    v = atm_iv(quotes, forward=24050.0, t_years=2 / 365)
    assert v is not None and 0.10 < v < 0.35


def test_atm_iv_none_when_no_usable_points():
    assert atm_iv([(24000, 1.0, True)], forward=24000.0, t_years=2 / 365) is None
