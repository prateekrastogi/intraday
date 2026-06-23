"""IV/greeks wrapper tests -- exercise the real Nautilus Black-76 inverter."""

import math

from volarb.pricing.greeks import YEAR_MINUTES, atm_iv, iv_greeks, years_to_expiry

T2D = 2 / 365  # 2-day weekly tenor reused across cases
R = 0.065  # DEFAULT_R, restated so the discounted-intrinsic guard can be reconstructed in-test


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


# --------------------------------------------------------------------------------------------
# Gap-filling coverage from the independent math-verification pass.
# Black-76 (b=0, s=forward): C = e^{-rT}[F N(d1) - K N(d2)], no-arb floor = e^{-rT} max(F-K,0).
# atm_iv re-inverts each quote at `forward`, vega-filters, dedups strikes, then does a LINEAR
# interpolation of (strike, vol) to K=forward with inclusive flat clamping outside the knots.
# --------------------------------------------------------------------------------------------


def test_years_to_expiry_floor_value_and_negative():
    # floored to exactly 1 minute so the solver always sees t>0, never <=0.
    assert years_to_expiry(0) == 1.0 / YEAR_MINUTES
    assert years_to_expiry(-5) == 1.0 / YEAR_MINUTES  # negative minutes also floored, not negative
    assert years_to_expiry(1) == 1.0 / YEAR_MINUTES


def test_iv_greeks_rejects_t_negative_not_just_zero():
    # the original suite pins t==0; t<0 must also short-circuit to None before the solver.
    assert iv_greeks(150.0, 24100.0, 24100.0, -1.0, True) is None


def test_iv_greeks_intrinsic_epsilon_boundary_inclusive():
    # guard is price <= discounted_intrinsic + 1e-6 -> reject. Pin both sides of the epsilon edge.
    F, K = 24500.0, 24100.0
    intrinsic = math.exp(-R * T2D) * (F - K)
    assert iv_greeks(intrinsic + 1e-6, F, K, T2D, True) is None  # boundary is rejected
    assert iv_greeks(intrinsic + 0.5, F, K, T2D, True) is not None  # just above -> inverts


def test_iv_greeks_deep_itm_just_above_floor_inverts_with_high_delta():
    F, K = 24500.0, 24100.0
    intrinsic = math.exp(-R * T2D) * (F - K)
    g = iv_greeks(intrinsic + 5.0, F, K, T2D, True)
    assert g is not None
    assert g["delta"] > 0.85  # deep ITM call delta -> 1
    assert g["vol"] > 0 and g["vega"] > 0


def test_iv_greeks_deep_otm_call_inverts_with_low_delta():
    g = iv_greeks(0.5, 24100.0, 26000.0, T2D, True)
    assert g is not None
    assert 0.0 < g["delta"] < 0.05  # deep OTM call delta -> 0
    assert g["vol"] > 0


def test_iv_greeks_monotone_in_price():
    # at fixed strike, a richer option must imply a higher vol (inverter monotonicity).
    F = K = 24100.0
    vols = [iv_greeks(p, F, K, T2D, True)["vol"] for p in (80.0, 120.0, 160.0, 200.0)]
    assert all(vols[i] < vols[i + 1] for i in range(len(vols) - 1))


def test_iv_greeks_put_call_iv_agree_at_atm():
    # under b=0 a same-strike call and put quoted at the same price imply the same vol.
    F = K = 24100.0
    gc = iv_greeks(150.0, F, K, T2D, True)
    gp = iv_greeks(150.0, F, K, T2D, False)
    assert gc["vol"] == pytest_approx(gp["vol"])
    assert gp["delta"] < 0  # put delta sign


def test_atm_iv_interpolation_is_exact_linear_off_knot():
    # F=24075 sits midway (w=0.5) between the 24050 and 24100 knots -> arithmetic mean of the
    # two re-inverted vols, not merely "in a plausible band".
    quotes = [(24000, 165.0, True), (24050, 138.0, True), (24100, 113.0, True)]
    F = 24075.0
    v_lo = iv_greeks(138.0, F, 24050, T2D, True)["vol"]
    v_hi = iv_greeks(113.0, F, 24100, T2D, True)["vol"]
    assert atm_iv(quotes, forward=F, t_years=T2D) == pytest_approx(0.5 * (v_lo + v_hi))


def test_atm_iv_returns_exact_knot_vol_when_forward_on_interior_knot():
    quotes = [(24000, 165.0, True), (24050, 138.0, True), (24100, 113.0, True)]
    F = 24050.0
    expected = iv_greeks(138.0, F, 24050, T2D, True)["vol"]
    assert atm_iv(quotes, forward=F, t_years=T2D) == pytest_approx(expected)


def test_atm_iv_clamps_flat_below_lowest_strike():
    quotes = [(24000, 165.0, True), (24100, 113.0, True)]
    F = 23900.0  # below all knots -> return the lowest-strike vol unchanged
    expected = iv_greeks(165.0, F, 24000, T2D, True)["vol"]
    assert atm_iv(quotes, forward=F, t_years=T2D) == pytest_approx(expected)


def test_atm_iv_clamps_flat_above_highest_strike():
    quotes = [(23000, 1120.0, True), (23100, 1030.0, True)]
    F = 23500.0  # above all knots, but calls still above discounted intrinsic so they invert
    expected = iv_greeks(1030.0, F, 23100, T2D, True)["vol"]
    assert atm_iv(quotes, forward=F, t_years=T2D) == pytest_approx(expected)


def test_atm_iv_dedup_is_arithmetic_mean_of_call_and_put():
    # call + put at the SAME strike collapse to one knot = mean of their two inverted vols.
    F = 24050.0
    q = [(24000, 165.0, True), (24050, 138.0, True), (24050, 137.0, False), (24100, 113.0, True)]
    vc = iv_greeks(138.0, F, 24050, T2D, True)["vol"]
    vp = iv_greeks(137.0, F, 24050, T2D, False)["vol"]
    assert atm_iv(q, forward=F, t_years=T2D) == pytest_approx(0.5 * (vc + vp))


def test_atm_iv_drops_sub_vega_floor_knots():
    # a far-OTM quote with vega < DEFAULT_VEGA_FLOOR (0.5) must be filtered out; with only one
    # surviving usable point the result is None.
    far = iv_greeks(0.5, 24050.0, 26000.0, T2D, True)
    assert far["vega"] < 0.5  # precondition: this knot is below the floor
    assert atm_iv([(24050, 138.0, True), (26000, 0.5, True)], forward=24050.0, t_years=T2D) is None
    # but with two real knots present the sub-floor one is simply ignored, not fatal.
    q = [(24000, 165.0, True), (24050, 138.0, True), (26000, 0.5, True)]
    assert atm_iv(q, forward=24025.0, t_years=T2D) is not None


def pytest_approx(x):
    import pytest

    return pytest.approx(x, rel=1e-9, abs=1e-12)
