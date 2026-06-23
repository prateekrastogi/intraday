"""Coverage additions from the independent math-verification pass.

These pin the LOAD-BEARING properties the original tests only sanity-checked: no-lookahead,
unit flags, exact identities, normalizer constants, boundary dates, and the divide-by-zero /
negative-variance / duplicate-strike edge cases the verifiers flagged. Synthetic data only.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest

from volarb.costs.nse_fees import option_order_charges, stt_sell_rate
from volarb.data import panel
from volarb.pricing.greeks import YEAR_MINUTES, atm_iv, iv_greeks
from volarb.signals.lee_mykland import local_bipower_sigma
from volarb.signals.quarticity import harq_features, realized_quarticity, tripower_quarticity
from volarb.signals.semivariance import realized_semivariance, signed_jump_variation
from volarb.signals.spread import annualize_rv
from volarb.signals.surface import iv_percentile, johnson_gate, term_slope
from volarb.strategy.vol_gate import cost_hurdle, entry_decision

TRADE_DATE = dt.date(2026, 6, 23)


# T1 -- lee_mykland: sigma_hat[t] must be invariant to r[t] (no-lookahead), the headline property.
def test_lee_mykland_sigma_invariant_to_current_return():
    r = np.random.default_rng(7).normal(0, 1e-3, 120)
    base = local_bipower_sigma(r, 27)
    for t in (40, 80, 119):
        r2 = r.copy()
        r2[t] = 1.0
        s2 = local_bipower_sigma(r2, 27)
        assert np.isclose(base[t], s2[t], equal_nan=True)  # current return never leaks
        if t + 1 < r.size and np.isfinite(base[t + 1]):
            assert not np.isclose(base[t + 1], s2[t + 1])  # next window DOES see it


# T2 -- spread: the is_variance flag is load-bearing, and negative variance is clipped not NaN.
def test_annualize_rv_flag_is_load_bearing():
    h, target = 30.0, 0.20
    hv_var = target**2 * (h / YEAR_MINUTES)
    assert annualize_rv(hv_var, h, is_variance=True) == pytest.approx(0.20)
    assert (
        annualize_rv(hv_var, h, is_variance=False) < 1e-3
    )  # wrong flag collapses it (not a no-op)


def test_annualize_rv_clips_negative_variance():
    assert annualize_rv(-1e-4, 30.0, is_variance=True) == 0.0  # OLS can forecast <0 -> clip, no NaN


# T3 -- greeks: duplicate strike (call+put at same K) must not divide-by-zero; put side inverts.
def test_atm_iv_duplicate_strike_no_crash():
    q = [(24000, 165.0, True), (24050, 138.0, True), (24050, 137.0, False), (24100, 90.0, False)]
    v = atm_iv(q, forward=24050.0, t_years=2 / 365)
    assert v is None or 0.03 < v < 0.8


def test_iv_greeks_put_inversion_and_intrinsic():
    assert iv_greeks(50.0, 24100.0, 24300.0, 2 / 365, False) is None  # below intrinsic -> None
    g = iv_greeks(150.0, 24100.0, 24115.0, 2 / 365, False)
    assert g and 0.03 < g["vol"] < 0.8 and -0.8 < g["delta"] < -0.2 and g["vega"] > 0


# T4 -- panel: interior gap -> price stays null (no blanket ffill), bridging return dropped, OI ffilled.
def _bars(day, closes, oi=1000):
    base = dt.datetime.combine(day, dt.time(9, 15))
    rows = [
        {
            "datetime": base + dt.timedelta(minutes=i),
            "open": c,
            "high": c,
            "low": c,
            "close": c,
            "volume": 100,
            "open_interest": oi + i,
        }
        for i, c in enumerate(closes)
        if c is not None
    ]
    return pd.DataFrame(rows)


def test_panel_interior_gap_no_fabricated_return(tmp_path):
    day = dt.date(2026, 6, 16)
    closes = [24000.0, 24010.0, None, 24030.0, 24040.0]  # offset 2 (09:17) is a real hole
    p = tmp_path / "futures" / f"{day}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    _bars(day, closes).to_parquet(p)
    u = panel.load_underlying(tmp_path, day).sort("datetime")
    c, lr, oi = u["close"].to_list(), u["log_ret"].to_list(), u["open_interest"].to_list()
    assert c[2] is None  # price NOT blanket-ffilled
    assert lr[0] is None  # no overnight/roll return
    assert lr[2] is None and lr[3] is None  # gap + bridging return dropped (not 0)
    assert lr[1] is not None and lr[4] is not None
    assert oi[2] is not None and oi[2] == oi[1]  # OI IS ffilled (stock, not flow)


# T5 -- nse_fees: STT date-ladder cutovers are inclusive; charges linear in qty, brokerage flat.
def test_stt_cutover_boundaries_inclusive():
    assert stt_sell_rate(dt.date(2024, 9, 30)) == 0.000625
    assert stt_sell_rate(dt.date(2024, 10, 1)) == 0.0010
    assert stt_sell_rate(dt.date(2026, 3, 31)) == 0.0010
    assert stt_sell_rate(dt.date(2026, 4, 1)) == 0.0015


def test_charges_linear_in_qty_brokerage_flat():
    c1 = option_order_charges("SELL", 100.0, 75, dt.date(2026, 6, 22))
    c2 = option_order_charges("SELL", 100.0, 150, dt.date(2026, 6, 22))
    for f in ("stt", "exchange", "sebi"):
        assert getattr(c2, f) == pytest.approx(2 * getattr(c1, f))
    assert c1.brokerage == 20.0 and c2.brokerage == 20.0  # flat, not per-unit


# T6 -- vol_gate: cost_hurdle is a small annualized-vol number and differences cleanly vs the spread.
def _atm_vega(F=23000.0, K=23000.0, sigma=0.13, r=0.065, days=5):
    T = (days * 24 * 60) / YEAR_MINUTES
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    phi = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
    return F * math.exp(-r * T) * phi * math.sqrt(T)


def test_cost_hurdle_is_vol_unit_with_realistic_vega():
    h = cost_hurdle(premium=160.0, qty=75, trade_date=TRADE_DATE, vega=_atm_vega())
    assert 0.0003 < h < 0.02  # vol points, NOT ~0.17 (a units regression)


def test_cost_hurdle_flows_into_entry_decision():
    h = cost_hurdle(160.0, 75, TRADE_DATE, vega=_atm_vega())
    regime = {"iv_pct": 0.25, "slope": -0.01, "jump_flag": False}
    assert entry_decision(spread=h * 2.0, regime=regime, hurdle=h)["enter"] is True
    assert entry_decision(spread=h * 0.5, regime=regime, hurdle=h)["enter"] is False


# T7 -- surface: strict-< self-excluded percentile (ties, non-finite, flat history) + gate semantics.
def test_iv_percentile_strict_tie_self_excluded():
    assert iv_percentile(0.20, [0.10, 0.20, 0.30]) == pytest.approx(1 / 3)
    assert iv_percentile(0.20, [0.10, 0.10, 0.10, 0.20]) == pytest.approx(3 / 4)


def test_iv_percentile_nonfinite_now_is_nan():
    assert math.isnan(iv_percentile(float("nan"), [0.1, 0.2]))
    assert math.isnan(iv_percentile(float("inf"), [0.1, 0.2]))


def test_iv_percentile_flat_history_is_zero_and_opens_gate():
    pct = iv_percentile(0.20, [0.20, 0.20, 0.20])
    assert pct == pytest.approx(0.0)
    assert johnson_gate(slope=0.04, iv_pct=pct) is True
    assert math.isnan(term_slope(float("nan"), 0.22))  # missing front IV -> nan slope


# T8 -- quarticity: exact mu_{4/3} normalizer, short-lag interaction, variance^2 scaling.
def test_tripower_normalizer_exact_on_unit_series():
    n = 10
    mu43 = 2.0 ** (2.0 / 3.0) * math.gamma(7.0 / 6.0) / math.sqrt(math.pi)
    assert tripower_quarticity(np.ones(n)) == pytest.approx(n * mu43**-3 * (n - 2), rel=1e-12)


def test_harq_interacts_short_lag_only():
    rv = np.array([[2.0, 5.0, 7.0], [3.0, 11.0, 13.0]])
    rq = np.array([4.0, 9.0])  # sqrt -> 2, 3
    design = harq_features(rv, rq)
    np.testing.assert_allclose(design[:, 1], np.array([2.0 * 2.0, 3.0 * 3.0]))
    assert not np.allclose(design[:, 1], np.array([2.0, 3.0]) * rv[:, 1])


def test_rq_scales_as_variance_squared():
    r = np.random.default_rng(7).normal(0, 1e-3, 500)
    assert realized_quarticity(3.0 * r) == pytest.approx(3.0**4 * realized_quarticity(r), rel=1e-12)


# semivariance: RS+ + RS- == RV exactly even with zeros present; signed jump scales quadratically.
def test_semivariance_identity_with_zeros():
    rng = np.random.default_rng(11)
    r = rng.standard_normal(300) * 1e-3
    r[::5] = 0.0
    rs_plus, rs_minus = realized_semivariance(r)
    assert rs_plus + rs_minus == pytest.approx(float(np.sum(r * r)), rel=1e-12, abs=1e-18)


def test_signed_jump_quadratic_scaling():
    r = np.random.default_rng(12).standard_normal(200) * 1e-3
    assert signed_jump_variation(3.7 * r) == pytest.approx(
        3.7**2 * signed_jump_variation(r), rel=1e-12
    )


# 2-leg STRUCTURE cost: a straddle round-trip is 4 orders (Rs80 brokerage), both legs' STT/exch.
def test_structure_round_trip_is_two_legs():
    from volarb.costs.nse_fees import round_trip_charges, structure_round_trip_charges

    d = dt.date(2026, 6, 22)
    s = structure_round_trip_charges([(150.0, 150.0), (140.0, 140.0)], 75, d)
    call = round_trip_charges(150.0, 150.0, 75, d)
    put = round_trip_charges(140.0, 140.0, 75, d)
    assert s.brokerage == pytest.approx(80.0)  # 4 orders (2 legs x buy+sell), not 40
    assert s.stt == pytest.approx(call.stt + put.stt)  # both legs pay STT on their own premium
    assert s.exchange == pytest.approx(call.exchange + put.exchange)
    assert s.total == pytest.approx(call.total + put.total)
    assert s.total > call.total  # the structure costs more than one leg


def test_structure_round_trip_empty_raises():
    from volarb.costs.nse_fees import structure_round_trip_charges

    with pytest.raises(ValueError):
        structure_round_trip_charges([], 75, dt.date(2026, 6, 22))


def test_structure_cost_hurdle_symmetric_straddle_matches_per_leg():
    from volarb.strategy.vol_gate import cost_hurdle, structure_cost_hurdle

    d, leg_vega = dt.date(2026, 6, 22), 8.0
    per_leg = cost_hurdle(150.0, 75, d, vega=leg_vega)
    struct = structure_cost_hurdle([(150.0, 150.0), (150.0, 150.0)], 75, d, total_vega=2 * leg_vega)
    assert struct == pytest.approx(per_leg)  # symmetric straddle: doubling cancels


def test_structure_cost_hurdle_rejects_bad_inputs():
    from volarb.strategy.vol_gate import structure_cost_hurdle

    with pytest.raises(ValueError):
        structure_cost_hurdle([(150.0, 150.0)], 0, dt.date(2026, 6, 22), total_vega=8.0)
    with pytest.raises(ValueError):
        structure_cost_hurdle([(150.0, 150.0)], 75, dt.date(2026, 6, 22), total_vega=0.0)
