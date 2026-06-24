"""Unit tests for the HARDENED EOD overnight->open module + daily HAR (causality + walk-forward) + spread cost.

Covers the math-soundness review's must-haves: real HAR causality (y[t]==rv[t+1], not index bounds),
walk-forward no-lookahead, the absolute cost-hurdle gate + shrinkage, drive-decay/no-stop exit,
premium-based aggregate-capped Kelly sizing, entry greeks check, and all comparison boundaries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volarb.costs.nse_fees import structure_spread_cost
from volarb.signals.har_rv import (
    fit_har,
    forecast,
    har_daily_features,
    walk_forward_daily_forecast,
)
from volarb.strategy.eod_vol import (
    EodParams,
    corrected_edge,
    entry_decision,
    entry_greeks_ok,
    exit_decision,
    overnight_gap,
    position_lots,
    reliability_lambda,
    risk_gate,
)


# ---------- daily HAR: real causality (not just index bounds) ----------
def test_har_daily_causality_forward_shift_and_means():
    rng = np.random.default_rng(0)
    rv = pd.Series(np.abs(rng.normal(1e-4, 3e-5, 120)), index=pd.RangeIndex(120))
    X, y = har_daily_features(rv, lags=(1, 5, 22))
    assert list(X.columns) == ["rv_1d", "rv_5d", "rv_22d"]
    # FORWARD target (a y=shift(+1) lookahead bug would fail here, unlike index-bound checks):
    assert y.iloc[0] == pytest.approx(rv.iloc[22])            # first target = rv at t+1 (t=21)
    assert X.iloc[0]["rv_1d"] == pytest.approx(rv.iloc[21])   # rolling(1) ending at t
    assert X.iloc[0]["rv_5d"] == pytest.approx(rv.iloc[17:22].mean())
    assert X.iloc[0]["rv_22d"] == pytest.approx(rv.iloc[0:22].mean())
    assert forecast(fit_har(y, X, horizon_min=5, bar_min=5), X.iloc[-1]) >= 0.0


def test_walk_forward_is_no_lookahead():
    rng = np.random.default_rng(1)
    rv = pd.Series(np.abs(rng.normal(1e-4, 3e-5, 160)), index=pd.RangeIndex(160))
    wf = walk_forward_daily_forecast(rv, lags=(1, 5, 22), min_train=40)
    assert len(wf) > 0 and (wf >= 0).all()
    # altering a FUTURE rv must NOT change an early walk-forward forecast (causal fit on data < t)
    rv2 = rv.copy()
    rv2.iloc[-1] = rv.iloc[-1] * 100.0
    wf2 = walk_forward_daily_forecast(rv2, lags=(1, 5, 22), min_train=40)
    early = wf.index[0]
    assert wf.loc[early] == pytest.approx(wf2.loc[early])


# ---------- gap + shrinkage ----------
def test_overnight_gap():
    assert overnight_gap(0.20, 0.15) == pytest.approx(0.05)


def test_reliability_lambda_clip_and_coldstart():
    assert reliability_lambda([0.01] * 5, [0.0] * 5) == 1.0  # < 15 -> no shrink
    est = list(np.linspace(0, 0.1, 30))
    assert reliability_lambda(est, [0.0] * 30) == 0.0          # realized var 0 -> full shrink
    assert reliability_lambda(est, est) == pytest.approx(1.0)  # realized==est -> lambda 1 (clipped)


def test_corrected_edge_centered_posterior():
    est = list(np.linspace(-0.02, 0.06, 30))
    m = float(np.mean(est))
    # lambda=0 (realized constant) -> shrink fully to the prior mean m
    assert corrected_edge(0.05, est, [0.0] * 30) == pytest.approx(m)
    # lambda=1 (realized==est) -> no shrink, returns the gap
    assert corrected_edge(0.05, est, est) == pytest.approx(0.05)
    assert corrected_edge(0.05, [0.01] * 5, [0.0] * 5) == pytest.approx(0.05)  # cold start


# ---------- entry gate (absolute cost-hurdle, not quantile) ----------
def test_entry_gate_hurdle_boundary_and_structure():
    p = EodParams()
    assert entry_decision(0.06, 0.04, 0.50, params=p)["enter"] is True      # edge > hurdle
    assert entry_decision(0.04, 0.04, 0.50, params=p)["enter"] is False     # boundary: edge==hurdle -> no
    assert entry_decision(0.03, 0.04, 0.50, params=p)["enter"] is False
    assert entry_decision(0.06, 0.04, 0.50, params=p)["structure"] == "straddle"
    assert entry_decision(0.06, 0.04, 0.10, params=p)["structure"] == "strangle"
    assert entry_decision(0.06, 0.04, 0.15, params=p)["structure"] == "strangle"   # boundary <=
    assert entry_decision(0.06, 0.04, 0.1501, params=p)["structure"] == "straddle"


def test_entry_event_and_dte_exclusions():
    p = EodParams(min_dte_days=2)
    assert entry_decision(0.06, 0.04, 0.5, event_flag=True, params=p)["enter"] is False   # event night
    assert entry_decision(0.06, 0.04, 0.5, dte_days=1, params=p)["enter"] is False         # expiry-eve
    assert entry_decision(0.06, 0.04, 0.5, dte_days=2, params=p)["enter"] is True          # ok at min


# ---------- exit: drive-decay / time backstop / NO price stop ----------
def test_exit_drive_decay_time_and_no_price_stop():
    p = EodParams(exit_minute=60)
    assert exit_decision(60, p)["exit"] is True                                  # fixed backstop
    assert exit_decision(59, p)["exit"] is False                                 # off-by-one before
    assert exit_decision(20, p, theta_dominates=True)["exit"] is True            # drive-decay crossover
    assert exit_decision(20, p, theta_dominates=False)["exit"] is False
    assert exit_decision(20, p, pnl_fraction=0.4, take_profit_frac=0.25)["exit"] is True   # optional TP
    # NO price stop: a deep adverse mark before the backstop with no drive-decay must HOLD
    assert exit_decision(20, p, theta_dominates=False, pnl_fraction=-0.7)["exit"] is False


# ---------- entry greeks check (wires greeks_envelope) ----------
def test_entry_greeks_ok():
    assert entry_greeks_ok(10.0, 1.0, 2.0, 0.05)["ok"] is True               # long vega, delta-neutral, drive
    assert entry_greeks_ok(0.0, 1.0, 2.0, 0.05)["ok"] is False               # non-positive vega
    assert entry_greeks_ok(10.0, 3.0, 2.0, 0.05)["ok"] is False              # theta already dominates
    assert entry_greeks_ok(10.0, 1.0, 2.0, 0.50, delta_max=0.15)["ok"] is False  # delta breach


# ---------- sizing: premium-based, aggregate cap, margin cap ----------
def test_position_lots_premium_aggregate_and_margin_caps():
    p = EodParams(max_premium_frac=0.05, lam_kelly=0.5)
    # strong edge -> Kelly huge; the 5% premium cap binds: floor(0.05*1e6/11000)=4
    assert position_lots(0.05, 0.001, 50, capital=1e6, premium_per_lot=11_000,
                         margin_per_lot=120_000, params=p) == 4
    # aggregate: 30k premium already committed -> remaining 20k -> floor(20000/11000)=1
    assert position_lots(0.05, 0.001, 50, capital=1e6, premium_per_lot=11_000,
                         margin_per_lot=120_000, committed_premium=30_000, params=p) == 1
    # margin binds: tiny premium (prem cap huge) but large margin -> floor(1e6/200000)=5
    assert position_lots(0.05, 0.001, 50, capital=1e6, premium_per_lot=500,
                         margin_per_lot=200_000, params=p) == 5
    # weak edge -> Kelly (not a cap) binds: f~0.03 -> floor(0.03*1e6/11000)=2 < prem cap 4
    assert position_lots(0.0834, 0.278, 50, capital=1e6, premium_per_lot=11_000,
                         margin_per_lot=120_000, params=p) == 2
    # no / negative edge -> 0; degenerate inputs -> 0
    assert position_lots(-0.05, 0.001, 50, capital=1e6, premium_per_lot=11_000, margin_per_lot=120_000, params=p) == 0
    assert position_lots(0.05, 0.001, 50, capital=1e6, premium_per_lot=0, margin_per_lot=120_000, params=p) == 0


def test_risk_gate_killswitch_and_concentration():
    p = EodParams(max_overnight_positions=1)
    assert risk_gate(-60_000, 50_000, 0, p)["allow"] is False
    assert risk_gate(0.0, 50_000, 1, p)["allow"] is False
    assert risk_gate(0.0, 50_000, 0, p)["allow"] is True


# ---------- spread cost: asymmetric in/out + empty-leg guard ----------
def test_structure_spread_cost_asymmetric_and_empty():
    # asymmetric open-vs-close spread (the realistic case): per leg (s_in/2*pin + s_out/2*pout)*qty
    c = structure_spread_cost([(150.0, 160.0), (120.0, 110.0)], 75, 0.004, 0.008)
    exp = (0.002 * 150 + 0.004 * 160 + 0.002 * 120 + 0.004 * 110) * 75
    assert c == pytest.approx(exp) and c > 0
    assert structure_spread_cost([(150.0, 160.0)], 75, 0.0) == 0.0
    with pytest.raises(ValueError):
        structure_spread_cost([], 75, 0.006)           # empty legs now raise (was silent 0.0)
    with pytest.raises(ValueError):
        structure_spread_cost([(150.0, 160.0)], 0, 0.006)


def test_annualized_daily_vol_uses_trading_days_not_calendar_minutes():
    from volarb.strategy.eod_vol import TRADING_DAYS_PER_YEAR, annualized_daily_vol

    var = 2.4e-5  # a typical NIFTY daily realized variance
    assert annualized_daily_vol(var) == pytest.approx((var * TRADING_DAYS_PER_YEAR) ** 0.5)
    assert 0.05 < annualized_daily_vol(var) < 0.15  # plausible ~0.078; NOT ~0.18 (over-annualized bug)
    assert annualized_daily_vol(-1.0) == 0.0        # clips negative variance
