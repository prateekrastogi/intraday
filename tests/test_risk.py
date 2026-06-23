"""Verification tests for risk.py.

The Leung-Li exit math is checked against the PAPER's own propositions (arXiv:1411.5062) as
oracles; Kelly sizing and the decision/risk-floor logic are self-contained. Math tests use
well-conditioned OU params (r/mu ~ O(1)).
"""

from __future__ import annotations

import numpy as np
import pytest

from volarb.strategy.risk import (
    OUParams,
    bm_take_profit,
    exit_decision,
    fit_ou,
    F,
    Fp,
    fractional_kelly,
    G,
    Gp,
    greeks_envelope,
    kelly_lots,
    kill_switch,
    l_star,
    take_profit,
    take_profit_stoploss,
)

P = OUParams(mu=2.0, theta=0.0, sigma=0.5, ok=True)
R, C = 1.0, 0.05


# ---- OU MLE fit (Eq 2.2) ----
def test_fit_ou_recovers_known_params():
    rng = np.random.default_rng(0)
    mu, theta, sigma, dt, n = 3.0, 0.10, 0.40, 1 / 252, 4000
    b, x = np.exp(-mu * dt), np.empty(4000)
    sd = sigma * np.sqrt((1 - b * b) / (2 * mu))
    x[0] = theta
    for i in range(1, n):
        x[i] = theta + b * (x[i - 1] - theta) + rng.normal(0, sd)
    p = fit_ou(x, dt)
    assert p.ok
    assert p.theta == pytest.approx(theta, abs=0.05)
    assert p.mu == pytest.approx(mu, rel=0.4)
    assert p.sigma == pytest.approx(sigma, rel=0.2)


def test_fit_ou_rejects_random_walk():
    rng = np.random.default_rng(1)
    rw = np.cumsum(rng.normal(0, 1e-3, 500))
    assert fit_ou(rw, 1 / 252).ok is False


# ---- F, G fundamental-solution shape (paper p.7: F up/F'>0, G down/G'<0) ----
def test_F_increasing_G_decreasing():
    xs = [-0.5, 0.0, 0.5, 1.0]
    fv = [F(x, P, R) for x in xs]
    gv = [G(x, P, R) for x in xs]
    assert all(fv[i] < fv[i + 1] for i in range(len(xs) - 1))
    assert all(gv[i] > gv[i + 1] for i in range(len(xs) - 1))
    assert all(Fp(x, P, R) > 0 for x in xs)
    assert all(Gp(x, P, R) < 0 for x in xs)


# ---- L* (Eq 4.1) ----
def test_l_star_formula():
    assert l_star(P, R, C) == pytest.approx((P.mu * P.theta + R * C) / (P.mu + R))


# ---- take-profit, no stop (Eq 4.3) + Prop 4.3 (b* increasing in cost c) ----
def test_take_profit_above_cost_and_increasing_in_c():
    b_lo = take_profit(P, R, 0.02)
    b_hi = take_profit(P, R, 0.20)
    assert b_lo is not None and b_lo > 0.02
    assert b_hi is not None and b_hi > b_lo


# ---- take-profit WITH stop-loss (Eq 5.5): Prop 5.3 + Cor 5.2 ----
def test_take_profit_stoploss_decreasing_in_L():
    ls = l_star(P, R, C)
    levels = [ls - 1.0, ls - 0.5, ls - 0.2, ls - 0.05]
    bs = [take_profit_stoploss(P, R, C, ll) for ll in levels]
    assert all(b is not None for b in bs)
    assert all(bs[i] > bs[i + 1] for i in range(len(bs) - 1))  # Prop 5.3


def test_stoploss_meets_Lstar_and_no_trade_above():
    ls = l_star(P, R, C)
    assert take_profit_stoploss(P, R, C, ls + 0.01) is None  # Cor 5.2: L>=L* -> immediate liq
    near = take_profit_stoploss(P, R, C, ls - 1e-3)
    assert near is not None and near == pytest.approx(ls, abs=5e-2)  # b*_L -> L* as L -> L*-


# ---- Brownian degenerate (Remark 4.7) ----
def test_bm_take_profit_formula():
    assert bm_take_profit(C, P.sigma, R) == pytest.approx(C + P.sigma / np.sqrt(2 * R))


# ---- fractional Kelly ----
def test_fractional_kelly_basic_and_cap():
    f = fractional_kelly(0.02, 0.04, 400, lam_base=0.5)
    assert 0 < f <= 0.02 / 0.04  # never exceeds full Kelly
    assert fractional_kelly(0.02, 0.04, 400, 0.25) < fractional_kelly(0.02, 0.04, 400, 0.5)


def test_fractional_kelly_no_edge_is_zero():
    assert fractional_kelly(-0.01, 0.04, 400) == 0.0
    assert fractional_kelly(0.02, 0.0, 400) == 0.0


def test_fractional_kelly_tstat_shrink():
    assert fractional_kelly(0.02, 0.04, 3) < fractional_kelly(0.02, 0.04, 4000)


def test_kelly_lots_rounds_down():
    lots = kelly_lots(0.02, 0.04, 400, 1_000_000, 150_000, lam_base=0.5)
    assert isinstance(lots, int) and lots >= 0


# ---- model-free risk floor ----
def test_kill_switch():
    assert kill_switch(-5000.0, 4000.0) is True
    assert kill_switch(-3000.0, 4000.0) is False


def test_greeks_envelope_flags():
    e = greeks_envelope(3.0, 10.0, 5.0, 2.0, 0.6, vega_floor_frac=0.5, delta_max=0.5)
    assert e["vega_dead"] and e["theta_dominates"] and e["delta_breach"]
    ok = greeks_envelope(8.0, 10.0, 1.0, 5.0, 0.1, vega_floor_frac=0.5, delta_max=0.5)
    assert not (ok["vega_dead"] or ok["theta_dominates"] or ok["delta_breach"])


# ---- exit decision precedence ----
def test_exit_decision_precedence():
    assert exit_decision(0.0, 0.3, -0.5, 1.0, 0)["exit"] is True
    assert "stop" in exit_decision(-0.6, 0.3, -0.5, 1.0, 60)["reason"]
    assert "take-profit" in exit_decision(0.4, 0.3, -0.5, 1.0, 60)["reason"]
    assert "drive-decay" in exit_decision(0.1, 0.3, -0.5, -0.1, 60)["reason"]
    assert exit_decision(0.1, 0.3, -0.5, 1.0, 60)["exit"] is False
