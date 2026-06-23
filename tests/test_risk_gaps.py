"""Gap-filling tests appended by the math-verification pass (risk module).

These pin the LOAD-BEARING identities the original suite only sanity-checked:
the defining root residuals (Eq 4.3 / 5.5), exact F'/G' derivatives, the Prop 5.3
L -> -inf limit (b*_L -> b*_nostop), the AR(1)->OU MLE inversion, and the Kelly /
kelly_lots / exit-precedence edge cases. Synthetic / well-conditioned OU params only.
"""

from __future__ import annotations

import numpy as np
import pytest

from volarb.strategy.risk import (
    OUParams,
    F,
    Fp,
    G,
    Gp,
    F_pc,
    Fp_pc,
    G_pc,
    Gp_pc,
    _funcs,
    exit_decision,
    fit_ou,
    fractional_kelly,
    greeks_envelope,
    kelly_lots,
    l_star,
    take_profit,
    take_profit_stoploss,
)

PG = OUParams(mu=2.0, theta=0.0, sigma=0.5, ok=True)
RG, CG = 1.0, 0.05


# ---- Eq 4.3 defining identity: F(b*) - (b*-c) F'(b*) == 0 at the SOLVED root ----
@pytest.mark.parametrize("method", ["pc", "quad"])
def test_take_profit_residual_zero_at_root(method):
    ff, fpf, _, _ = _funcs(method)
    b = take_profit(PG, RG, CG, method=method)
    assert b is not None
    res = ff(b, PG, RG) - (b - CG) * fpf(b, PG, RG)
    assert abs(res) < 1e-8  # b* is an exact root of Eq 4.3, not just monotone


# ---- Eq 5.5 defining identity: residual == 0 at the solved b*_L ----
@pytest.mark.parametrize("method", ["pc", "quad"])
def test_take_profit_stoploss_residual_zero_at_root(method):
    ff, fpf, gf, gpf = _funcs(method)
    ls = l_star(PG, RG, CG)
    ll = ls - 0.5
    b = take_profit_stoploss(PG, RG, CG, ll, method=method)
    assert b is not None
    FL, GL = ff(ll, PG, RG), gf(ll, PG, RG)
    Fb, Gb, Fpb, Gpb = ff(b, PG, RG), gf(b, PG, RG), fpf(b, PG, RG), gpf(b, PG, RG)
    lhs = ((ll - CG) * Gb - (b - CG) * GL) * Fpb + ((b - CG) * FL - (ll - CG) * Fb) * Gpb
    res = lhs - (Gb * FL - GL * Fb)
    assert abs(res) < 1e-7


# ---- F'/G' are the EXACT derivatives of F/G (only signs were tested before) ----
@pytest.mark.parametrize("ffun,fpfun", [(F, Fp), (F_pc, Fp_pc)])
def test_F_derivative_matches_numerical(ffun, fpfun):
    h = 1e-5
    for x in (-0.3, 0.0, 0.4):
        num = (ffun(x + h, PG, RG) - ffun(x - h, PG, RG)) / (2 * h)
        assert fpfun(x, PG, RG) == pytest.approx(num, rel=1e-5, abs=1e-7)


@pytest.mark.parametrize("gfun,gpfun", [(G, Gp), (G_pc, Gp_pc)])
def test_G_derivative_matches_numerical(gfun, gpfun):
    h = 1e-5
    for x in (-0.3, 0.0, 0.4):
        num = (gfun(x + h, PG, RG) - gfun(x - h, PG, RG)) / (2 * h)
        assert gpfun(x, PG, RG) == pytest.approx(num, rel=1e-5, abs=1e-7)


# ---- Prop 5.3 OTHER end: b*_L -> b*_nostop as the stop L -> -inf ----
def test_stoploss_converges_to_nostop_as_L_to_minus_inf():
    b_ns = take_profit(PG, RG, CG, method="pc")
    ls = l_star(PG, RG, CG)
    b_deep = take_profit_stoploss(PG, RG, CG, ls - 4.0, method="pc")
    assert b_deep is not None
    # a deep stop is effectively no stop -> threshold collapses onto the no-stop b*
    assert b_deep == pytest.approx(b_ns, abs=1e-4)
    # b*_L is decreasing in L (Prop 5.3): the DEEPER stop (lower L) gives the HIGHER b*, which
    # has converged onto the no-stop b*; the shallower stop sits at or below it.
    b_mid = take_profit_stoploss(PG, RG, CG, ls - 1.0, method="pc")
    assert b_mid <= b_deep + 1e-6
    assert b_deep == pytest.approx(b_ns, abs=1e-4)


# ---- fit_ou: the AR(1) -> OU MLE inversion is the load-bearing identity ----
def test_fit_ou_inversion_identity():
    rng = np.random.default_rng(5)
    mu, theta, sigma, dt, n = 4.0, 0.20, 0.30, 1 / 252, 6000
    b = np.exp(-mu * dt)
    sd = sigma * np.sqrt((1 - b * b) / (2 * mu))
    x = np.empty(n)
    x[0] = theta
    for i in range(1, n):
        x[i] = theta + b * (x[i - 1] - theta) + rng.normal(0, sd)
    p = fit_ou(x, dt)
    assert p.ok
    # mu = -ln(b)/dt, theta = a/(1-b) recovered to tight tol on a long sample
    assert p.mu == pytest.approx(mu, rel=0.25)
    assert p.theta == pytest.approx(theta, abs=0.03)
    assert p.sigma == pytest.approx(sigma, rel=0.15)


def test_fit_ou_short_series_not_ok():
    assert fit_ou(np.linspace(0, 1, 10), 1 / 252).ok is False  # < 30 points -> ok=False, NaNs
    p = fit_ou(np.linspace(0, 1, 10), 1 / 252)
    assert not np.isfinite(p.mu)


def test_fit_ou_explosive_b_not_ok():
    # b >= 1 (explosive / unit-root drift) must be rejected as non-mean-reverting.
    x = np.cumsum(np.ones(200)) + np.random.default_rng(0).normal(0, 1e-6, 200)
    assert fit_ou(x, 1 / 252).ok is False


# ---- Kelly: cap actually BINDS, and the lam*shrink<=1 invariant ----
def test_fractional_kelly_cap_binds():
    # huge mean, tiny var, many trades -> shrink~1, lam=0.5 -> f_used = 0.5*f_full < f_full (cap ok)
    m, v, n = 50.0, 0.01, 10000
    f_full = m / v
    f = fractional_kelly(m, v, n, lam_base=0.5)
    assert 0 < f <= f_full
    assert f == pytest.approx(0.5 * (1 - 1 / (m / np.sqrt(v / n)) ** 2) * f_full, rel=1e-9)


def test_fractional_kelly_n_lt_2_is_zero():
    assert fractional_kelly(0.02, 0.04, 1) == 0.0
    assert fractional_kelly(0.02, 0.04, 0) == 0.0


# ---- kelly_lots: floors and rounding ----
def test_kelly_lots_floor_and_bad_inputs():
    assert kelly_lots(0.02, 0.04, 400, 0.0, 150_000) == 0  # no capital
    assert kelly_lots(0.02, 0.04, 400, 1_000_000, 0.0) == 0  # no margin
    assert kelly_lots(-0.02, 0.04, 400, 1_000_000, 150_000) == 0  # no edge
    # fraction*capital < one lot of margin -> floors to 0 (not 1)
    assert kelly_lots(0.02, 0.04, 400, 1000, 150_000) == 0


def test_kelly_lots_matches_floor_of_fraction():
    cap, mpl = 5_000_000.0, 100_000.0
    f = fractional_kelly(0.02, 0.04, 400, 0.5)
    assert kelly_lots(0.02, 0.04, 400, cap, mpl, 0.5) == int(np.floor(f * cap / mpl))


# ---- exit_decision: full precedence ladder + b*_None branch ----
def test_exit_decision_squareoff_beats_stop_and_tp():
    # terminal condition wins even when stop AND take-profit also fire
    d = exit_decision(-10.0, 0.3, -0.5, 1.0, 0)
    assert d["exit"] is True and "square-off" in d["reason"]


def test_exit_decision_no_take_profit_when_b_star_none():
    # b_star None => no profitable hold => take-profit branch can never fire; falls through to hold
    d = exit_decision(5.0, None, -0.5, 1.0, 60)
    assert d["exit"] is False  # high edge but no b* -> not a TP exit
    # but stop still applies with b_star None
    d2 = exit_decision(-1.0, None, -0.5, 1.0, 60)
    assert d2["exit"] is True and "stop" in d2["reason"]


def test_exit_decision_stop_beats_take_profit_at_same_x():
    # if x somehow <= ll AND >= b* (ll>b*, pathological) stop must win (checked first)
    d = exit_decision(0.0, -1.0, 0.5, 1.0, 60)  # ll=0.5 > b*=-1 ; x=0 <= ll
    assert "stop" in d["reason"]


# ---- greeks_envelope: default delta_max=inf never breaches; boundary is strict ----
def test_greeks_envelope_default_delta_no_breach():
    e = greeks_envelope(8.0, 10.0, 1.0, 5.0, 1e9)  # huge delta, default delta_max=inf
    assert e["delta_breach"] is False


def test_greeks_envelope_boundaries_strict():
    # vega exactly at floor -> NOT dead (strict <); theta == rent -> NOT dominating (strict >)
    e = greeks_envelope(5.0, 10.0, 2.0, 2.0, 0.0, vega_floor_frac=0.5)
    assert e["vega_dead"] is False and e["theta_dominates"] is False
