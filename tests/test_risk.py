"""Tests for the trimmed risk primitives: fractional_kelly, kelly_lots, kill_switch, greeks_envelope."""

from __future__ import annotations


from volarb.strategy.risk import fractional_kelly, greeks_envelope, kelly_lots, kill_switch


def test_fractional_kelly_no_edge_and_shrink():
    assert fractional_kelly(0.05, 0.0, 50) == 0.0          # var<=0 -> 0
    assert fractional_kelly(0.05, 0.001, 1) == 0.0         # n<2 -> 0
    assert fractional_kelly(-0.05, 0.001, 50) == 0.0       # negative edge -> 0
    f = fractional_kelly(0.05, 0.001, 50, lam_base=0.5)    # strong t-stat -> positive, <= full
    assert 0.0 < f <= 0.05 / 0.001
    # weak t-stat (t<1) -> shrink kills it
    assert fractional_kelly(0.001, 0.01, 50) == 0.0


def test_kelly_lots_floor_and_guards():
    assert kelly_lots(0.05, 0.001, 50, risk_capital=1e6, margin_per_lot=50_000) >= 1
    assert kelly_lots(0.05, 0.001, 50, risk_capital=1e6, margin_per_lot=0) == 0
    assert kelly_lots(-0.05, 0.001, 50, risk_capital=1e6, margin_per_lot=50_000) == 0


def test_kill_switch_boundary():
    assert kill_switch(-50_000.0, 50_000) is True          # boundary triggers (<=)
    assert kill_switch(-49_999.0, 50_000) is False
    assert kill_switch(10_000.0, 50_000) is False


def test_greeks_envelope_flags():
    e = greeks_envelope(3.0, 10.0, 5.0, 2.0, 0.6, vega_floor_frac=0.5, delta_max=0.5)
    assert e["vega_dead"] and e["theta_dominates"] and e["delta_breach"]
    ok = greeks_envelope(8.0, 10.0, 1.0, 5.0, 0.1, vega_floor_frac=0.5, delta_max=0.5)
    assert not (ok["vega_dead"] or ok["theta_dominates"] or ok["delta_breach"])
