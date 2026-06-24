"""Tests for backspread sizing: min(Kelly-on-max-loss, margin budget, aggregate-risk cap)."""

from __future__ import annotations

from volarb.strategy.sizing import max_structures, size_structures


def test_max_structures():
    assert max_structures(100_000.0, 4_000.0) == 25
    assert max_structures(100_000.0, 0.0) == 0


def test_size_min_of_caps_and_zero_on_no_edge():
    # strong edge; margin budget the binding cap: 50000/4000 = 12
    n = size_structures(0.05, 0.001, 50, risk_capital=1e6, max_loss_per_structure=4_000.0,
                        margin_per_structure=4_000.0, margin_budget=50_000.0,
                        lam_kelly=0.5, aggregate_maxloss_frac=0.50)
    assert n == 12
    # no edge -> 0
    assert size_structures(-0.05, 0.001, 50, risk_capital=1e6, max_loss_per_structure=4_000.0,
                          margin_per_structure=4_000.0, margin_budget=50_000.0) == 0
    # degenerate max-loss -> 0
    assert size_structures(0.05, 0.001, 50, risk_capital=1e6, max_loss_per_structure=0.0,
                          margin_per_structure=4_000.0, margin_budget=50_000.0) == 0
    # aggregate-risk cap binds: 0.10*1e6/4000 = 25 < margin(50000/4000=12)? no -> agg 25 vs margin 12 -> 12
    n2 = size_structures(0.05, 0.001, 50, risk_capital=1e6, max_loss_per_structure=4_000.0,
                         margin_per_structure=4_000.0, margin_budget=1e9, lam_kelly=0.5,
                         aggregate_maxloss_frac=0.02)
    assert n2 == 5   # 0.02*1e6/4000 = 5 (aggregate cap binds)
