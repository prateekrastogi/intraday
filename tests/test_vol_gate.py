"""Synthetic tests for the long-vol entry/exit gate.

No real lake, no creds, no network. The cost foundation (`volarb.costs.nse_fees`) is
stubbed with a deterministic ~0.8%-of-premium round-trip so the math is checkable; the
stub matches the documented signature
`round_trip_charges(premium_in, premium_out, qty, trade_date) -> Charges(.total)`.
"""

from __future__ import annotations

import datetime as dt
import sys
import types
from dataclasses import dataclass

import pytest

# --- stub the cost foundation BEFORE importing the module under test. -------------
_FEE_RATE = 0.008  # ~0.8% of round-trip premium notional on an ATM NIFTY option.


@dataclass(frozen=True)
class _Charges:
    total: float


def _round_trip_charges(premium_in, premium_out, qty, trade_date):
    return _Charges(total=_FEE_RATE * (premium_in + premium_out) * qty / 2.0)


_costs_pkg = types.ModuleType("volarb.costs")
_nse = types.ModuleType("volarb.costs.nse_fees")
_nse.round_trip_charges = _round_trip_charges
_nse.Charges = _Charges
sys.modules.setdefault("volarb.costs", _costs_pkg)
sys.modules["volarb.costs.nse_fees"] = _nse

from volarb.strategy.vol_gate import (  # noqa: E402
    GateState,
    cost_hurdle,
    entry_decision,
    exit_decision,
)

TRADE_DATE = dt.date(2026, 6, 23)


# --------------------------------------------------------------------------------- #
# cost_hurdle
# --------------------------------------------------------------------------------- #
def test_cost_hurdle_premium_fraction_matches_fee_times_cushion():
    from volarb.strategy.vol_gate import VRP_CUSHION

    h = cost_hurdle(premium=120.0, qty=75, trade_date=TRADE_DATE)
    assert h == pytest.approx(VRP_CUSHION * _FEE_RATE)


def test_cost_hurdle_vol_units_inverts_vega():
    from volarb.strategy.vol_gate import VRP_CUSHION

    premium, qty, vega = 120.0, 75, 8.0
    charges_total = _FEE_RATE * premium * qty
    expected = VRP_CUSHION * charges_total / (vega * qty)
    h = cost_hurdle(premium, qty, TRADE_DATE, vega=vega, t_years=0.001)
    assert h == pytest.approx(expected)


def test_cost_hurdle_rejects_bad_inputs():
    with pytest.raises(ValueError):
        cost_hurdle(premium=0.0, qty=75, trade_date=TRADE_DATE)
    with pytest.raises(ValueError):
        cost_hurdle(premium=120.0, qty=0, trade_date=TRADE_DATE)
    with pytest.raises(ValueError):
        cost_hurdle(120.0, 75, TRADE_DATE, vega=0.0, t_years=0.001)


# --------------------------------------------------------------------------------- #
# entry_decision
# --------------------------------------------------------------------------------- #
def _favorable_regime(**kw):
    base = {"iv_pct": 0.25, "slope": -0.01, "jump_flag": False}
    base.update(kw)
    return base


def test_spread_below_hurdle_no_entry():
    out = entry_decision(spread=0.01, regime=_favorable_regime(), hurdle=0.012)
    assert out["enter"] is False
    assert "hurdle" in out["reason"]


def test_spread_above_hurdle_favorable_regime_enters_straddle():
    out = entry_decision(spread=0.05, regime=_favorable_regime(iv_pct=0.25), hurdle=0.012)
    assert out["enter"] is True
    assert out["structure"] == "straddle"
    assert "magnitude" in out["reason"]


def test_very_cheap_vol_selects_strangle():
    out = entry_decision(spread=0.05, regime=_favorable_regime(iv_pct=0.10), hurdle=0.012)
    assert out["enter"] is True
    assert out["structure"] == "strangle"


def test_rich_regime_blocks_entry_even_with_edge():
    rich = {"iv_pct": 0.80, "slope": 0.02, "jump_flag": False}
    out = entry_decision(spread=0.05, regime=rich, hurdle=0.012)
    assert out["enter"] is False
    assert "rich" in out["reason"]


def test_jump_relaxes_cheap_band_but_is_not_a_trigger():
    near = {"iv_pct": 0.35, "slope": 0.01}
    assert entry_decision(0.05, {**near, "jump_flag": True}, 0.012)["enter"] is True
    assert entry_decision(0.05, {**near, "jump_flag": False}, 0.012)["enter"] is False


def test_incomplete_regime_blocks_entry():
    out = entry_decision(spread=0.05, regime={"iv_pct": 0.2}, hurdle=0.012)
    assert out["enter"] is False
    assert "incomplete" in out["reason"]


# --------------------------------------------------------------------------------- #
# exit_decision
# --------------------------------------------------------------------------------- #
def _holding_state(**kw):
    base = dict(
        minutes_held=30,
        minutes_to_square_off=60,
        entry_spread=0.05,
        current_spread=0.045,
        pnl_fraction=0.05,
    )
    base.update(kw)
    return GateState(**base)


def test_square_off_timer_forces_exit():
    out = exit_decision(_holding_state(minutes_to_square_off=0))
    assert out["exit"] is True
    assert "square-off" in out["reason"]


def test_square_off_timer_takes_precedence_over_healthy_position():
    out = exit_decision(
        _holding_state(minutes_to_square_off=-1, pnl_fraction=0.20, current_spread=0.05)
    )
    assert out["exit"] is True
    assert "square-off" in out["reason"]


def test_stop_loss_triggers():
    out = exit_decision(_holding_state(pnl_fraction=-0.45, stop_loss_fraction=0.40))
    assert out["exit"] is True
    assert "stop-loss" in out["reason"]


def test_spread_collapse_triggers():
    out = exit_decision(
        _holding_state(current_spread=0.01, spread_collapse_ratio=0.30, pnl_fraction=0.0)
    )
    assert out["exit"] is True
    assert "collapse" in out["reason"]


def test_healthy_position_holds():
    out = exit_decision(_holding_state())
    assert out["exit"] is False
    assert "hold" in out["reason"]
