"""Synthetic tests for the long-vol entry/exit gate.

No real lake, no creds, no network. Uses the REAL volarb.costs.nse_fees foundation (no stub)
so the cost-hurdle math is checked end-to-end against the actual NSE round-trip charges.
"""

from __future__ import annotations

import datetime as dt

import pytest

from volarb.costs.nse_fees import round_trip_charges
from volarb.strategy.vol_gate import GateState, cost_hurdle, entry_decision, exit_decision

TRADE_DATE = dt.date(2026, 6, 23)


# --------------------------------------------------------------------------------- #
# cost_hurdle  (single unit: annualized vol; vega required)
# --------------------------------------------------------------------------------- #
def test_cost_hurdle_requires_vega():
    with pytest.raises(TypeError):
        cost_hurdle(premium=120.0, qty=75, trade_date=TRADE_DATE)  # vega is required


def test_cost_hurdle_vol_units_inverts_vega():
    from volarb.strategy.vol_gate import VRP_CUSHION

    premium, qty, vega = 120.0, 75, 8.0
    expected = (
        VRP_CUSHION * round_trip_charges(premium, premium, qty, TRADE_DATE).total / (vega * qty)
    )
    h = cost_hurdle(premium, qty, TRADE_DATE, vega=vega)
    assert h == pytest.approx(expected) and h > 0


def test_cost_hurdle_rejects_bad_inputs():
    with pytest.raises(ValueError):
        cost_hurdle(premium=0.0, qty=75, trade_date=TRADE_DATE, vega=8.0)
    with pytest.raises(ValueError):
        cost_hurdle(premium=120.0, qty=0, trade_date=TRADE_DATE, vega=8.0)
    with pytest.raises(ValueError):
        cost_hurdle(120.0, 75, TRADE_DATE, vega=0.0)


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


# --------------------------------------------------------------------------------- #
# GAP-FILLING TESTS (independent verification pass)
# --------------------------------------------------------------------------------- #
from volarb.strategy.vol_gate import (  # noqa: E402
    IV_PCT_FLOOR,
    SLOPE_INVERSION,
    STRANGLE_IV_PCT,
    VRP_CUSHION,
    structure_cost_hurdle,
)
from volarb.costs.nse_fees import structure_round_trip_charges  # noqa: E402


# --- structure_cost_hurdle: previously UNTESTED end-to-end ---------------------- #
def test_structure_hurdle_symmetric_straddle_equals_per_leg():
    """Docstring identity: a symmetric ATM straddle's structure hurdle ~= the per-leg hurdle.

    Brokerage is flat-per-order (2 orders single leg, 4 orders for the 2-leg structure = exactly
    2x), STT/exch/sebi/gst all scale with premium*qty, and total_vega doubles -- so the ratio is
    exactly 1 here, not merely approximate.
    """
    premium, qty, leg_vega = 120.0, 75, 8.0
    single = cost_hurdle(premium, qty, TRADE_DATE, vega=leg_vega)
    legs = [(premium, premium), (premium, premium)]
    struct = structure_cost_hurdle(legs, qty, TRADE_DATE, total_vega=2 * leg_vega)
    assert struct == pytest.approx(single)


def test_structure_hurdle_matches_explicit_formula():
    legs = [(150.0, 150.0), (140.0, 140.0)]
    qty, total_vega = 75, 17.0
    expected = (
        VRP_CUSHION * structure_round_trip_charges(legs, qty, TRADE_DATE).total / (total_vega * qty)
    )
    assert structure_cost_hurdle(legs, qty, TRADE_DATE, total_vega=total_vega) == pytest.approx(
        expected
    )


def test_structure_hurdle_asymmetric_strangle_differs_from_per_leg():
    """An asymmetric strangle (unequal premia/vega) does NOT collapse to the per-leg hurdle."""
    legs = [(80.0, 80.0), (40.0, 40.0)]  # call wing dearer than put wing
    qty, total_vega = 75, 11.0
    struct = structure_cost_hurdle(legs, qty, TRADE_DATE, total_vega=total_vega)
    per_leg_call = cost_hurdle(80.0, qty, TRADE_DATE, vega=total_vega / 2)
    assert struct != pytest.approx(per_leg_call)
    assert 0.0 < struct < 1.0


def test_structure_hurdle_rejects_bad_inputs():
    with pytest.raises(ValueError):
        structure_cost_hurdle([(80.0, 80.0)], qty=0, trade_date=TRADE_DATE, total_vega=8.0)
    with pytest.raises(ValueError):
        structure_cost_hurdle([(80.0, 80.0)], qty=75, trade_date=TRADE_DATE, total_vega=0.0)
    with pytest.raises(ValueError):
        structure_cost_hurdle([(80.0, 80.0)], qty=75, trade_date=TRADE_DATE, total_vega=-1.0)
    with pytest.raises(ValueError):  # empty structure (propagated from nse_fees)
        structure_cost_hurdle([], qty=75, trade_date=TRADE_DATE, total_vega=8.0)


def test_structure_hurdle_cushion_scales_linearly():
    legs = [(150.0, 150.0), (140.0, 140.0)]
    h1 = structure_cost_hurdle(legs, 75, TRADE_DATE, total_vega=17.0, cushion=1.0)
    h3 = structure_cost_hurdle(legs, 75, TRADE_DATE, total_vega=17.0, cushion=3.0)
    assert h3 == pytest.approx(3.0 * h1)


# --- entry_decision boundaries ------------------------------------------------- #
def test_entry_spread_equal_to_hurdle_does_not_enter():
    """`<=` guard: spread exactly at the hurdle is NOT an edge (must strictly beat)."""
    out = entry_decision(spread=0.012, regime=_favorable_regime(), hurdle=0.012)
    assert out["enter"] is False


def test_structure_selection_boundary_at_strangle_threshold():
    """iv_pct == STRANGLE_IV_PCT picks strangle (<=); just above picks straddle."""
    at = entry_decision(0.05, _favorable_regime(iv_pct=STRANGLE_IV_PCT), 0.012)
    above = entry_decision(0.05, _favorable_regime(iv_pct=STRANGLE_IV_PCT + 1e-9), 0.012)
    assert at["structure"] == "strangle"
    assert above["structure"] == "straddle"


def test_structure_populated_on_nogo():
    """`structure` is the structure we WOULD trade, populated even when we decline."""
    out = entry_decision(spread=0.001, regime=_favorable_regime(iv_pct=0.10), hurdle=0.012)
    assert out["enter"] is False
    assert out["structure"] == "strangle"  # iv_pct 0.10 <= STRANGLE_IV_PCT


def test_cheap_boundary_inclusive_and_slope_zero_not_inverted():
    """iv_pct == IV_PCT_FLOOR is cheap (<=); slope == SLOPE_INVERSION is NOT inverted (strict <)."""
    # cheap at the floor (slope flat, no jump) -> enters on the cheap leg alone
    cheap_at = {"iv_pct": IV_PCT_FLOOR, "slope": 0.01, "jump_flag": False}
    assert entry_decision(0.05, cheap_at, 0.012)["enter"] is True
    above_floor = {"iv_pct": IV_PCT_FLOOR + 1e-9, "slope": 0.01, "jump_flag": False}
    assert entry_decision(0.05, above_floor, 0.012)["enter"] is False
    # slope exactly at the inversion boundary is flat, not inverted -> rich iv vetoes
    flat_rich = {"iv_pct": 0.80, "slope": SLOPE_INVERSION, "jump_flag": False}
    assert entry_decision(0.05, flat_rich, 0.012)["enter"] is False


def test_inverted_term_structure_opens_gate_even_when_rich():
    """Inversion alone is sufficient: a rich-but-inverted curve still passes the regime gate."""
    inv_rich = {"iv_pct": 0.95, "slope": -0.02, "jump_flag": False}
    out = entry_decision(0.05, inv_rich, 0.012)
    assert out["enter"] is True
    assert "inverted term structure" in out["reason"]


def test_jump_relaxed_band_upper_boundary():
    """Jump widens the cheap band to IV_PCT_FLOOR+0.10 (inclusive); above it, jump cannot save."""
    edge = IV_PCT_FLOOR + 0.10
    at = {"iv_pct": edge, "slope": 0.01, "jump_flag": True}
    above = {"iv_pct": edge + 1e-9, "slope": 0.01, "jump_flag": True}
    assert entry_decision(0.05, at, 0.012)["enter"] is True
    assert entry_decision(0.05, above, 0.012)["enter"] is False


# --- exit_decision precedence + boundaries ------------------------------------- #
def test_timer_takes_precedence_over_simultaneous_stop_and_collapse():
    out = exit_decision(
        _holding_state(minutes_to_square_off=0, pnl_fraction=-0.9, current_spread=0.0)
    )
    assert out["exit"] is True
    assert "square-off" in out["reason"]


def test_stop_takes_precedence_over_collapse():
    out = exit_decision(_holding_state(pnl_fraction=-0.5, current_spread=0.0))
    assert out["exit"] is True
    assert "stop-loss" in out["reason"]


def test_stop_loss_exact_boundary_triggers():
    """`<=` guard: pnl exactly at -stop_loss_fraction fires the stop."""
    out = exit_decision(_holding_state(pnl_fraction=-0.40, stop_loss_fraction=0.40))
    assert out["exit"] is True
    assert "stop-loss" in out["reason"]


def test_stop_loss_uses_abs_of_fraction():
    """A negative stop_loss_fraction is normalised by abs() -- same trigger as positive."""
    out = exit_decision(_holding_state(pnl_fraction=-0.45, stop_loss_fraction=-0.40))
    assert out["exit"] is True
    assert "stop-loss" in out["reason"]


def test_spread_collapse_exact_boundary_holds():
    """Strict `<`: current_spread exactly at the collapse level is NOT a collapse."""
    state = _holding_state(
        entry_spread=0.05, current_spread=0.30 * 0.05, spread_collapse_ratio=0.30, pnl_fraction=0.0
    )
    out = exit_decision(state)
    assert out["exit"] is False


def test_nonpositive_entry_spread_skips_collapse_branch():
    """entry_spread <= 0 means there was no edge to collapse -> collapse never fires."""
    assert (
        exit_decision(_holding_state(entry_spread=0.0, current_spread=-1.0, pnl_fraction=0.0))[
            "exit"
        ]
        is False
    )
    assert (
        exit_decision(_holding_state(entry_spread=-0.05, current_spread=-1.0, pnl_fraction=0.0))[
            "exit"
        ]
        is False
    )
