"""Tests for jump-continuation entry/exit (synthetic returns; no-lookahead)."""

from __future__ import annotations

import numpy as np

from volarb.strategy.jump_entry import JumpEntryParams, confirmed_jump, entry_decision, exit_decision


def _calm_with_jump(n=250, sign=+1, seed=0):
    r = np.random.default_rng(seed).normal(0.0, 1e-4, n)
    r[-3:] = sign * np.array([1.5e-3, 2.0e-3, 2.5e-3])   # a sustained jump+continuation at the end
    return r


def test_confirmed_up_jump_maps_to_call():
    jumped, direction = confirmed_jump(_calm_with_jump(sign=+1))
    assert jumped is True and direction == "call"


def test_confirmed_down_jump_maps_to_put():
    jumped, direction = confirmed_jump(_calm_with_jump(sign=-1))
    assert jumped is True and direction == "put"


def test_calm_series_no_jump():
    r = np.random.default_rng(1).normal(0.0, 1e-4, 250)
    assert confirmed_jump(r) == (False, None)


def test_entry_guards():
    r = _calm_with_jump(sign=+1)
    p = JumpEntryParams(cooldown_bars=15, min_dte_days=2, min_hold_minutes=30)
    ok = entry_decision(r, in_position=False, bars_since_last_action=99, dte_days=5, minutes_to_squareoff=120, params=p)
    assert ok["enter"] is True and ok["direction"] == "call"
    assert entry_decision(r, in_position=True, bars_since_last_action=99, dte_days=5, minutes_to_squareoff=120, params=p)["enter"] is False
    assert entry_decision(r, in_position=False, bars_since_last_action=3, dte_days=5, minutes_to_squareoff=120, params=p)["enter"] is False
    assert entry_decision(r, in_position=False, bars_since_last_action=99, dte_days=1, minutes_to_squareoff=120, params=p)["enter"] is False
    assert entry_decision(r, in_position=False, bars_since_last_action=99, dte_days=5, minutes_to_squareoff=10, params=p)["enter"] is False


def test_exit_precedence_and_no_price_stop():
    assert exit_decision(0, 0.1)["exit"] is True                                   # square-off
    assert exit_decision(60, 0.5, take_profit_r=0.3)["exit"] is True               # take-profit
    assert exit_decision(60, -0.1, take_profit_r=0.3)["exit"] is False             # below TP -> hold
    assert exit_decision(60, 0.0, reversed_into_middle=True, thesis_invalidated=True)["exit"] is True
    assert exit_decision(60, 0.0, greeks_flags={"theta_dominates": True})["exit"] is True
    # NO price stop: a deep adverse mark with no reversal/thesis-invalidation/drive-decay -> HOLD
    assert exit_decision(60, -0.8, reversed_into_middle=False)["exit"] is False
