"""Tests for ratio-backspread construction + payoff (synthetic Black-76 chain; module stays BS-free)."""

from __future__ import annotations

import math

import pytest

from volarb.strategy.backspread import (
    BackspreadParams,
    construct_backspread,
    max_loss_rupees,
    payoff,
)


def _b76(forward, strike, t, vol, is_call, r=0.065):
    df = math.exp(-r * t)
    srt = vol * math.sqrt(t)
    d1 = (math.log(forward / strike) + 0.5 * vol * vol * t) / srt
    d2 = d1 - srt
    def _n(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    return df * (forward * _n(d1) - strike * _n(d2)) if is_call else df * (strike * _n(-d2) - forward * _n(-d1))


def _chain(forward, t, vol, is_call, lo=-600, hi=600, step=50):
    return [(forward + k, _b76(forward, forward + k, t, vol, is_call)) for k in range(lo, hi + 1, step)]


def test_call_backspread_is_convex_and_longs_first():
    F, t, vol = 25000.0, 5 / 365, 0.13
    bs = construct_backspread("call", _chain(F, t, vol, True), F, t)
    assert bs is not None
    assert bs.n_long > bs.n_short and bs.long_strike > bs.short_strike       # OTM-up longs, ratio>1
    assert bs.net_gamma > 0 and bs.net_vega > 0                              # convexity HARD gate
    assert bs.net_delta > 0                                                  # directional (up)
    assert bs.legs[0].qty > 0 and bs.legs[1].qty < 0                         # LONGS FIRST, then short


def test_payoff_regions_and_max_loss_identity():
    F, t, vol = 25000.0, 5 / 365, 0.13
    bs = construct_backspread("call", _chain(F, t, vol, True), F, t)
    assert bs is not None
    lot = 75
    # wrong-way (far below short) -> lose the (small) debit
    assert payoff(bs, bs.short_strike - 2000, lot) == pytest.approx(-bs.debit * lot, abs=1e-6)
    # worst at the long strike = -max_loss
    assert payoff(bs, bs.long_strike, lot) == pytest.approx(-bs.max_loss * lot, abs=1e-6)
    # convex far above the long strike -> strongly positive
    assert payoff(bs, bs.long_strike + 3000, lot) > 0
    # max-loss identity
    assert bs.max_loss == pytest.approx(bs.debit + bs.n_short * abs(bs.long_strike - bs.short_strike))
    assert max_loss_rupees(bs, lot) == pytest.approx(max(bs.max_loss, 0.0) * lot)


def test_put_backspread_mirror():
    F, t, vol = 25000.0, 5 / 365, 0.13
    bs = construct_backspread("put", _chain(F, t, vol, False), F, t)
    assert bs is not None
    assert bs.long_strike < bs.short_strike and bs.net_delta < 0             # OTM-down longs, directional down
    assert bs.net_gamma > 0 and bs.net_vega > 0


def test_abstain_when_no_convex_wing():
    # only ATM + far-OTM (delta < delta_long_min) -> no valid long wing -> abstain
    F, t, vol = 25000.0, 1 / 365, 0.10   # very short dated; far wings worthless
    chain = [(F, _b76(F, F, t, vol, True)), (F + 1500, _b76(F, F + 1500, t, vol, True))]
    assert construct_backspread("call", chain, F, t, BackspreadParams(delta_long_min=0.25)) is None
