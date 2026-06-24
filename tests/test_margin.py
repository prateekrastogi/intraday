"""Tests for the broker-aligned backspread margin (max-loss based; the hedge lowers margin)."""

from __future__ import annotations

import math

import pytest

from volarb.strategy.backspread import construct_backspread, max_loss_rupees
from volarb.strategy.margin import incremental_margin, structure_margin


def _b76(forward, strike, t, vol, is_call, r=0.065):
    df = math.exp(-r * t)
    srt = vol * math.sqrt(t)
    d1 = (math.log(forward / strike) + 0.5 * vol * vol * t) / srt
    d2 = d1 - srt
    def _n(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    return df * (forward * _n(d1) - strike * _n(d2)) if is_call else df * (strike * _n(-d2) - forward * _n(-d1))


def _bs(F=25000.0, t=5 / 365, vol=0.13):
    chain = [(F + k, _b76(F, F + k, t, vol, True)) for k in range(-600, 601, 50)]
    return construct_backspread("call", chain, F, t), F


def test_margin_is_max_loss_plus_exposure():
    bs, F = _bs()
    lot = 75
    assert structure_margin(bs, lot, F) == pytest.approx(max_loss_rupees(bs, lot))          # exposure_frac=0
    with_exp = structure_margin(bs, lot, F, exposure_frac=0.02)
    assert with_exp == pytest.approx(max_loss_rupees(bs, lot) + 0.02 * bs.n_short * F * lot)


def test_hedge_makes_margin_a_small_fraction_of_short_notional():
    bs, F = _bs()
    lot = 75
    short_notional = bs.n_short * F * lot
    # the defined-risk hedge: margin (= max-loss) is a tiny fraction of the short's notional
    # (a naked short would be margined off that notional/SPAN) -> "a sell lowers margin"
    assert structure_margin(bs, lot, F) < 0.05 * short_notional


def test_incremental_margin_nonneg():
    assert incremental_margin(100_000.0, 130_000.0) == pytest.approx(30_000.0)
    assert incremental_margin(130_000.0, 120_000.0) == 0.0   # offsetting structure -> floored at 0
