## tests/signals/test_surface.py

"""Synthetic-data tests for volarb.signals.surface (Johnson 2017 SLOPE gating).

No real lake, creds, or network: all IVs are hand-constructed numbers so the assertions
pin the MATH (slope sign, empirical percentile rank, gate truth table).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from volarb.signals.surface import iv_percentile, johnson_gate, term_slope


# --- term_slope: sign convention (next - front) ----------------------------------------


def test_term_slope_inverted_is_negative() -> None:
    # front richer than next => backwardation => negative slope (stress).
    assert term_slope(atm_iv_front=0.30, atm_iv_next=0.22) == pytest.approx(-0.08)


def test_term_slope_contango_is_positive() -> None:
    # normal upward-sloping term structure => positive slope.
    assert term_slope(atm_iv_front=0.18, atm_iv_next=0.25) == pytest.approx(0.07)


def test_term_slope_flat_is_zero() -> None:
    assert term_slope(0.20, 0.20) == pytest.approx(0.0)


# --- iv_percentile: empirical rank correctness -----------------------------------------


def test_iv_percentile_rank_known_values() -> None:
    hist = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26]
    # now = 0.18 -> values <= 0.18 in population {hist + 0.18} = 6 of 10 = 0.6.
    assert iv_percentile(0.18, hist) == pytest.approx(0.6)


def test_iv_percentile_max_is_one() -> None:
    assert iv_percentile(0.50, [0.10, 0.12, 0.14]) == pytest.approx(1.0)


def test_iv_percentile_min_is_low() -> None:
    # now below all history => only itself is <= now => 1 of 4 = 0.25.
    assert iv_percentile(0.05, [0.10, 0.12, 0.14]) == pytest.approx(0.25)


def test_iv_percentile_always_in_unit_interval() -> None:
    rng = np.random.default_rng(0)
    hist = rng.uniform(0.05, 0.40, size=250)
    for now in rng.uniform(0.0, 0.50, size=50):
        assert 0.0 <= iv_percentile(float(now), hist) <= 1.0


def test_iv_percentile_ignores_nans_in_history() -> None:
    hist = [0.10, float("nan"), 0.20, math.inf]
    # finite history = {0.10, 0.20}; now = 0.20 -> 3 of 3 <= 0.20 = 1.0.
    assert iv_percentile(0.20, hist) == pytest.approx(1.0)


def test_iv_percentile_empty_history_is_nan() -> None:
    assert math.isnan(iv_percentile(0.20, []))
    assert math.isnan(iv_percentile(0.20, [float("nan")]))


# --- johnson_gate: truth table ---------------------------------------------------------


def test_gate_inverted_slope_opens() -> None:
    assert johnson_gate(slope=-0.05, iv_pct=0.90) is True


def test_gate_low_percentile_opens() -> None:
    assert johnson_gate(slope=0.04, iv_pct=0.10) is True


def test_gate_high_percentile_upward_slope_closes() -> None:
    assert johnson_gate(slope=0.04, iv_pct=0.90) is False


def test_gate_boundaries_are_strict() -> None:
    assert johnson_gate(slope=0.0, iv_pct=0.25) is False
    assert johnson_gate(slope=-1e-9, iv_pct=0.25) is True
    assert johnson_gate(slope=0.0, iv_pct=0.25 - 1e-9) is True


def test_gate_custom_thresholds() -> None:
    assert johnson_gate(slope=0.01, iv_pct=0.90, slope_inverted_below=0.02) is True
    assert johnson_gate(slope=0.30, iv_pct=0.40, iv_pct_low_below=0.50) is True


def test_gate_nan_inputs_do_not_open() -> None:
    assert johnson_gate(slope=float("nan"), iv_pct=float("nan")) is False
    assert johnson_gate(slope=float("nan"), iv_pct=0.10) is True
    assert johnson_gate(slope=-0.05, iv_pct=float("nan")) is True
