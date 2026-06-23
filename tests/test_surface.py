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


# --- iv_percentile: empirical rank correctness (self excluded, strict <) ----------------


def test_iv_percentile_rank_known_values() -> None:
    hist = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26]
    # now = 0.18 -> hist strictly below 0.18 = {.10,.12,.14,.16} = 4 of 9.
    assert iv_percentile(0.18, hist) == pytest.approx(4 / 9)


def test_iv_percentile_max_is_one() -> None:
    assert iv_percentile(0.50, [0.10, 0.12, 0.14]) == pytest.approx(1.0)


def test_iv_percentile_strict_all_time_low_is_zero() -> None:
    # now below all history => 0 of 3 strictly below => 0.0 (no 1/(n+1) floor).
    assert iv_percentile(0.05, [0.10, 0.12, 0.14]) == pytest.approx(0.0)
    # and an all-time low with SHORT history opens the Johnson low-percentile gate.
    assert johnson_gate(slope=0.04, iv_pct=iv_percentile(0.05, [0.10, 0.12, 0.14])) is True


def test_iv_percentile_always_in_unit_interval() -> None:
    rng = np.random.default_rng(0)
    hist = rng.uniform(0.05, 0.40, size=250)
    for now in rng.uniform(0.0, 0.50, size=50):
        assert 0.0 <= iv_percentile(float(now), hist) <= 1.0


def test_iv_percentile_ignores_nans_in_history() -> None:
    hist = [0.10, float("nan"), 0.20, math.inf]
    # finite history = {0.10, 0.20}; strictly below 0.20 = {0.10} = 1 of 2 = 0.5.
    assert iv_percentile(0.20, hist) == pytest.approx(0.5)


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


# --- gap-filling tests (independent verification) --------------------------------------


def test_iv_percentile_monotone_nondecreasing_in_now() -> None:
    # As `now` rises through a fixed history, the strictly-below count can only grow:
    # percentile must be non-decreasing in `now` (a defining property of empirical rank).
    hist = [0.10, 0.20, 0.20, 0.30, 0.45]
    prev = -1.0
    for now in np.linspace(0.0, 0.60, 200):
        pct = iv_percentile(float(now), hist)
        assert pct >= prev - 1e-12
        prev = pct
    # ends span the full unit interval.
    assert iv_percentile(-1.0, hist) == pytest.approx(0.0)
    assert iv_percentile(1.0, hist) == pytest.approx(1.0)


def test_iv_percentile_single_element_history() -> None:
    # degenerate n=1 population: strict-< gives clean {0, 0} buckets, no divide surprises.
    assert iv_percentile(0.30, [0.20]) == pytest.approx(1.0)
    assert iv_percentile(0.10, [0.20]) == pytest.approx(0.0)
    assert iv_percentile(0.20, [0.20]) == pytest.approx(0.0)  # tie, self excluded


def test_iv_percentile_returns_native_python_float() -> None:
    # downstream gate/JSON expect a plain float, not a numpy scalar.
    out = iv_percentile(0.18, [0.10, 0.20])
    assert type(out) is float
    nan_out = iv_percentile(0.18, [])
    assert type(nan_out) is float and math.isnan(nan_out)


def test_term_slope_antisymmetry_and_native_float() -> None:
    # slope(a,b) == -slope(b,a); and return type is a plain float.
    assert term_slope(0.30, 0.22) == pytest.approx(-term_slope(0.22, 0.30))
    assert type(term_slope(0.30, 0.22)) is float


def test_gate_returns_native_python_bool() -> None:
    # not numpy.bool_ -- the `is True` / `is False` assertions elsewhere rely on this.
    assert type(johnson_gate(slope=-0.05, iv_pct=0.9)) is bool
    assert type(johnson_gate(slope=0.04, iv_pct=0.9)) is bool


def test_gate_custom_iv_pct_threshold_is_strict() -> None:
    # pct exactly at a CUSTOM low threshold must NOT open (strict <); just below opens.
    assert johnson_gate(slope=0.04, iv_pct=0.50, iv_pct_low_below=0.50) is False
    assert johnson_gate(slope=0.04, iv_pct=0.50 - 1e-9, iv_pct_low_below=0.50) is True


def test_gate_custom_slope_threshold_is_strict() -> None:
    # slope exactly at a CUSTOM inversion threshold must NOT open; just below opens.
    assert johnson_gate(slope=0.02, iv_pct=0.9, slope_inverted_below=0.02) is False
    assert johnson_gate(slope=0.02 - 1e-9, iv_pct=0.9, slope_inverted_below=0.02) is True


def test_gate_inf_inputs_treated_as_not_met() -> None:
    # +/-inf are non-finite -> neither condition can be satisfied via that input.
    assert johnson_gate(slope=float("-inf"), iv_pct=0.9) is False
    assert johnson_gate(slope=float("inf"), iv_pct=0.9) is False
    assert johnson_gate(slope=0.04, iv_pct=float("inf")) is False
