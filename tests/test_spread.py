"""Synthetic-data tests for volarb.signals.spread.

No real lake, no creds, no network. Every fixture is hand-built numpy/pandas so each
assertion pins a known closed-form result. The unit-reconciliation tests are the load
bearing ones: they prove RV and IV are in identical units *before* differencing.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from volarb.signals.spread import (
    annualize_rv,
    debias_vrp,
    rv_iv_spread,
    vix_benchmark,
)

# Mirror the foundation constant locally so the test pins the expected calendar-minute
# basis independently of the module under test (and documents what "annualized" means).
YEAR_MINUTES = 365 * 24 * 60
assert YEAR_MINUTES == 525_600


# --------------------------------------------------------------------------------------
# annualize_rv: unit reconciliation
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("horizon_min", [15.0, 30.0, 60.0, 375.0])
def test_annualize_rv_vol_roundtrip_is_exact(horizon_min: float) -> None:
    # Build a per-horizon vol that is *defined* to be 20% annualized, then check we get
    # 0.20 back. This is the "20%-annualized RV forecast" target from the spec.
    ann_target = 0.20
    horizon_vol = ann_target / math.sqrt(YEAR_MINUTES / horizon_min)
    out = annualize_rv(horizon_vol, horizon_min, is_variance=False)
    assert out == pytest.approx(ann_target)


@pytest.mark.parametrize("horizon_min", [15.0, 30.0, 60.0, 375.0])
def test_annualize_rv_variance_matches_vol_path(horizon_min: float) -> None:
    # Feeding the variance must equal feeding the corresponding vol: sqrt(var)=vol.
    horizon_vol = 0.013
    horizon_var = horizon_vol**2
    from_vol = annualize_rv(horizon_vol, horizon_min, is_variance=False)
    from_var = annualize_rv(horizon_var, horizon_min, is_variance=True)
    assert from_var == pytest.approx(from_vol)


def test_annualize_rv_variance_scales_linearly_vol_as_sqrt() -> None:
    # Direct formula check: ann_vol = horizon_vol * sqrt(YEAR_MINUTES / horizon_min).
    horizon_vol = 0.011
    horizon_min = 30.0
    expected = horizon_vol * math.sqrt(YEAR_MINUTES / horizon_min)
    assert annualize_rv(horizon_vol, horizon_min, is_variance=False) == pytest.approx(expected)


def test_annualize_rv_doubling_horizon_variance_scales_inverse() -> None:
    # Same per-horizon variance over a 2x-longer window annualizes to a smaller number,
    # by exactly sqrt(2): annualized variance ~ 1/horizon_min.
    horizon_var = 1e-4
    short = annualize_rv(horizon_var, 30.0, is_variance=True)
    long = annualize_rv(horizon_var, 60.0, is_variance=True)
    assert short / long == pytest.approx(math.sqrt(2.0))


def test_annualize_rv_vectorized_over_series() -> None:
    idx = pd.date_range("2024-01-02 09:15", periods=4, freq="min", tz="Asia/Kolkata")
    horizon_min = 30.0
    vols = pd.Series([0.005, 0.010, 0.015, 0.020], index=idx)
    out = annualize_rv(vols, horizon_min, is_variance=False)
    factor = math.sqrt(YEAR_MINUTES / horizon_min)
    expected = vols.to_numpy() * factor
    np.testing.assert_allclose(np.asarray(out), expected)


def test_annualize_rv_rejects_nonpositive_horizon() -> None:
    with pytest.raises(ValueError):
        annualize_rv(0.01, 0.0, is_variance=False)
    with pytest.raises(ValueError):
        annualize_rv(0.01, -30.0, is_variance=False)


# --------------------------------------------------------------------------------------
# rv_iv_spread: identical units before differencing
# --------------------------------------------------------------------------------------


def test_spread_20pct_rv_vs_18pct_iv_is_plus_2pct() -> None:
    # The headline spec example: 20% annualized RV forecast vs 18% ATM IV -> +2%.
    spread = rv_iv_spread(0.20, 0.18)
    assert spread == pytest.approx(0.02)


def test_spread_consumes_annualized_rv_in_iv_units() -> None:
    # End-to-end: a per-horizon vol that annualizes to 20% must, after annualize_rv,
    # produce a +2% spread against an 18% ATM IV. This proves RV is put into IV units
    # *before* the subtraction.
    horizon_min = 30.0
    atm_iv = 0.18
    horizon_vol = 0.20 / math.sqrt(YEAR_MINUTES / horizon_min)
    rv_ann = annualize_rv(horizon_vol, horizon_min, is_variance=False)
    assert rv_ann == pytest.approx(0.20)  # identical units to atm_iv
    assert rv_iv_spread(rv_ann, atm_iv) == pytest.approx(0.02)


def test_spread_negative_when_iv_above_rv() -> None:
    assert rv_iv_spread(0.15, 0.18) == pytest.approx(-0.03)


def test_spread_vectorized() -> None:
    rv = pd.Series([0.20, 0.18, 0.22])
    iv = pd.Series([0.18, 0.18, 0.18])
    out = rv_iv_spread(rv, iv)
    pd.testing.assert_series_equal(out, pd.Series([0.02, 0.0, 0.04]), check_exact=False)


# --------------------------------------------------------------------------------------
# debias_vrp: removes a constant regime VRP, no lookahead
# --------------------------------------------------------------------------------------


def test_debias_removes_constant_vrp() -> None:
    # A constant spread (a pure, unchanging VRP) must debias to ~0 once the trailing
    # window is full.
    idx = pd.date_range("2024-01-02 09:15", periods=10, freq="min", tz="Asia/Kolkata")
    spread = pd.Series(np.full(10, -0.03), index=idx)
    out = debias_vrp(spread, window=3)
    # First window-1 points have no full window -> NaN.
    assert out.iloc[:2].isna().all()
    np.testing.assert_allclose(out.iloc[2:].to_numpy(), 0.0, atol=1e-12)


def test_debias_leaves_residual_above_trailing_mean() -> None:
    # Constant -0.03 VRP plus a +0.01 bump on the last bar: the residual should be the
    # bump shrunk by the window-averaging of its own value, i.e. strictly positive and
    # equal to bump - bump/window for a single spike at the window edge.
    idx = pd.date_range("2024-01-02 09:15", periods=6, freq="min", tz="Asia/Kolkata")
    base = -0.03
    vals = np.full(6, base)
    vals[-1] = base + 0.01
    spread = pd.Series(vals, index=idx)
    window = 3
    out = debias_vrp(spread, window=window)
    # Last point: value - mean(last 3) = (base+0.01) - (base + 0.01/3) = 0.01*(1 - 1/3).
    assert out.iloc[-1] == pytest.approx(0.01 * (1 - 1 / window))
    # A pre-bump interior point sits in an all-constant window -> 0.
    assert out.iloc[3] == pytest.approx(0.0)


def test_debias_is_causal_no_lookahead() -> None:
    # Changing a *future* value must not alter a past debiased value. Build two series
    # identical up to index k and differing only after; debiased[:k+1] must match.
    n = 12
    idx = pd.date_range("2024-01-02 09:15", periods=n, freq="min", tz="Asia/Kolkata")
    rng = np.random.default_rng(0)
    base = pd.Series(rng.normal(-0.02, 0.005, n), index=idx)
    perturbed = base.copy()
    k = 6
    perturbed.iloc[k + 1 :] += 0.10  # only the future changes
    window = 4
    out_base = debias_vrp(base, window=window)
    out_pert = debias_vrp(perturbed, window=window)
    pd.testing.assert_series_equal(out_base.iloc[: k + 1], out_pert.iloc[: k + 1])


def test_debias_rejects_bad_window() -> None:
    s = pd.Series([0.0, 1.0, 2.0])
    with pytest.raises(ValueError):
        debias_vrp(s, window=0)


# --------------------------------------------------------------------------------------
# vix_benchmark: context
# --------------------------------------------------------------------------------------


def test_vix_benchmark_subtracts_vix_in_same_units() -> None:
    # spread of +0.02 against a VIX benchmark of 0.14 -> -0.12 (forecast does not beat
    # the exchange IV benchmark even though it beats our ATM IV).
    assert vix_benchmark(0.02, 0.14) == pytest.approx(-0.12)


def test_vix_benchmark_vectorized() -> None:
    spread = pd.Series([0.20, 0.16])
    vix = pd.Series([0.14, 0.18])
    pd.testing.assert_series_equal(
        vix_benchmark(spread, vix), pd.Series([0.06, -0.02]), check_exact=False
    )


# --------------------------------------------------------------------------------------
# GAP-FILLING TESTS (independent math-verification pass)
# Pin the edges/properties the originals only ran past: required-keyword contract,
# vectorized negative-variance clip, zero/boundary inputs, the window=1 and NaN-propagation
# behavior of debias, and the shift-invariance identity that DEFINES debiasing.
# --------------------------------------------------------------------------------------


def test_annualize_rv_requires_is_variance_keyword() -> None:
    # is_variance is a REQUIRED kw with no default so a variance is never silently
    # square-rooted-then-squared (the load-bearing safety property). Omitting it must
    # be a hard TypeError, not a silent default.
    with pytest.raises(TypeError):
        annualize_rv(0.01, 30.0)  # type: ignore[call-arg]


def test_annualize_rv_clips_negative_variance_vectorized() -> None:
    # The clip is element-wise: a negative-variance element comes back exactly 0.0 (not
    # NaN, which would poison the spread/gate downstream) while neighbors are unaffected.
    arr = np.array([-1e-4, 1e-4, 0.0])
    out = np.asarray(annualize_rv(arr, 30.0, is_variance=True))
    assert out[0] == 0.0  # clipped, not NaN
    assert out[2] == 0.0  # exact zero passes through
    assert out[1] == pytest.approx(math.sqrt(1e-4 * YEAR_MINUTES / 30.0))
    assert not np.isnan(out).any()


def test_annualize_rv_zero_input_is_zero_both_paths() -> None:
    # Degenerate zero vol / zero variance -> zero annualized vol on either path.
    assert annualize_rv(0.0, 30.0, is_variance=False) == 0.0
    assert annualize_rv(0.0, 30.0, is_variance=True) == 0.0


def test_annualize_rv_negative_vol_path_squares_to_positive() -> None:
    # On the vol path a (nonsensically) negative vol is squared to a positive variance,
    # so the magnitude is preserved -- it must NOT be clipped to zero like a negative
    # variance would be. Pins that the clip lives strictly on the variance, post-square.
    assert annualize_rv(-0.013, 30.0, is_variance=False) == pytest.approx(
        annualize_rv(0.013, 30.0, is_variance=False)
    )


def test_debias_window_one_is_all_zero() -> None:
    # Boundary window=1: each point's trailing mean is itself, so the debiased series is
    # identically zero with NO leading NaN (window-1 == 0 missing points).
    s = pd.Series([0.10, -0.02, 0.30, -0.05])
    out = debias_vrp(s, window=1)
    assert not out.isna().any()
    np.testing.assert_allclose(out.to_numpy(), 0.0, atol=1e-15)


def test_debias_is_shift_invariant() -> None:
    # The defining property of debiasing: adding a constant level shift (a different
    # regime VRP) to the WHOLE series leaves the debiased residual unchanged, because the
    # trailing mean absorbs the shift exactly. Stronger than "constant -> 0".
    rng = np.random.default_rng(1)
    base = pd.Series(rng.normal(0.0, 0.01, 20))
    shifted = base + 0.5
    a = debias_vrp(base, window=5)
    b = debias_vrp(shifted, window=5)
    pd.testing.assert_series_equal(a, b, check_exact=False, atol=1e-12)


def test_debias_propagates_interior_nan_no_fabrication() -> None:
    # A NaN (missing spread) inside the trailing window must NOT be silently treated as
    # zero or skipped: with min_periods=window the affected windows come back NaN rather
    # than fabricating a debiased value from incomplete data.
    s = pd.Series([0.10, np.nan, 0.30, 0.40, 0.50])
    out = debias_vrp(s, window=2)
    assert out.iloc[0:3].isna().all()  # idx0 (warmup), idx1 (is nan), idx2 (window incl nan)
    assert out.iloc[3:].notna().all()  # clear of the nan -> finite again


def test_debias_preserves_index_and_length() -> None:
    # Output is index-aligned to the input (callers join it back to the spread series).
    idx = pd.date_range("2024-01-02 09:15", periods=8, freq="min", tz="Asia/Kolkata")
    s = pd.Series(np.linspace(-0.03, -0.01, 8), index=idx)
    out = debias_vrp(s, window=4)
    pd.testing.assert_index_equal(out.index, s.index)
    assert len(out) == len(s)
