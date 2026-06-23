"""Synthetic-data tests for volarb.signals.quarticity (HARQ / BPQ 2016).

No real lake, no creds, no network. Asserts the MATH:
  - RQ of a known constant series,
  - tripower <= ordinary RQ on a series with a planted outlier,
  - tripower consistency for integrated quarticity on a no-jump series,
  - harq_features shape, column layout, interaction term, and alignment.
"""

from __future__ import annotations

import numpy as np
import pytest

from volarb.signals.quarticity import (
    harq_features,
    realized_quarticity,
    tripower_quarticity,
)


def test_realized_quarticity_known_constant_series() -> None:
    # n returns each equal to c -> RQ = (n/3) * n * c^4.
    n = 12
    c = 0.001
    r = np.full(n, c)
    expected = (n / 3.0) * n * c**4
    assert realized_quarticity(r) == pytest.approx(expected)


def test_realized_quarticity_uses_window_length() -> None:
    r = np.array([0.01, -0.02, 0.0])
    assert realized_quarticity(r) == pytest.approx(
        (3 / 3.0) * np.sum(r**4)
    )  # n == len(r), no override


def test_realized_quarticity_empty_is_zero() -> None:
    assert realized_quarticity(np.array([])) == 0.0


def test_tripower_zero_when_too_short() -> None:
    assert tripower_quarticity(np.array([0.01, -0.02])) == 0.0


def test_tripower_le_ordinary_on_outlier_series() -> None:
    # Smooth small returns plus one big spike: the r^4 sum is dominated by the
    # spike, while tripower spreads it across neighbours -> TPQ << RQ.
    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 1e-3, size=200)
    r[100] = 0.05  # planted jump/outlier
    rq = realized_quarticity(r)
    tpq = tripower_quarticity(r)
    assert tpq <= rq
    assert tpq < 0.5 * rq  # robustness: clearly damped, not marginally


def test_tripower_consistent_for_integrated_quarticity_no_jump() -> None:
    # GBM-like diffusion, no jumps: for r_i ~ N(0, sigma^2/n) over a unit
    # interval, integrated quarticity IQ = sigma^4. Both RQ and TPQ are
    # consistent estimators, so on a fine grid both land near sigma^4 and near
    # each other.
    rng = np.random.default_rng(42)
    n = 20_000
    sigma2 = 0.04  # integrated variance over the window
    r = rng.normal(0.0, np.sqrt(sigma2 / n), size=n)
    iq = sigma2**2
    rq = realized_quarticity(r)
    tpq = tripower_quarticity(r)
    assert rq == pytest.approx(iq, rel=0.05)
    assert tpq == pytest.approx(iq, rel=0.05)
    assert tpq == pytest.approx(rq, rel=0.05)


def test_harq_features_shape_and_layout() -> None:
    t, k = 5, 3
    rv = np.arange(1.0, t * k + 1.0).reshape(t, k)
    rq = np.full(t, 4.0)  # sqrt = 2.0
    design = harq_features(rv, rq)

    # columns = [intercept, sqrt(RQ)*RV_short, RV_lag_1..k]
    assert design.shape == (t, k + 2)
    assert np.all(design[:, 0] == 1.0)  # intercept
    np.testing.assert_allclose(design[:, 1], 2.0 * rv[:, 0])  # interaction
    np.testing.assert_allclose(design[:, 2:], rv)  # plain HAR lags preserved


def test_harq_features_interaction_attenuates_short_lag() -> None:
    # The defining HARQ property: higher RQ -> larger sqrt(RQ)*RV_short term,
    # which (with a negative b1Q) shrinks the effective short-lag loading.
    rv = np.array([[0.02, 0.018, 0.016], [0.02, 0.018, 0.016]])
    rq = np.array([0.0, 9.0])  # sqrt -> 0 and 3
    design = harq_features(rv, rq)
    assert design[0, 1] == pytest.approx(0.0)
    assert design[1, 1] == pytest.approx(3.0 * 0.02)


def test_harq_features_accepts_1d_short_lag() -> None:
    rv = np.array([0.1, 0.2, 0.3])  # single-lag HAR
    rq = np.array([1.0, 4.0, 9.0])  # sqrt -> 1, 2, 3
    design = harq_features(rv, rq)
    assert design.shape == (3, 3)  # intercept, interaction, one lag
    np.testing.assert_allclose(design[:, 1], np.array([1.0, 2.0, 3.0]) * rv)


def test_harq_features_negative_rq_is_clipped() -> None:
    # Defensive: a numerically-negative RQ must not NaN the sqrt.
    rv = np.array([[0.05, 0.04]])
    design = harq_features(rv, np.array([-1.0]))
    assert design[0, 1] == 0.0
    assert not np.isnan(design).any()


def test_harq_features_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        harq_features(np.zeros((4, 2)), np.zeros(3))


# ---------------------------------------------------------------------------
# Gap-filling tests from the independent math-verification pass.
# Math re-derived from first principles and BNS/BPQ definitions; all the
# constants/identities below were numerically confirmed.
# ---------------------------------------------------------------------------

import math  # noqa: E402


def test_realized_quarticity_single_return() -> None:
    # Boundary: n=1 is a valid window (n>=1), RQ = (1/3) r^4, not the n>=3 guard.
    r = np.array([0.01])
    assert realized_quarticity(r) == pytest.approx((1 / 3.0) * 0.01**4)


def test_realized_quarticity_returns_plain_float_and_nonnegative() -> None:
    # Contract: scalar python float (spread.py annualizes it), and r^4 >= 0 so
    # RQ is never negative regardless of return signs.
    out = realized_quarticity(np.array([-0.01, -0.02, -0.03]))
    assert isinstance(out, float)
    assert out >= 0.0


def test_realized_quarticity_ravels_2d_input() -> None:
    # asarray(...).ravel() means a column vector must give the same RQ as 1-d.
    r = np.array([0.01, -0.02, 0.03])
    assert realized_quarticity(r.reshape(-1, 1)) == pytest.approx(realized_quarticity(r))


def test_tripower_exact_on_three_element_window() -> None:
    # n=3 is the minimal non-degenerate window: exactly one triple.
    # TPQ = 3 * mu^-3 * (|2|^(4/3)|3|^(4/3)|4|^(4/3)).
    r = np.array([2.0, 3.0, 4.0])
    mu43 = 2.0 ** (2.0 / 3.0) * math.gamma(7.0 / 6.0) / math.sqrt(math.pi)
    expected = 3 * mu43**-3 * (2 ** (4 / 3) * 3 ** (4 / 3) * 4 ** (4 / 3))
    assert tripower_quarticity(r) == pytest.approx(expected, rel=1e-12)


def test_tripower_zero_return_kills_overlapping_triples() -> None:
    # A single zero return zeroes every triple that contains it. With 4 returns
    # both triples touch index 1, so TPQ == 0 exactly -- the multiplicative
    # structure (not additive) is what makes it jump-robust.
    r = np.array([0.01, 0.0, 0.02, 0.03])
    assert tripower_quarticity(r) == pytest.approx(0.0)


def test_tripower_scales_as_variance_squared() -> None:
    # Like RQ, TPQ is an integrated-quarticity estimator -> homogeneous degree 4:
    # sum |s*r|^(4/3) over three factors = |s|^4 * sum |r|^(4/3).
    r = np.random.default_rng(7).normal(0, 1e-3, 500)
    assert tripower_quarticity(2.5 * r) == pytest.approx(2.5**4 * tripower_quarticity(r), rel=1e-12)


def test_harq_interacts_only_short_not_mid_or_long() -> None:
    # With distinct mid/long lags, confirm column 1 is sqrt(RQ)*short and is NOT
    # accidentally sqrt(RQ)*mid or *long (HARQ, not HARQ-F).
    rv = np.array([[2.0, 5.0, 7.0], [3.0, 11.0, 13.0]])
    rq = np.array([4.0, 9.0])  # sqrt -> 2, 3
    design = harq_features(rv, rq)
    np.testing.assert_allclose(design[:, 1], [2.0 * 2.0, 3.0 * 3.0])
    assert not np.allclose(design[:, 1], np.array([2.0, 3.0]) * rv[:, 1])
    assert not np.allclose(design[:, 1], np.array([2.0, 3.0]) * rv[:, 2])


def test_harq_interaction_monotone_in_rq() -> None:
    # Defining HARQ behaviour: for a fixed positive short lag the interaction
    # column is strictly increasing in RQ.
    rv = np.array([[0.02, 0.01]] * 3)
    design = harq_features(rv, np.array([1.0, 4.0, 9.0]))
    assert design[0, 1] < design[1, 1] < design[2, 1]


def test_harq_features_no_column_guard_raises() -> None:
    # k < 1 (zero RV columns) must raise -- there is no short lag to interact.
    with pytest.raises(ValueError):
        harq_features(np.zeros((3, 0)), np.zeros(3))


def test_harq_features_passes_through_nan_no_drop() -> None:
    # CONTRACT: alignment/no-lookahead is the caller's job; the function does NOT
    # drop or impute NaNs. A NaN short lag propagates into the interaction.
    rv = np.array([[np.nan, 0.1], [0.05, 0.04]])
    design = harq_features(rv, np.array([4.0, 4.0]))
    assert np.isnan(design[0, 1])  # nan short -> nan interaction, not silently 0
    assert design[1, 1] == pytest.approx(2.0 * 0.05)


def test_harq_features_ravels_2d_rq() -> None:
    # rq given as a (T,1) column must be raveled, not broadcast.
    design = harq_features(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[4.0], [9.0]]))
    np.testing.assert_allclose(design[:, 1], [2.0 * 1.0, 3.0 * 3.0])


def test_harq_design_recovers_betas_via_ols() -> None:
    # End-to-end: a clean OLS on the HARQ design recovers planted coefficients,
    # confirming the columns are independent regressors in the documented order
    # [intercept, sqrt(RQ)*RV_short, RV_short, RV_mid, RV_long].
    rng = np.random.default_rng(3)
    t = 600
    rv = np.abs(rng.normal(0, 1.0, (t, 3)))  # O(1) -> well-conditioned
    rq = np.abs(rng.normal(0, 1.0, t))
    x = harq_features(rv, rq)
    beta = np.array([0.5, -0.3, 0.4, 0.2, 0.1])
    # einsum avoids a spurious matmul-fast-path RuntimeWarning on some BLAS
    # builds; inputs are O(1) and the result is exactly finite.
    y = np.einsum("ij,j->i", x, beta)
    bhat, *_ = np.linalg.lstsq(x, y, rcond=None)
    np.testing.assert_allclose(bhat, beta, atol=1e-8)
