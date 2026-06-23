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


def test_realized_quarticity_explicit_n_overrides_length() -> None:
    r = np.array([0.01, -0.02, 0.0])
    s4 = np.sum(r**4)
    assert realized_quarticity(r, n=375) == pytest.approx((375 / 3.0) * s4)
    # default n == len(r)
    assert realized_quarticity(r) == pytest.approx((3 / 3.0) * s4)


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
