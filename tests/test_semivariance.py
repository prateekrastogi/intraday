"""Math tests for volarb.signals.semivariance on synthetic numpy data (no lake)."""

from __future__ import annotations

import numpy as np
import pytest

from volarb.signals.semivariance import (
    downside_har_features,
    realized_semivariance,
    signed_jump_variation,
)


def test_decomposition_sums_to_realized_variance() -> None:
    rng = np.random.default_rng(0)
    r = rng.standard_normal(500) * 1e-3
    rs_plus, rs_minus = realized_semivariance(r)
    rv = float(np.sum(r * r))
    assert rs_plus + rs_minus == pytest.approx(rv, rel=1e-12, abs=1e-18)
    assert rs_plus >= 0.0 and rs_minus >= 0.0


def test_known_series_exact_values() -> None:
    r = np.array([0.1, -0.3, 0.2, 0.0])
    rs_plus, rs_minus = realized_semivariance(r)
    assert rs_plus == pytest.approx(0.01 + 0.04)
    assert rs_minus == pytest.approx(0.09)
    assert signed_jump_variation(r) == pytest.approx(0.05 - 0.09)


def test_all_negative_series_has_zero_upside() -> None:
    r = -np.abs(np.random.default_rng(1).standard_normal(64)) - 1e-6
    rs_plus, rs_minus = realized_semivariance(r)
    assert rs_plus == 0.0
    assert rs_minus > 0.0
    assert signed_jump_variation(r) == pytest.approx(-rs_minus)


def test_all_positive_series_has_zero_downside() -> None:
    r = np.abs(np.random.default_rng(2).standard_normal(64)) + 1e-6
    rs_plus, rs_minus = realized_semivariance(r)
    assert rs_minus == 0.0
    assert signed_jump_variation(r) == pytest.approx(rs_plus)


def test_zeros_drop_out_cleanly() -> None:
    r = np.array([0.0, 0.0, 0.0, 0.01, -0.01])
    rs_plus, rs_minus = realized_semivariance(r)
    assert rs_plus == pytest.approx(1e-4)
    assert rs_minus == pytest.approx(1e-4)
    assert signed_jump_variation(r) == pytest.approx(0.0)


def test_signed_jump_sign_matches_return_skew() -> None:
    from scipy.stats import skew

    rng = np.random.default_rng(7)
    base = rng.standard_normal(2000) * 1e-3
    down = base.copy()
    down[::97] -= 0.05
    up = base.copy()
    up[::97] += 0.05

    assert skew(down) < 0 and signed_jump_variation(down) < 0
    assert skew(up) > 0 and signed_jump_variation(up) > 0


def test_empty_and_nan_inputs() -> None:
    assert realized_semivariance(np.array([])) == (0.0, 0.0)
    r = np.array([0.02, np.nan, -0.02])
    rs_plus, rs_minus = realized_semivariance(r)
    assert rs_plus == pytest.approx(4e-4)
    assert rs_minus == pytest.approx(4e-4)
    assert np.isfinite(signed_jump_variation(r))


def test_downside_har_features_no_lookahead_by_construction() -> None:
    rng = np.random.default_rng(3)
    r = rng.standard_normal(120) * 1e-3
    feats_t = downside_har_features(r[:60], lags=(15, 30, 60))
    for h in (15, 30, 60):
        rs_plus, rs_minus = realized_semivariance(r[:60][-h:])
        assert feats_t[f"rs_minus_{h}"] == pytest.approx(rs_minus)
        assert feats_t[f"rv_{h}"] == pytest.approx(rs_plus + rs_minus)
        assert feats_t[f"sj_{h}"] == pytest.approx(rs_plus - rs_minus)
    feats_future = downside_har_features(r, lags=(15, 30))
    assert feats_future["rs_minus_15"] != pytest.approx(feats_t["rs_minus_15"])


def test_downside_har_features_short_history_uses_all() -> None:
    r = np.array([0.01, -0.02, 0.03])
    feats = downside_har_features(r, lags=(15,))
    rs_plus, rs_minus = realized_semivariance(r)
    assert feats["rs_minus_15"] == pytest.approx(rs_minus)
    assert feats["rv_15"] == pytest.approx(rs_plus + rs_minus)


def test_downside_har_features_rejects_nonpositive_lag() -> None:
    with pytest.raises(ValueError):
        downside_har_features(np.array([0.01, -0.01]), lags=(0,))
