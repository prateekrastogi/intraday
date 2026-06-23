import numpy as np
import pytest
from volarb.signals.semivariance import (
    realized_semivariance,
    signed_jump_variation,
    downside_har_features,
)


def test_rs_minus_equals_rv_minus_sj_over_two():
    rng = np.random.default_rng(21)
    r = rng.standard_normal(400) * 1e-3
    r[::6] = 0.0
    rs_plus, rs_minus = realized_semivariance(r)
    rv = rs_plus + rs_minus
    sj = signed_jump_variation(r)
    assert rs_minus == pytest.approx((rv - sj) / 2.0, rel=1e-12, abs=1e-18)
    assert rs_plus == pytest.approx((rv + sj) / 2.0, rel=1e-12, abs=1e-18)


def test_semivariance_concatenation_additivity():
    rng = np.random.default_rng(22)
    a = rng.standard_normal(80) * 1e-3
    b = rng.standard_normal(80) * 1e-3
    pa, ma = realized_semivariance(a)
    pb, mb = realized_semivariance(b)
    pc, mc = realized_semivariance(np.concatenate([a, b]))
    assert pa + pb == pytest.approx(pc, rel=1e-12, abs=1e-18)
    assert ma + mb == pytest.approx(mc, rel=1e-12, abs=1e-18)


def test_semivariance_components_scale_quadratically():
    rng = np.random.default_rng(23)
    r = rng.standard_normal(256) * 1e-3
    c = 4.3
    p, m = realized_semivariance(r)
    p2, m2 = realized_semivariance(c * r)
    assert p2 == pytest.approx(c * c * p, rel=1e-12, abs=1e-18)
    assert m2 == pytest.approx(c * c * m, rel=1e-12, abs=1e-18)


def test_infinity_is_filtered_like_nan():
    r = np.array([0.02, np.inf, -np.inf, -0.02])
    rs_plus, rs_minus = realized_semivariance(r)
    assert rs_plus == pytest.approx(4e-4)
    assert rs_minus == pytest.approx(4e-4)
    assert np.isfinite(signed_jump_variation(r))


def test_all_zero_series_is_degenerate_zero():
    rs_plus, rs_minus = realized_semivariance(np.zeros(32))
    assert rs_plus == 0.0 and rs_minus == 0.0
    assert signed_jump_variation(np.zeros(32)) == 0.0


def test_components_nonnegative_and_sj_bounded_by_rv():
    rng = np.random.default_rng(24)
    r = rng.standard_normal(300) * 1e-3
    r[::3] = 0.0
    rs_plus, rs_minus = realized_semivariance(r)
    rv = rs_plus + rs_minus
    assert rs_plus >= 0.0 and rs_minus >= 0.0
    assert abs(signed_jump_variation(r)) <= rv + 1e-18


def test_har_window_exactly_history_length_matches_full():
    rng = np.random.default_rng(25)
    r = rng.standard_normal(30) * 1e-3
    feats = downside_har_features(r, lags=(30,))
    rs_plus, rs_minus = realized_semivariance(r)
    assert feats["rs_minus_30"] == pytest.approx(rs_minus)
    assert feats["rv_30"] == pytest.approx(rs_plus + rs_minus)
    assert feats["sj_30"] == pytest.approx(rs_plus - rs_minus)


def test_har_overlength_window_equals_full_history():
    rng = np.random.default_rng(26)
    r = rng.standard_normal(20) * 1e-3
    over = downside_har_features(r, lags=(60,))
    full = downside_har_features(r, lags=(20,))
    assert over["rs_minus_60"] == pytest.approx(full["rs_minus_20"])
    assert over["rv_60"] == pytest.approx(full["rv_20"])


def test_har_per_window_identity_holds_for_every_lag():
    rng = np.random.default_rng(27)
    r = rng.standard_normal(120) * 1e-3
    feats = downside_har_features(r, lags=(15, 30, 60))
    for h in (15, 30, 60):
        rs_minus = feats[f"rs_minus_{h}"]
        rv = feats[f"rv_{h}"]
        sj = feats[f"sj_{h}"]
        rs_plus = rv - rs_minus
        assert sj == pytest.approx(rs_plus - rs_minus, rel=1e-12, abs=1e-18)
        assert rs_plus >= 0.0 and rs_minus >= 0.0


def test_har_rejects_negative_lag():
    with pytest.raises(ValueError):
        downside_har_features(np.array([0.01, -0.01]), lags=(-5,))
