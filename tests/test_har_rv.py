from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from volarb.signals.har_rv import fit_har, forecast, har_features, realized_vol

def _minute_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2026-06-23 09:15", periods=n, freq="1min")

def test_realized_vol_of_known_series():
    r = pd.Series([0.1, -0.2, 0.3, 0.0, 0.4, -0.1], index=_minute_index(6))
    rv = realized_vol(r, bar_min=3)
    assert len(rv) == 2
    np.testing.assert_allclose(rv.to_numpy(), [0.1**2+0.2**2+0.3**2, 0.0+0.4**2+0.1**2])

def test_realized_vol_bar1_is_squared_returns():
    r = pd.Series([0.01, -0.02, 0.03], index=_minute_index(3))
    np.testing.assert_allclose(realized_vol(r, bar_min=1).to_numpy(), r.to_numpy() ** 2)

def test_realized_vol_labels_bar_with_closing_timestamp():
    r = pd.Series([0.1, -0.2, 0.3, 0.0, 0.4, -0.1], index=_minute_index(6))
    rv = realized_vol(r, bar_min=3)
    assert rv.index[0] == r.index[2] and rv.index[1] == r.index[5]

def test_realized_vol_drops_trailing_partial_bar():
    assert len(realized_vol(pd.Series(np.full(7, 0.1), index=_minute_index(7)), bar_min=3)) == 2

def test_features_are_trailing_means_ending_at_t():
    rv = pd.Series([1.,2.,3.,4.,5.,6.,7.,8.], index=_minute_index(8))
    X, _ = har_features(rv, lags_min=(3,), horizon_min=2, bar_min=1)
    assert X["rv_3"].iloc[0] == pytest.approx(2.0)
    assert X.index[0] == rv.index[2]
    assert X["rv_3"].iloc[1] == pytest.approx(np.mean([2.,3.,4.]))

def test_target_is_forward_window_strictly_after_t():
    rv = pd.Series(np.arange(1,11,dtype=float), index=_minute_index(10))
    _, y = har_features(rv, lags_min=(1,), horizon_min=2, bar_min=1)
    assert y.iloc[0] == pytest.approx(np.sqrt(rv.iloc[1] + rv.iloc[2]))

def test_no_lookahead_feature_independent_of_future_mutation():
    rv = pd.Series(np.linspace(1.,5.,12), index=_minute_index(12))
    before = har_features(rv, lags_min=(3,), horizon_min=3, bar_min=1)[0]["rv_3"].iloc[3]
    rv2 = rv.copy(); rv2.iloc[8:] *= 100.0
    after = har_features(rv2, lags_min=(3,), horizon_min=3, bar_min=1)[0]["rv_3"].iloc[3]
    assert after == pytest.approx(before)

def test_X_and_y_share_index_and_have_no_nans():
    rng = np.random.default_rng(0)
    rv = pd.Series(np.abs(rng.normal(size=120))*1e-4, index=_minute_index(120))
    X, y = har_features(rv, lags_min=(15,30,60), horizon_min=30, bar_min=5)
    assert list(X.index) == list(y.index)
    assert not X.isna().to_numpy().any() and not y.isna().any()
    assert list(X.columns) == ["rv_15", "rv_30", "rv_60"]

def test_last_rows_dropped_for_incomplete_forward_window():
    rv = pd.Series(np.abs(np.sin(np.arange(100)))+0.1, index=_minute_index(100))
    _, y = har_features(rv, lags_min=(5,), horizon_min=15, bar_min=5)
    assert y.index[-1] <= rv.index[-3]

def _ar1_variance_series(n, phi=0.9, seed=7):
    rng = np.random.default_rng(seed); x = np.empty(n); x[0] = 1.0
    for i in range(1, n): x[i] = 0.1 + phi*x[i-1] + rng.normal(scale=0.05)
    return pd.Series(np.abs(x)*1e-4, index=_minute_index(n))

def test_fit_and_forecast_runs_and_is_finite():
    X, y = har_features(_ar1_variance_series(400), lags_min=(15,30,60), horizon_min=30, bar_min=5)
    res = fit_har(y, X, horizon_min=30, bar_min=5)
    assert res.params.shape[0] == 4
    pred = forecast(res, X.iloc[-1]); assert np.isfinite(pred) and pred > 0

def test_persistent_vol_gives_positive_lag_loading():
    X, y = har_features(_ar1_variance_series(600, phi=0.92), lags_min=(15,30,60), horizon_min=30, bar_min=5)
    assert fit_har(y, X, horizon_min=30, bar_min=5).params[1:].sum() > 0.0

def test_uses_hac_covariance():
    X, y = har_features(_ar1_variance_series(300), lags_min=(15,30,60), horizon_min=30, bar_min=5)
    assert fit_har(y, X, horizon_min=30, bar_min=5).cov_type == "HAC"
