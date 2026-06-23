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
    np.testing.assert_allclose(rv.to_numpy(), [0.1**2 + 0.2**2 + 0.3**2, 0.0 + 0.4**2 + 0.1**2])


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
    rv = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], index=_minute_index(8))
    X, _ = har_features(rv, lags_min=(3,), horizon_min=2, bar_min=1)
    assert X["rv_3"].iloc[0] == pytest.approx(2.0)
    assert X.index[0] == rv.index[2]
    assert X["rv_3"].iloc[1] == pytest.approx(np.mean([2.0, 3.0, 4.0]))


def test_target_is_forward_window_strictly_after_t():
    rv = pd.Series(np.arange(1, 11, dtype=float), index=_minute_index(10))
    _, y = har_features(rv, lags_min=(1,), horizon_min=2, bar_min=1)
    assert y.iloc[0] == pytest.approx(rv.iloc[1] + rv.iloc[2])  # forward VARIANCE, no sqrt


def test_no_lookahead_feature_independent_of_future_mutation():
    rv = pd.Series(np.linspace(1.0, 5.0, 12), index=_minute_index(12))
    before = har_features(rv, lags_min=(3,), horizon_min=3, bar_min=1)[0]["rv_3"].iloc[3]
    rv2 = rv.copy()
    rv2.iloc[8:] *= 100.0
    after = har_features(rv2, lags_min=(3,), horizon_min=3, bar_min=1)[0]["rv_3"].iloc[3]
    assert after == pytest.approx(before)


def test_X_and_y_share_index_and_have_no_nans():
    rng = np.random.default_rng(0)
    rv = pd.Series(np.abs(rng.normal(size=120)) * 1e-4, index=_minute_index(120))
    X, y = har_features(rv, lags_min=(15, 30, 60), horizon_min=30, bar_min=5)
    assert list(X.index) == list(y.index)
    assert not X.isna().to_numpy().any() and not y.isna().any()
    assert list(X.columns) == ["rv_15", "rv_30", "rv_60"]


def test_last_rows_dropped_for_incomplete_forward_window():
    rv = pd.Series(np.abs(np.sin(np.arange(100))) + 0.1, index=_minute_index(100))
    _, y = har_features(rv, lags_min=(5,), horizon_min=15, bar_min=5)
    assert y.index[-1] <= rv.index[-3]


def _ar1_variance_series(n, phi=0.9, seed=7):
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = 1.0
    for i in range(1, n):
        x[i] = 0.1 + phi * x[i - 1] + rng.normal(scale=0.05)
    return pd.Series(np.abs(x) * 1e-4, index=_minute_index(n))


def test_fit_and_forecast_runs_and_is_finite():
    X, y = har_features(_ar1_variance_series(400), lags_min=(15, 30, 60), horizon_min=30, bar_min=5)
    res = fit_har(y, X, horizon_min=30, bar_min=5)
    assert res.params.shape[0] == 4
    pred = forecast(res, X.iloc[-1])
    assert np.isfinite(pred) and pred > 0


def test_persistent_vol_gives_positive_lag_loading():
    X, y = har_features(
        _ar1_variance_series(600, phi=0.92), lags_min=(15, 30, 60), horizon_min=30, bar_min=5
    )
    assert fit_har(y, X, horizon_min=30, bar_min=5).params[1:].sum() > 0.0


def test_uses_hac_covariance():
    X, y = har_features(_ar1_variance_series(300), lags_min=(15, 30, 60), horizon_min=30, bar_min=5)
    assert fit_har(y, X, horizon_min=30, bar_min=5).cov_type == "HAC"


# ---------------------------------------------------------------------------
# Coverage additions (independent math-verification pass).
# Pin the LOAD-BEARING properties the originals only ran: the negative-forecast
# clip, the exact HAC maxlags (overlap depth), input-validation failure modes,
# NaN/degenerate handling, the X/y same-functional identity, distinct multi-lag
# windows, and the production-default forward-sum value.
# ---------------------------------------------------------------------------


class _FakeRes:
    """Minimal stand-in for a fitted statsmodels result with a fixed linear map."""

    def __init__(self, beta):
        self._beta = np.asarray(beta, dtype=float)

    def predict(self, x):
        return np.atleast_1d(x @ self._beta)


def test_forecast_clips_negative_to_zero():
    # A genuinely negative point forecast (OLS on variance can go <0) must clip to 0.0, not pass through.
    x_row = pd.Series([1.0, 1.0, 1.0], index=["rv_15", "rv_30", "rv_60"])
    res_neg = _FakeRes([-10.0, 0.0, 0.0, 0.0])  # intercept -10 -> prediction -10
    assert forecast(res_neg, x_row) == 0.0


def test_forecast_passes_through_positive_and_prepends_intercept():
    x_row = pd.Series([2.0, 3.0, 4.0], index=["rv_15", "rv_30", "rv_60"])
    # beta = [intercept, b15, b30, b60]; prediction = 0.5 + 1*2 + 0*3 + 0*4 = 2.5
    res = _FakeRes([0.5, 1.0, 0.0, 0.0])
    assert forecast(res, x_row) == pytest.approx(2.5)


def test_fit_har_hac_maxlags_equals_overlap_depth():
    # horizon 30m / bar 5m = 6 forward bars -> overlapping windows share up to 5 bars -> maxlags=5.
    X, y = har_features(_ar1_variance_series(400), lags_min=(15, 30, 60), horizon_min=30, bar_min=5)
    res = fit_har(y, X, horizon_min=30, bar_min=5)
    assert res.cov_kwds["maxlags"] == 30 // 5 - 1 == 5


def test_fit_har_maxlags_floored_at_one_for_single_bar_horizon():
    # horizon == bar_min -> 1 forward bar -> naive maxlags=0; must be floored to 1 (valid HAC lag).
    rv = _ar1_variance_series(200)
    X, y = har_features(rv, lags_min=(15,), horizon_min=5, bar_min=5)
    res = fit_har(y, X, horizon_min=5, bar_min=5)
    assert res.cov_kwds["maxlags"] == 1


def test_realized_vol_rejects_bar_min_below_one():
    r = pd.Series([0.1, 0.2], index=_minute_index(2))
    with pytest.raises(ValueError):
        realized_vol(r, bar_min=0)


def test_realized_vol_drops_nans_before_blocking():
    # A NaN return is dropped, then the remaining returns are re-blocked (does NOT zero-fill in place).
    r = pd.Series([0.1, np.nan, 0.2, 0.3, 0.4, 0.5], index=_minute_index(6))
    rv = realized_vol(r, bar_min=2)
    # dropna -> [0.1,0.2,0.3,0.4,0.5]; bars (0.1,0.2) and (0.3,0.4); trailing 0.5 dropped.
    np.testing.assert_allclose(rv.to_numpy(), [0.1**2 + 0.2**2, 0.3**2 + 0.4**2])
    assert len(rv) == 2


def test_realized_vol_empty_when_no_full_bar():
    r = pd.Series([0.1, 0.2], index=_minute_index(2))
    assert realized_vol(r, bar_min=3).empty


def test_har_features_rejects_bar_min_below_one():
    rv = pd.Series(np.ones(10), index=_minute_index(10))
    with pytest.raises(ValueError):
        har_features(rv, lags_min=(15,), horizon_min=30, bar_min=0)


def test_har_features_rejects_lag_shorter_than_one_bar():
    # 3-min lag with 5-min bars -> 0 bars -> must raise, not silently produce an all-NaN feature.
    rv = pd.Series(np.ones(20), index=_minute_index(20))
    with pytest.raises(ValueError):
        har_features(rv, lags_min=(3,), horizon_min=15, bar_min=5)


def test_har_features_rejects_horizon_shorter_than_one_bar():
    rv = pd.Series(np.ones(20), index=_minute_index(20))
    with pytest.raises(ValueError):
        har_features(rv, lags_min=(15,), horizon_min=3, bar_min=5)


def test_features_and_target_are_the_same_functional():
    # Headline math claim: X (trailing-MEAN variance) and y (forward-SUM variance) are both built
    # from the SAME variance series with NO sqrt anywhere. Equal windows -> y == feature * window_bars.
    rv = pd.Series(np.arange(1.0, 13.0), index=_minute_index(12))
    X, y = har_features(rv, lags_min=(3,), horizon_min=3, bar_min=1)
    # feature is a 3-bar trailing mean; target is a 3-bar forward sum = mean(forward 3) * 3.
    # For any aligned row t: forward sum = rv[t+1]+rv[t+2]+rv[t+3].
    for ts in X.index:
        pos = list(rv.index).index(ts)
        assert y.loc[ts] == pytest.approx(rv.iloc[pos + 1 : pos + 4].sum())
        # variance functional: target stays in variance^1 units (no sqrt) -> scales linearly.
    rv_scaled = rv * 4.0
    _, y_scaled = har_features(rv_scaled, lags_min=(3,), horizon_min=3, bar_min=1)
    np.testing.assert_allclose(y_scaled.to_numpy(), 4.0 * y.to_numpy())


def test_distinct_multi_lag_windows_are_separate_trailing_means():
    rv = pd.Series(np.arange(1.0, 13.0), index=_minute_index(12))
    X, _ = har_features(rv, lags_min=(2, 4), horizon_min=2, bar_min=1)
    # first valid row is the 4th bar (needs 4-bar window). value rv=4.
    assert X.index[0] == rv.index[3]
    assert X["rv_2"].iloc[0] == pytest.approx(np.mean([3.0, 4.0]))
    assert X["rv_4"].iloc[0] == pytest.approx(np.mean([1.0, 2.0, 3.0, 4.0]))


def test_production_default_forward_sum_value():
    # Pin the production default (horizon=30, bar=5 -> 6 forward bars) forward-SUM at one row.
    rv = pd.Series(np.arange(1.0, 41.0), index=_minute_index(40))
    _, y = har_features(rv, lags_min=(15, 30, 60), horizon_min=30, bar_min=5)
    first_ts = y.index[0]
    pos = list(rv.index).index(first_ts)
    assert y.loc[first_ts] == pytest.approx(rv.iloc[pos + 1 : pos + 7].sum())  # 6 forward bars


def test_no_lookahead_target_independent_of_pre_t_mutation():
    # Complement to the feature no-lookahead test: the FORWARD target at t must be invariant
    # to mutating rows at-or-before t (it only reads strictly future bars).
    rv = pd.Series(np.linspace(1.0, 5.0, 14), index=_minute_index(14))
    _, y_before = har_features(rv, lags_min=(3,), horizon_min=3, bar_min=1)
    target_ts = y_before.index[5]
    pos = list(rv.index).index(target_ts)
    rv2 = rv.copy()
    rv2.iloc[: pos + 1] *= 100.0  # blow up everything up to and including t
    _, y_after = har_features(rv2, lags_min=(3,), horizon_min=3, bar_min=1)
    assert y_after.loc[target_ts] == pytest.approx(y_before.loc[target_ts])
