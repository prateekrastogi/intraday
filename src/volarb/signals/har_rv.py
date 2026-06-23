"""Intraday-adapted HAR-RV on the underlying (NIFTY futures), 1-min log-returns.

This is the classic Corsi (2009) HAR-RV cascade re-scaled for *intraday* use:
the daily 1/5/22-day lags are replaced by trailing windows measured in MINUTES
(default 15/30/60) and the target is the realized vol over a FORWARD intraday
window (t, t+horizon], rather than the next day's RV.

Design rules baked in (IMPLEMENTATION.md vol_research s1):
  * 1-min log-returns are the atomic input; RV is aggregated into ``bar_min``
    blocks (5-min components are fine and reduce microstructure noise).
  * NO-LOOKAHEAD: every feature at row t uses only data with index <= t; the
    target uses only the forward window strictly after t.
  * WITHIN-SESSION: the caller passes one session's returns at a time, so a
    return is never taken across the close / the monthly futures roll.
  * statsmodels does the regression (HAC/Newey-West covariance); we only build
    the causal feature/target matrices and the variance cascade.

Nothing here annualizes: RV/vol is returned in the SAME raw return units the
caller supplies. The downstream RV-IV spread module is responsible for
annualizing to the calendar-minute basis used by ``volarb.pricing.greeks``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def realized_vol(log_returns: pd.Series, bar_min: int = 5) -> pd.Series:
    """Per-bar realized variance: sum of squared 1-min log-returns per block.

    Parameters
    ----------
    log_returns
        1-min log-returns for a single session, time-ordered. The index is
        preserved positionally; only its order matters.
    bar_min
        Number of 1-min returns aggregated into one realized-variance bar.
        ``bar_min=1`` returns the squared returns themselves.

    Returns
    -------
    pd.Series
        Realized variance per ``bar_min`` block (sum of squared returns). The
        index is the *last* 1-min timestamp of each block, so the value is
        known exactly at that time (causal). A trailing partial block (fewer
        than ``bar_min`` returns) is dropped so every bar spans equal time.
    """
    if bar_min < 1:
        raise ValueError("bar_min must be >= 1")

    r = log_returns.dropna()
    sq = np.square(r.to_numpy(dtype=float))
    n_full = len(sq) // bar_min
    if n_full == 0:
        return pd.Series(dtype=float)

    usable = n_full * bar_min
    # Sum squared returns within each consecutive bar_min block.
    rv = sq[:usable].reshape(n_full, bar_min).sum(axis=1)
    # Label each bar with the timestamp of its closing 1-min return.
    bar_index = r.index[bar_min - 1 : usable : bar_min]
    return pd.Series(rv, index=bar_index, name="rv")


def har_features(
    rv: pd.Series,
    lags_min: tuple[int, ...] = (15, 30, 60),
    horizon_min: int = 30,
    bar_min: int = 5,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the causal HAR design matrix X and forward-vol target y.

    The HAR cascade regresses future volatility on trailing *averages* of
    realized variance over several horizons. Here each horizon is a window of
    minutes (``lags_min``) and the target is the realized vol over the forward
    window ``(t, t+horizon_min]``.

    All windows are expressed in bars via ``// bar_min`` (e.g. 15-min lag with
    5-min bars = 3 bars). ``rv`` must already be per-bar realized *variance*
    (the output of :func:`realized_vol`).

    Returns
    -------
    (X, y)
        ``X`` columns are ``rv_<lag>`` = the trailing mean realized variance
        over each lag window ENDING at the bar's own timestamp (inclusive of t,
        so it is known at t -> no lookahead). ``y`` is the realized *variance*
        (summed forward variance) over the next ``horizon_min`` minutes,
        i.e. strictly after t. Rows where any feature window or the forward
        target window is incomplete are dropped, so X and y share an index and
        are causally aligned by construction.
    """
    if bar_min < 1:
        raise ValueError("bar_min must be >= 1")
    lag_bars = tuple(lag // bar_min for lag in lags_min)
    horizon_bars = horizon_min // bar_min
    if min(lag_bars) < 1 or horizon_bars < 1:
        raise ValueError("each lag and the horizon must be >= bar_min minutes")

    rv = rv.astype(float)

    # --- features: trailing-mean RV over each lag window, ENDING at t ----------
    # rolling(window).mean() at row t uses rows [t-window+1 .. t] -> only <= t.
    feats = {
        f"rv_{lag}": rv.rolling(window=bars, min_periods=bars).mean()
        for lag, bars in zip(lags_min, lag_bars)
    }
    X = pd.DataFrame(feats, index=rv.index)

    # --- target: realized vol over the FORWARD window (t, t+horizon] -----------
    # Forward sum of variance strictly after t: reverse, trailing-sum, reverse,
    # then shift by one bar so row t excludes its own bar (uses t+1 .. t+h).
    fwd_var = rv[::-1].rolling(window=horizon_bars, min_periods=horizon_bars).sum()[::-1].shift(-1)
    # Target = forward realized VARIANCE (variance-on-variance HAR): same functional as the
    # features, so the downstream RV-IV spread differences like-for-like. sqrt(->vol) is applied
    # only at annualization in spread.py (annualize_rv(..., is_variance=True)).
    y = fwd_var.copy()
    y.name = "var_fwd"

    # --- align causally: keep rows with complete features AND complete target --
    valid = X.notna().all(axis=1) & y.notna()
    return X.loc[valid], y.loc[valid]


def fit_har(y: pd.Series, X: pd.DataFrame, horizon_min: int = 30, bar_min: int = 5):
    """Fit the HAR regression via statsmodels OLS with HAC (Newey-West) errors.

    Overlapping forward windows make the residuals serially correlated, so we
    use a heteroskedasticity-and-autocorrelation-consistent covariance with
    ``maxlags`` ~ the horizon length in bars (the overlap depth). The point
    estimates are plain OLS; only the standard errors change.
    """
    X_const = sm.add_constant(X, has_constant="add")
    maxlags = max(1, horizon_min // bar_min - 1)
    model = sm.OLS(y.to_numpy(dtype=float), X_const.to_numpy(dtype=float))
    return model.fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})


def forecast(res, x_row: pd.Series) -> float:
    """Point forecast of forward realized VARIANCE for one feature row (clipped >= 0).

    ``x_row`` holds the same lag features as the training X (no constant); the
    intercept is prepended here to match :func:`fit_har`.
    """
    x = np.concatenate([[1.0], x_row.to_numpy(dtype=float)])
    return max(0.0, float(res.predict(x[None, :])[0]))
