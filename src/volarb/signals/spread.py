"""RV-IV spread -- THE edge axis.

This module reconciles a *realized-vol forecast* (produced upstream from the HAR/HARQ
machinery on NIFTY futures, in per-horizon units) against the *self-built ATM implied
vol* (produced by ``volarb.pricing.greeks.atm_iv``, an annualized vol on a calendar-minute
basis). The whole strategy edge is the residual of that difference after removing the
regime variance-risk-premium (VRP); everything here exists to make that subtraction
happen in *identical units*.

Unit contract
-------------
``volarb.pricing.greeks`` annualizes time with ``YEAR_MINUTES = 365 * 24 * 60`` calendar
minutes (``years_to_expiry(minutes) = minutes / YEAR_MINUTES``). So ATM IV is an
annualized vol on that exact basis. A realized-vol number measured over a forward window
of ``horizon_min`` minutes lives on a *per-horizon* basis and is meaningless to subtract
from IV until it is annualized on the same calendar-minute clock. ``annualize_rv`` is the
single explicit place that reconciliation happens; ``rv_iv_spread`` then assumes (and
cheaply checks) that both arguments are already annualized vols.

Only numpy/pandas here. ATM IV is computed upstream -- never recomputed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from volarb.pricing.greeks import YEAR_MINUTES

__all__ = [
    "annualize_rv",
    "rv_iv_spread",
    "debias_vrp",
    "vix_benchmark",
]


def annualize_rv(
    rv_forecast_variance_or_vol: float | np.ndarray | pd.Series,
    horizon_min: float,
    *,
    is_variance: bool = False,
) -> float | np.ndarray | pd.Series:
    """Annualize a per-horizon RV forecast onto the ATM-IV calendar-minute basis.

    Variance is additive in time, so a variance measured over ``horizon_min`` minutes
    annualizes by the linear factor ``YEAR_MINUTES / horizon_min``; vol annualizes by the
    square root of that factor. This is the explicit RV<->IV unit reconciliation: the
    result is an annualized **vol** directly comparable to ``atm_iv``.

    Parameters
    ----------
    rv_forecast_variance_or_vol:
        The forward-window realized-vol forecast, on a *per-horizon* basis. By default
        this is interpreted as a **vol** (standard deviation of the horizon return). Pass
        ``is_variance=True`` if you are handing in a realized **variance** instead (e.g.
        a raw sum-of-squared-returns RV); it is square-rooted internally.
    horizon_min:
        Length of the forward window the forecast covers, in minutes (e.g. 15/30/60).
        Must be strictly positive.
    is_variance:
        Whether ``rv_forecast_variance_or_vol`` is a variance (``True``) or a vol
        (``False``, default).

    Returns
    -------
    Annualized vol, same container type as the input (scalar/ndarray/Series).
    """
    if horizon_min <= 0:
        raise ValueError(f"horizon_min must be > 0, got {horizon_min}")

    scale = YEAR_MINUTES / horizon_min
    if is_variance:
        horizon_variance = rv_forecast_variance_or_vol
    else:
        # vol -> variance so the time-scaling is the simple linear factor.
        horizon_variance = np.square(rv_forecast_variance_or_vol)

    annualized_variance = horizon_variance * scale
    return np.sqrt(annualized_variance)


def rv_iv_spread(
    rv_forecast_annualized_vol: float | np.ndarray | pd.Series,
    atm_iv: float | np.ndarray | pd.Series,
) -> float | np.ndarray | pd.Series:
    """Raw RV-IV spread: ``forecast_annualized_vol - atm_iv``.

    Both inputs MUST already be annualized vols on the same calendar-minute basis -- run
    the RV forecast through :func:`annualize_rv` first. A positive spread means realized
    vol is forecast to *exceed* implied, the long-vol-positive direction. ATM IV is the
    upstream ``volarb.pricing.greeks.atm_iv`` output and is never recomputed here.
    """
    return rv_forecast_annualized_vol - atm_iv


def debias_vrp(
    spread_series: pd.Series,
    window: int,
) -> pd.Series:
    """Strip the regime variance-risk-premium from the spread.

    IV structurally sits above realized vol (the variance-risk-premium), so the raw spread
    carries a persistent negative bias that is *not* tradable edge -- it is the premium
    sellers earn. Subtracting a trailing mean removes that slowly-moving regime VRP and
    leaves the residual the strategy actually trades.

    Uses a *causal* trailing mean (``min_periods=window``), so the debiased value at time
    ``t`` depends only on data ``<= t``: no-lookahead by construction. The first
    ``window - 1`` points have no full trailing window and come back ``NaN``.

    Parameters
    ----------
    spread_series:
        Time-indexed raw spread (output of :func:`rv_iv_spread`).
    window:
        Number of observations in the trailing mean. Must be ``>= 1``.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    trailing_vrp = spread_series.rolling(window=window, min_periods=window).mean()
    return spread_series - trailing_vrp


def vix_benchmark(
    spread: float | np.ndarray | pd.Series,
    vix: float | np.ndarray | pd.Series,
) -> float | np.ndarray | pd.Series:
    """Optional context: how far the spread sits beyond the India-VIX benchmark.

    India VIX is itself an annualized vol (in percent points; pass it as a fraction, e.g.
    ``0.14`` for 14, so units match the spread). Returns ``spread - vix`` -- the part of
    the forecast that beats the *exchange's own* IV benchmark rather than our self-built
    ATM IV. Pure context for gating/diagnostics; not part of the core edge subtraction.
    """
    return spread - vix
