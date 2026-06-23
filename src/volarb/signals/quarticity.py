"""HARQ realized quarticity (Bollerslev, Patton & Quaedvlieg 2016).

Realized quarticity (RQ) estimates the integrated quarticity -- the variance of
the realized-variance estimator itself. BPQ use it to build the HARQ model:
the short (e.g. daily / shortest-horizon) RV lag is interacted with sqrt(RQ) so
that the AR coefficient on RV is *attenuated* in periods where RV is measured
noisily (high RQ). This corrects the attenuation bias of plain HAR and yields
better out-of-sample RV forecasts.

1-MIN RQ-NOISE CAVEAT
---------------------
RQ scales like n * sum(r^4): at 1-min sampling it is extremely sensitive to
microstructure noise and to single large (jump) returns, so the raw estimator
is very noisy in exactly the regimes we care about. Prefer ``tripower_quarticity``
(jump-/noise-robust) for the HARQ interaction term at high frequency. And treat
HARQ as opt-in: it MUST beat plain HAR out-of-sample on this data before it is
allowed into the live signal. This module only computes the design; model
selection happens upstream.

Pure numpy. No lookahead is introduced here -- callers align lags/targets.
"""

from __future__ import annotations

import math

import numpy as np

# E[|Z|^(4/3)]^-3 normalizer for the tripower quarticity (Barndorff-Nielsen &
# Shephard). mu_{4/3} = E|Z|^(4/3) = 2^(2/3) * Gamma(7/6) / Gamma(1/2); the
# estimator divides by mu_{4/3}^3 so it is consistent for integrated quarticity
# under no jumps.
_MU_43 = 2.0 ** (2.0 / 3.0) * math.gamma(7.0 / 6.0) / math.sqrt(math.pi)
_TRIPOWER_NORM = _MU_43**-3


def realized_quarticity(log_returns: np.ndarray) -> float:
    """Realized quarticity RQ = (n/3) * sum(r_i^4), n = number of returns in the window.

    Parameters
    ----------
    log_returns : 1-d array of intraday log-returns within ONE window.

    n is ALWAYS the window length -- the only consistent integrated-quarticity estimator;
    there is no rescale knob (a mismatched n produces a meaningless hybrid). If you need
    annualized RQ, do it in spread.py: RQ annualizes as variance^2, i.e. by
    (YEAR_MINUTES / window_minutes)^2 -- never by substituting n. Very noisy at 1-min
    (see module docstring); prefer ``tripower_quarticity`` at high frequency.
    """
    r = np.asarray(log_returns, dtype=float).ravel()
    if r.size == 0:
        return 0.0
    n = r.size
    return (n / 3.0) * float(np.sum(r**4))


def tripower_quarticity(log_returns: np.ndarray) -> float:
    """Noise-/jump-robust realized quarticity (tripower quarticity, TPQ).

    TPQ = n * mu_{4/3}^-3 * sum_{i=3..n} |r_{i-2}|^(4/3) |r_{i-1}|^(4/3) |r_i|^(4/3)

    By spreading each term across three adjacent returns, a single large (jump)
    return cannot blow up the estimate the way it does in the r^4 sum, so on a
    series with an isolated outlier TPQ <= ordinary RQ. Preferred over
    ``realized_quarticity`` for the HARQ interaction at 1-min sampling. n is the
    window length (no rescale knob).
    """
    r = np.asarray(log_returns, dtype=float).ravel()
    n = r.size
    if r.size < 3:
        return 0.0
    a = np.abs(r) ** (4.0 / 3.0)
    triples = a[:-2] * a[1:-1] * a[2:]
    return n * _TRIPOWER_NORM * float(np.sum(triples))


def harq_features(rv_lags: np.ndarray, rq: np.ndarray) -> np.ndarray:
    """Build the HARQ design matrix (Bollerslev-Patton-Quaedvlieg 2016).

    The HARQ model augments HAR by interacting the SHORT (first) RV lag with
    sqrt(RQ) of that same period:

        RV_{t->t+h} = b0 + (b1 + b1Q*sqrt(RQ_t)) * RV_t^(short)
                         + b2 * RV_t^(mid) + b3 * RV_t^(long) + e

    so the loading on the short lag shrinks when RQ_t is large (noisy RV).

    CONTRACT: rv_lags[:, 0] (the short lag) and ``rq`` MUST be computed over the SAME window
    with matching normalization (both per-period levels) or the b1Q interaction scale is
    meaningless. Keep the RV-lag convention (level vs mean) identical between
    ``har_rv.har_features`` and the RQ fed here.

    Parameters
    ----------
    rv_lags : (T, k) array of HAR RV lags, columns ordered short -> long
              (e.g. [RV_15m, RV_30m, RV_60m]). The FIRST column is the one
              interacted with sqrt(RQ), per BPQ ("HARQ" interacts only the
              shortest lag; "HARQ-F" interacts all lags is a separate variant).
    rq      : (T,) array of realized (or tripower) quarticity aligned to the
              short lag, one value per row of ``rv_lags``.

    Returns
    -------
    design : (T, k+2) array = [intercept, sqrt(RQ)*RV_short, RV_lag_1..k].
             Column 0 is a constant; column 1 is the quarticity interaction
             term; the remaining k columns are the plain HAR lags. This matches
             the HARQ regressor set so the caller can OLS directly. No NaNs are
             dropped -- alignment/no-lookahead is the caller's contract.
    """
    rv = np.asarray(rv_lags, dtype=float)
    if rv.ndim == 1:
        rv = rv.reshape(-1, 1)
    q = np.asarray(rq, dtype=float).ravel()
    t, k = rv.shape
    if q.shape[0] != t:
        raise ValueError(f"rq length {q.shape[0]} != rv_lags rows {t}")
    if k < 1:
        raise ValueError("rv_lags must have at least one column (the short lag)")

    intercept = np.ones((t, 1))
    # sqrt of (clipped-nonneg) RQ times the short RV lag.
    interaction = (np.sqrt(np.clip(q, 0.0, None)) * rv[:, 0]).reshape(-1, 1)
    return np.hstack([intercept, interaction, rv])
