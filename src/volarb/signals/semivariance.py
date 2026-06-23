"""Realized semivariance and signed jump variation (Patton-Sheppard 2015).

Decomposes realized variance (RV = sum of squared log-returns) into upside and
downside components by the sign of each return:

    RS+ = sum_{r_i > 0} r_i^2        (upside semivariance)
    RS- = sum_{r_i < 0} r_i^2        (downside semivariance)

By construction RS+ + RS- == RV (zero returns contribute zero to either, so they
drop out cleanly). The signed jump variation SJ = RS+ - RS- isolates the
asymmetric / leverage component: it is negative when downside variation dominates
(the typical equity-index "leverage effect"), positive in upside-driven sessions.

Reference: A. J. Patton & K. Sheppard (2015), "Good Volatility, Bad Volatility:
Signed Jumps and the Persistence of Volatility", Review of Economics and
Statistics 97(3). Downside semivariance is the dominant predictor of future RV.

Pure numpy. Inputs are *log-returns* (already differenced); the caller owns the
no-lookahead contract (a feature at t uses returns over (.., t]; the HAR target is
the forward window). This module does not touch the data lake.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "realized_semivariance",
    "signed_jump_variation",
    "downside_har_features",
]


def realized_semivariance(log_returns: np.ndarray) -> tuple[float, float]:
    """Split realized variance into upside (RS+) and downside (RS-) semivariance.

    Args:
        log_returns: 1-D array of intraday log-returns within a single session.
            NaNs (e.g. an overnight/roll gap that must never enter) are ignored.

    Returns:
        (rs_plus, rs_minus): rs_plus = sum of r^2 over r > 0, rs_minus over r < 0.
        rs_plus + rs_minus equals the realized variance sum(r^2). Empty input -> (0, 0).
    """
    r = np.asarray(log_returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 0.0, 0.0
    sq = r * r
    rs_plus = float(sq[r > 0.0].sum())
    rs_minus = float(sq[r < 0.0].sum())
    return rs_plus, rs_minus


def signed_jump_variation(log_returns: np.ndarray) -> float:
    """Signed jump variation SJ = RS+ - RS- (Patton-Sheppard 2015).

    Negative when downside variation dominates (leverage effect), positive when
    upside dominates. Its sign matches the sign of the return-squared skew.
    """
    rs_plus, rs_minus = realized_semivariance(log_returns)
    return rs_plus - rs_minus


def downside_har_features(
    log_returns: np.ndarray,
    lags: tuple[int, ...] = (15, 30, 60),
) -> dict[str, float]:
    """Downside-semivariance HAR predictors over trailing windows (no-lookahead).

    For each lag h (in 1-min-return units), takes the most recent h returns ending
    at the last observation and computes RS- (the leverage-effect predictor), plus
    the matching realized variance and signed-jump value for that window. All
    windows end at the same point t, so every feature uses only data <= t.

    Args:
        log_returns: 1-D intraday log-returns within one session.
        lags: trailing window lengths, e.g. (15, 30, 60) minutes.

    Returns:
        Flat dict with keys ``rs_minus_{h}``, ``rv_{h}``, ``sj_{h}`` for each lag h.
        Windows longer than the available history just use all of it.
    """
    r = np.asarray(log_returns, dtype=float).ravel()
    out: dict[str, float] = {}
    for h in lags:
        if h <= 0:
            raise ValueError(f"lag must be positive, got {h}")
        window = r[-h:]
        rs_plus, rs_minus = realized_semivariance(window)
        out[f"rs_minus_{h}"] = rs_minus
        out[f"rv_{h}"] = rs_plus + rs_minus
        out[f"sj_{h}"] = rs_plus - rs_minus
    return out
