"""Term-structure + IV-percentile gating (Johnson 2017, "Risk Premia and the VIX Term Structure").

The documented long-vol-positive pocket is an INVERTED term structure (front-month ATM IV
above the next expiry, i.e. backwardation = stress) OR a low IV percentile (vol cheap relative
to its own recent history). This module turns two already-computed ATM IV scalars (front and
next expiry) plus a history of front ATM IV into a single boolean gate.

Inputs are ATM IVs in the SAME annualized (calendar-minute) basis produced by
``volarb.pricing.greeks.atm_iv`` -- this module does no inversion of its own.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def term_slope(atm_iv_front: float, atm_iv_next: float) -> float:
    """Term-structure slope = next-expiry IV minus front-expiry IV.

    Negative => INVERTED / backwardated term structure (front richer than next) = stress,
    the long-vol-positive regime. Positive => normal upward-sloping contango.
    """
    return float(atm_iv_next) - float(atm_iv_front)


def iv_percentile(atm_iv_now: float, atm_iv_history: Iterable[float]) -> float:
    """Percentile rank in [0, 1] of the current ATM IV within its history (self EXCLUDED).

    ``count(hist < now) / len(hist)`` -- fraction of HISTORICAL observations strictly below the
    current value. Self is not in the population, so a strict all-time low ranks 0.0 regardless
    of history length (no 1/(n+1) floor), letting the low-IV-percentile gate open early in the
    backfill when history is short.

    Returns ``nan`` if there is no finite history to rank against.
    """
    hist = np.asarray(list(atm_iv_history), dtype=float)
    hist = hist[np.isfinite(hist)]
    now = float(atm_iv_now)
    if hist.size == 0 or not np.isfinite(now):
        return float("nan")
    return float(np.count_nonzero(hist < now) / hist.size)


def johnson_gate(
    slope: float,
    iv_pct: float,
    *,
    slope_inverted_below: float = 0.0,
    iv_pct_low_below: float = 0.25,
) -> bool:
    """Open the long-vol gate in the Johnson (2017) SLOPE pocket.

    True when EITHER the term structure is inverted (``slope < slope_inverted_below``) OR the
    current IV percentile is low (``iv_pct < iv_pct_low_below``). High percentile combined with
    an upward (non-inverted) slope closes the gate.

    Non-finite inputs (``nan``) are treated as "condition not met" so a missing next-expiry IV
    or empty history never spuriously opens the gate.
    """
    inverted = bool(np.isfinite(slope) and slope < slope_inverted_below)
    low_pct = bool(np.isfinite(iv_pct) and iv_pct < iv_pct_low_below)
    return inverted or low_pct
