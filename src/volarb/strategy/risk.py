"""Sizing + model-free risk controls for the strategy layer (pure functions, no I/O).

Fractional-Kelly sizing on a thin, estimation-uncertain edge; a daily-loss kill-switch; an intraday
greeks envelope. (The Leung-Li optimal-stopping / OU machinery was removed in the jump-continuation
backspread rewrite -- that payoff is a one-shot directional bet, not a mean-reverting edge, so the OU
take-profit/stop math does not apply.)
"""

from __future__ import annotations

import numpy as np

__all__ = ["fractional_kelly", "kelly_lots", "kill_switch", "greeks_envelope"]


def fractional_kelly(mean_pnl: float, var_pnl: float, n_trades: int, lam_base: float = 0.5) -> float:
    """Fractional-Kelly bet fraction on a thin, ESTIMATED edge.

    Full Kelly f = mean/var; shrink by lam_base AND by max(0, 1 - 1/t^2), t = mean/SE(mean),
    SE = sqrt(var/n). No edge / weak t-stat -> 0. Never exceeds full Kelly; never negative. Pass
    lam_base=0.25 (quarter-Kelly) for a low-confidence signal. ``mean_pnl``/``var_pnl`` must be
    per-trade RETURN moments (so f is a dimensionless fraction).
    """
    if var_pnl <= 0 or n_trades < 2:
        return 0.0
    f_full = mean_pnl / var_pnl
    se = np.sqrt(var_pnl / n_trades)
    t = mean_pnl / se if se > 0 else 0.0
    shrink = max(0.0, 1.0 - 1.0 / (t * t)) if t != 0 else 0.0
    f_used = lam_base * shrink * f_full
    return float(max(0.0, min(f_used, f_full)))


def kelly_lots(
    mean_pnl: float,
    var_pnl: float,
    n_trades: int,
    risk_capital: float,
    margin_per_lot: float,
    lam_base: float = 0.5,
) -> int:
    """Whole-lot size from the fractional-Kelly fraction (rounded DOWN, >= 0).

    ``margin_per_lot`` is the at-risk unit per lot/structure (for the backspread = its max-loss);
    margin/budget caps are applied separately by the caller (sizing.max_structures).
    """
    if margin_per_lot <= 0 or risk_capital <= 0:
        return 0
    f = fractional_kelly(mean_pnl, var_pnl, n_trades, lam_base)
    return int(max(0, np.floor(f * risk_capital / margin_per_lot)))


def kill_switch(day_pnl: float, kill_rupees: float) -> bool:
    """Hard daily-loss kill-switch: True => flatten everything, no new trades today."""
    return day_pnl <= -abs(kill_rupees)


def greeks_envelope(
    vega_now: float,
    vega_entry: float,
    theta_bleed: float,
    gamma_rent: float,
    net_delta: float,
    *,
    vega_floor_frac: float = 0.5,
    delta_max: float = float("inf"),
) -> dict:
    """Intraday greeks limits -> action flags.

    vega floor: long-vol edge gone once vega < frac*entry; theta envelope: theta bleed exceeding marginal
    gamma rent is the drive-decay exit; delta band: |net_delta| over delta_max => trim. Returns
    {vega_dead, theta_dominates, delta_breach}.
    """
    return {
        "vega_dead": vega_now < vega_floor_frac * vega_entry,
        "theta_dominates": theta_bleed > gamma_rent,
        "delta_breach": abs(net_delta) > delta_max,
    }
