"""Position sizing for the backspread: min(Kelly on max-loss, margin budget, aggregate-risk cap)."""

from __future__ import annotations

from volarb.strategy.risk import fractional_kelly


def max_structures(margin_budget: float, margin_per_structure: float) -> int:
    """How many structures the margin budget can carry (>=0)."""
    if margin_per_structure <= 0:
        return 0
    return int(margin_budget // margin_per_structure)


def size_structures(
    mean_ret: float,
    var_ret: float,
    n_trades: int,
    *,
    risk_capital: float,
    max_loss_per_structure: float,
    margin_per_structure: float,
    margin_budget: float,
    lam_kelly: float = 0.25,
    aggregate_maxloss_frac: float = 0.10,
) -> int:
    """Units to trade = min(Kelly-on-max-loss, margin cap, aggregate-max-loss cap), >=0.

    ``mean_ret``/``var_ret`` are conditional per-trade RETURN moments (P&L / max-loss). The at-risk unit
    is the structure's MAX-LOSS (not premium). margin cap = floor(budget / margin_per_structure).
    """
    if max_loss_per_structure <= 0 or risk_capital <= 0:
        return 0
    f = fractional_kelly(mean_ret, var_ret, n_trades, lam_kelly)
    kelly_units = int(f * risk_capital // max_loss_per_structure)
    margin_units = max_structures(margin_budget, margin_per_structure)
    agg_units = int((aggregate_maxloss_frac * risk_capital) // max_loss_per_structure)
    return max(0, min(kelly_units, margin_units, agg_units))
