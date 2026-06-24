"""Jump-continuation entry/exit for the ratio-backspread strategy (1-min bars, intraday).

ENTRY: a CONFIRMED Lee-Mykland jump at the last bar, in the CONTINUATION (momentum) direction, corroborated
by signed-jump-variation + short-window momentum agreeing in sign (screens out one-bar reverting stubs).
Enter at the NEXT bar (no-lookahead: r_t and the jump flag are only known at the t close). Up-jump -> CALL
backspread, down-jump -> PUT backspread (a DIRECTIONAL bet, unlike the abandoned magnitude legs).

EXIT (precedence): EOD square-off (close the likely-ITM short before expiry -> avoid exercise STT) >
continuation take-profit > middle-zone reversal + thesis-invalidation > drive-decay (greeks envelope).
NO hard price stop: the backspread already has a DEFINED max-loss + a convex tail; a stop would truncate
exactly the tail we paid the debit to own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from volarb.signals.lee_mykland import default_window, lee_mykland
from volarb.signals.semivariance import signed_jump_variation


@dataclass(frozen=True)
class JumpEntryParams:
    alpha: float = 0.01
    cont_window: int = 10        # trailing min for signed-jump-variation continuation check
    mom_window: int = 5          # trailing min for the momentum (cum-return) agreement
    cooldown_bars: int = 15      # bars after last entry/exit before a new entry
    min_dte_days: int = 2        # skip expiry-eve (pin risk, wide wings)
    min_hold_minutes: int = 30   # need this long before square-off to let the move develop


def confirmed_jump(returns, *, alpha=0.01, cont_window=10, mom_window=5):
    """At the LAST bar of ``returns`` (1-min, within-session): confirmed jump + continuation?

    Returns (jumped: bool, direction: 'call'|'put'|None). NO-LOOKAHEAD: uses returns through the last bar
    only (lee_mykland's sigma_hat for bar t excludes r_t). Requires the jump flag AND sign-agreement of
    r_t, the trailing signed-jump-variation, and the trailing momentum.
    """
    r = np.asarray(returns, dtype=float)
    n = int(np.isfinite(r).sum())
    if n < default_window(max(n, 1)) + 2:
        return False, None
    jf = lee_mykland(r, alpha=alpha)["jump_flag"]
    if jf is None or len(jf) == 0 or not bool(jf[-1]):
        return False, None
    rt = float(r[-1])
    if not np.isfinite(rt) or rt == 0.0:
        return False, None
    tail = r[-cont_window:] if len(r) >= cont_window else r
    sj = signed_jump_variation(tail)
    mom = float(np.nansum(r[-mom_window:]))
    if np.sign(rt) == np.sign(sj) == np.sign(mom) != 0:
        return True, ("call" if rt > 0 else "put")
    return False, None


def entry_decision(
    returns,
    *,
    in_position: bool,
    bars_since_last_action: int,
    dte_days,
    minutes_to_squareoff: int,
    params: JumpEntryParams = JumpEntryParams(),
) -> dict:
    """Open a backspread iff a confirmed jump-continuation fires and all guards pass. Act at the NEXT bar."""
    if in_position:
        return {"enter": False, "direction": None, "reason": "one position already open"}
    if bars_since_last_action < params.cooldown_bars:
        return {"enter": False, "direction": None, "reason": "cooldown"}
    if dte_days is not None and dte_days < params.min_dte_days:
        return {"enter": False, "direction": None, "reason": f"DTE {dte_days} < {params.min_dte_days} (expiry-eve)"}
    if minutes_to_squareoff < params.min_hold_minutes:
        return {"enter": False, "direction": None, "reason": "too late in session"}
    jumped, direction = confirmed_jump(
        returns, alpha=params.alpha, cont_window=params.cont_window, mom_window=params.mom_window
    )
    if not jumped:
        return {"enter": False, "direction": None, "reason": "no confirmed jump-continuation"}
    return {"enter": True, "direction": direction, "reason": f"confirmed {direction} jump-continuation; enter next bar"}


def exit_decision(
    minutes_to_squareoff: int,
    pnl_fraction,
    *,
    reversed_into_middle: bool = False,
    thesis_invalidated: bool = False,
    greeks_flags: dict | None = None,
    take_profit_r=None,
) -> dict:
    """Precedence: EOD square-off > continuation take-profit > middle-zone reversal > drive-decay. No price stop."""
    if minutes_to_squareoff <= 0:
        return {"exit": True, "reason": "EOD square-off (close short before expiry; avoid exercise STT)"}
    if take_profit_r is not None and pnl_fraction is not None and pnl_fraction >= take_profit_r:
        return {"exit": True, "reason": f"continuation take-profit (+{pnl_fraction:.0%})"}
    if reversed_into_middle and thesis_invalidated:
        return {"exit": True, "reason": "middle-zone reversal + thesis invalidated"}
    if greeks_flags and (greeks_flags.get("theta_dominates") or greeks_flags.get("vega_dead")):
        return {"exit": True, "reason": "drive-decay: convexity stalled"}
    return {"exit": False, "reason": "hold the continuation"}
