"""Pure entry/exit decision logic for the long-vol straddle/strangle strategy.

Library-agnostic (no NautilusTrader, no I/O): the Strategy wiring that turns these
decisions into orders lives in step 5/6. Everything here is a deterministic function
of numbers the caller already has.

The edge axis is the RV-IV spread (forecast realized vol minus self-built ATM IV /
India VIX), expressed in the SAME calendar-minute-annualized vol units on both sides.
We only trade the residual that beats both round-trip costs (the `cost_hurdle`) and the
regime variance-risk-premium, and only when Johnson-style term-structure / IV-percentile
gating says we are in the documented long-vol-positive pocket. The bet is on MAGNITUDE,
never direction -- so structures are delta-neutral (straddle / strangle) and unlevered.
"""

from __future__ import annotations

from dataclasses import dataclass

from volarb.costs.nse_fees import round_trip_charges

# --- regime gate thresholds (Johnson 2017 SLOPE; documented long-vol pocket). ---
# Kept to a handful of named constants so the live param count stays <= 4-6.
IV_PCT_FLOOR = 0.30  # only buy vol when ATM IV percentile is in the cheap lower third.
SLOPE_INVERSION = 0.0  # front IV > next-expiry IV (slope < 0) = inverted term structure.
STRANGLE_IV_PCT = 0.15  # very cheap vol -> wider, cheaper strangle over the ATM straddle.

# Multiplier on the cost hurdle that proxies the regime variance-risk-premium (VRP):
# the spread must beat costs AND leave a VRP cushion before we pay up for gamma.
VRP_CUSHION = 1.5


@dataclass(frozen=True)
class GateState:
    """Live position state consumed by `exit_decision` (all optional / lib-agnostic)."""

    minutes_held: int
    minutes_to_square_off: int  # minutes remaining until the hard intraday square-off.
    entry_spread: float  # RV-IV spread (annualized vol) captured at entry.
    current_spread: float  # latest RV-IV spread in the same units.
    pnl_fraction: float  # mark-to-market P&L as a fraction of premium paid (signed).
    stop_loss_fraction: float = 0.40  # square off if we have lost this fraction of premium.
    spread_collapse_ratio: float = 0.30  # exit if edge has decayed below this * entry edge.


def cost_hurdle(
    premium: float,
    qty: int,
    trade_date,
    *,
    vega: float | None = None,
    t_years: float | None = None,
    cushion: float = VRP_CUSHION,
) -> float:
    """Minimum RV-IV spread the edge must clear before an entry is worthwhile.

    The hurdle is derived from real NSE round-trip frictions via
    `volarb.costs.nse_fees.round_trip_charges` (~0.8% of premium on an ATM NIFTY
    option). We assume a flat round-trip on the premium just paid (premium_in ==
    premium_out == `premium`) for the per-leg charge estimate.

    Units:
      * Default -> a PREMIUM FRACTION: charges / premium notional, scaled by `cushion`
        to embed the regime VRP. Use this when gating the spread expressed as a
        premium fraction.
      * If `vega` and `t_years` are supplied -> an ANNUALIZED-VOL hurdle in the same
        calendar-minute basis as ATM IV, so it can be differenced directly against the
        RV-IV spread. A 1-vol-point move changes premium by `vega` per unit, so the
        vol-points needed to cover charges is charges / (vega * qty), i.e. we invert
        the option's local vega sensitivity.

    Reuses the cost foundation; adds no pricing of its own.
    """
    if premium <= 0 or qty <= 0:
        raise ValueError("premium and qty must be positive")

    charges = round_trip_charges(premium, premium, qty, trade_date)
    notional = premium * qty

    if vega is not None and t_years is not None:
        if vega <= 0:
            raise ValueError("vega must be positive to convert to a vol-unit hurdle")
        # charges are total INR; vega is premium-change per 1.0 vol (per contract).
        # vol-fraction needed to recoup costs = total_charges / (vega * qty).
        vol_hurdle = charges.total / (vega * qty)
        return cushion * vol_hurdle

    # premium-fraction hurdle.
    return cushion * (charges.total / notional)


def _regime_ok(regime: dict) -> tuple[bool, str]:
    """Johnson-style long-vol gate: cheap and/or inverted term structure only.

    A jump in the recent regime (Lee-Mykland flag) is treated as confirming feature,
    NOT a reactive trigger -- it can relax the IV-percentile requirement slightly but
    cannot by itself open a trade.
    """
    iv_pct = regime.get("iv_pct")
    slope = regime.get("slope")
    jump_flag = bool(regime.get("jump_flag", False))

    if iv_pct is None or slope is None:
        return False, "regime incomplete (need iv_pct and slope)"

    inverted = slope < SLOPE_INVERSION
    cheap = iv_pct <= IV_PCT_FLOOR
    # a recent jump widens the cheap band a little (regime context, not a trigger).
    cheap_jump = jump_flag and iv_pct <= IV_PCT_FLOOR + 0.10

    if inverted or cheap or cheap_jump:
        bits = []
        if inverted:
            bits.append("inverted term structure")
        if cheap:
            bits.append(f"iv_pct<={IV_PCT_FLOOR:.2f}")
        if cheap_jump:
            bits.append("jump-relaxed cheap band")
        return True, "; ".join(bits)

    return False, f"regime rich (iv_pct={iv_pct:.2f}, slope={slope:+.3f})"


def entry_decision(spread: float, regime: dict, hurdle: float) -> dict:
    """Decide whether to open a delta-neutral long-vol position.

    Enter only when ALL hold:
      1. spread > hurdle           (RV-IV edge beats costs + VRP cushion),
      2. Johnson regime gate holds (cheap or inverted term structure),
      3. it is a magnitude bet     (delta-neutral structure, by construction here).

    Structure selection (still <= the param budget):
      * very cheap vol (iv_pct <= STRANGLE_IV_PCT) -> STRANGLE (wider, cheaper wings),
      * otherwise                                   -> STRADDLE (max gamma at ATM).

    `regime` carries {iv_pct, slope, jump_flag}. Returns
    {enter, structure, reason}. `structure` is always populated (the structure we
    *would* trade) so the caller can log intent even on a no-go.
    """
    iv_pct = regime.get("iv_pct")
    structure = "strangle" if (iv_pct is not None and iv_pct <= STRANGLE_IV_PCT) else "straddle"

    if spread <= hurdle:
        return {
            "enter": False,
            "structure": structure,
            "reason": f"spread {spread:.4f} <= hurdle {hurdle:.4f}",
        }

    ok, regime_reason = _regime_ok(regime)
    if not ok:
        return {"enter": False, "structure": structure, "reason": regime_reason}

    return {
        "enter": True,
        "structure": structure,
        "reason": (
            f"spread {spread:.4f} > hurdle {hurdle:.4f}; {regime_reason}; "
            f"magnitude bet ({structure}, delta-neutral)"
        ),
    }


def exit_decision(state: GateState) -> dict:
    """Square-off timer / stop-loss / spread-collapse exit.

    Precedence: hard square-off timer first (intraday discipline -- never carry the
    futures roll or hold past the close), then stop-loss, then edge collapse.
    Returns {exit, reason}.
    """
    if state.minutes_to_square_off <= 0:
        return {"exit": True, "reason": "square-off timer: forced intraday exit"}

    if state.pnl_fraction <= -abs(state.stop_loss_fraction):
        return {
            "exit": True,
            "reason": (
                f"stop-loss: pnl {state.pnl_fraction:.2%} <= "
                f"-{abs(state.stop_loss_fraction):.2%} of premium"
            ),
        }

    # spread collapse: the RV-IV edge we entered on has largely mean-reverted away.
    if state.entry_spread > 0:
        collapse_level = state.spread_collapse_ratio * state.entry_spread
        if state.current_spread < collapse_level:
            return {
                "exit": True,
                "reason": (
                    f"spread collapse: edge {state.current_spread:.4f} < "
                    f"{state.spread_collapse_ratio:.0%} of entry {state.entry_spread:.4f}"
                ),
            }

    return {"exit": False, "reason": "hold: timer/stop/edge all within bounds"}
