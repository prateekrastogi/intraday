"""EOD overnight->open long-vol harvest -- validated NIFTY strategy, HARDENED per the math-soundness review.

Thesis (bt11/bt12, 9mo NIFTY): NIFTY intraday vol is FRONT-LOADED (open 09:15-09:45 annualized realized
vol ~0.115 vs ~0.07 rest-of-day) and CLUSTERS. Forecast next-day vol from recent realized vol (daily
Corsi HAR-RV, walk-forward); on days where that forecast beats the close ATM IV by more than COSTS, BUY
the ATM straddle (or slight-OTM strangle when vol is very cheap) at the CLOSE, hold overnight, exit after
the opening expansion (~10:15). Gated net +2.75% (exit 10:15) in-sample -- TREAT AS OPTIMISTIC: it is a
conditional-on-selection mean and must be re-validated with the gate/shrinkage below.

--- ENTRY GATE (replaces the unsound trailing-quantile; review: entry-threshold) ---
A trailing top-quartile trades a constant ~25% of days regardless of edge sign (fires even when the
selected edge is negative) and is not a break-even. We instead use an ABSOLUTE cost hurdle in annualized-
vol units (volarb.strategy.vol_gate.structure_cost_hurdle) and SHRINK the forecast-IV gap for HAR
estimation error (errors-in-variables / regression-to-the-mean): selecting on a noisy estimate enriches
the upward error, so the realized edge ~= the CENTERED POSTERIOR  m + lambda*(gap - m), with
lambda = var(realized_edge)/var(estimated_gap) in [0,1] and m = mean(prior estimated gaps). Enter iff
corrected_edge > cost_hurdle. The SAME corrected_edge (not the raw gap) feeds the Kelly mean, so the
winner's-curse is not compounded into over-sizing.

--- EXIT (review: exit-vs-optimal-stopping = SOUND time/region family, refined) ---
The close->open mark is a ONE-SHOT, finite-horizon, transient convex payoff -- NOT a stationary recurrent
OU. So the Leung-Li take-profit boundary b* (risk.take_profit / risk.exit_decision) is the WRONG tool and
is LEFT UNWIRED for this leg BY DESIGN (note: risk.fit_ou's ok flag is NOT a safe rail -- it false-
positives ~99.7% on the trending overnight mark, so this must be enforced here, not assumed). The finite-
horizon optimum is exit at min(drive-decay crossover, fixed-time backstop): exit on the FIRST of
(i) theta_dominates -- marginal gamma rent <= theta (risk.greeks_envelope), the argmax of E[mark] when the
net vol-rate crosses zero inside the window; and (ii) the fixed exit_minute backstop -- itself the exact
argmax (right endpoint) when the net rate stays positive to T. NO price stop (convex defined-risk payoff;
Glynn-Iglehart -- a stop truncates the convex right tail). The fixed-fraction take_profit stays OFF for the
same reason (exposed only for live tuning).

--- SIZING (replaces the dimensionally-incoherent margin base; review: kelly-conditional-coherence) ---
For long premium the AT-RISK unit is the PREMIUM (max loss), so lots = floor(f*capital/premium_per_lot),
NOT /margin (margin is a capital-lockup constraint kept as a SEPARATE hard cap). f = fractional_kelly on
the CONDITIONAL gated-edge RETURN moments (P&L / premium-at-risk); fractional_kelly already applies the
t-stat estimation shrink, so we add only ONE behavioral haircut lam=0.5 (half-Kelly) -- NOT 0.25 stacked
on the t-shrink, which over-shrinks to ~0.15. Total premium across ALL open overnight positions is capped
at max_premium_frac of capital (AGGREGATE, via committed_premium). NOTE: f=mu/var is the log-quadratic
approximation; the exact growth-optimal f solving E[r/(1+f r)]=0 over the empirical conditional P&L is the
ideal refinement once a multi-regime P&L sample exists.

--- RISK CONTROLS ---
defined-risk (max loss = premium, an at-expiry bound; the realized close->10:15 loss is strictly smaller,
so the premium cap is deliberately conservative) + AGGREGATE premium-at-risk cap + concentration cap +
EVENT/DTE exclusion (skip expiry-eve and scheduled high-gap nights -- the kill-switch is structurally
blind overnight) + entry greeks check (delta-neutral / long-vega) + daily kill-switch.

Units: forecast_vol and atm_iv are annualized vol FRACTIONS on the volarb.pricing.greeks calendar-minute
basis. CRITICAL: the daily-HAR output is a per-DAY realized VARIANCE -> annualize with the TRADING-DAY
count via :func:`annualized_daily_vol` (= sqrt(var*252)), matching how ATM IV is quoted (realized ~0.08 vs
IV ~0.11 = the VRP). Do NOT feed a daily RV to the intraday calendar-minute annualize_rv(var, 375) -- it
treats a per-day variance as a 375-minute window and OVER-annualizes vol ~2.4x (spuriously forecast >> IV;
caught in re-validation). Use har_rv.walk_forward_daily_forecast then annualized_daily_vol on each. All gate history (prior gaps /
realized edges) MUST be strictly past.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from volarb.strategy.risk import fractional_kelly, greeks_envelope, kill_switch

TRADING_DAYS_PER_YEAR = 252  # a DAILY RV annualizes with the trading-day count (matches ATM IV), NOT calendar-minutes
GATE_CUSHION = 1.5     # VRP cushion on the cost hurdle (vol_gate.VRP_CUSHION)
EXIT_MINUTE = 60       # fixed-time backstop: square-off this many minutes after the open (10:15)
STRANGLE_IV_PCT = 0.15 # very cheap vol (percentile rank) -> wider, cheaper OTM strangle
MAX_PREMIUM_FRAC = 0.05  # AGGREGATE overnight premium-at-risk cap, fraction of capital
MAX_OVERNIGHT_POSITIONS = 1
MIN_DTE_DAYS = 2       # skip expiry-eve/expiry-day entries (pin risk, ITM exercise STT, wide exit spreads)
LAM_KELLY = 0.5        # single behavioral haircut; fractional_kelly already applies the t-stat shrink


@dataclass(frozen=True)
class EodParams:
    """Live parameters for the EOD overnight->open harvest (defaulted to the reviewed values)."""

    exit_minute: int = EXIT_MINUTE
    strangle_iv_pct: float = STRANGLE_IV_PCT
    max_premium_frac: float = MAX_PREMIUM_FRAC
    max_overnight_positions: int = MAX_OVERNIGHT_POSITIONS
    min_dte_days: int = MIN_DTE_DAYS
    lam_kelly: float = LAM_KELLY
    cushion: float = GATE_CUSHION


def overnight_gap(forecast_vol: float, atm_iv_close: float) -> float:
    """Forecast-vs-implied edge for the overnight->open window (annualized vol, same calendar-minute basis).

    ``forecast_vol`` MUST be the daily-HAR variance annualized via :func:`annualized_daily_vol`
    (sqrt(var*252)), NOT the raw variance and NOT the intraday calendar-minute annualize_rv on a daily RV.
    """
    return float(forecast_vol) - float(atm_iv_close)


def annualized_daily_vol(har_variance: float) -> float:
    """Annualize a daily HAR realized VARIANCE to a vol comparable with ATM IV: sqrt(var * 252).

    The daily RV is ONE session variance, so it annualizes with the trading-day count
    (TRADING_DAYS_PER_YEAR) -> ~0.08 vs IV ~0.11 (the VRP). The intraday calendar-minute
    annualize_rv(var, 375) over-annualizes a per-day variance ~2.4x and spuriously inverts the gap.
    """
    return float(np.sqrt(max(float(har_variance), 0.0) * TRADING_DAYS_PER_YEAR))


def reliability_lambda(prior_est_gaps, prior_realized_edges) -> float:
    """Errors-in-variables shrink factor lambda = var(realized_edge)/var(estimated_gap), clipped to [0,1].

    Both inputs MUST be strictly-past, length-matched (estimated gap at entry, realized edge ex-post).
    lambda=1 => forecast is noiseless (no shrink); lambda->0 => forecast is pure noise (shrink to the mean).
    Returns 1.0 on insufficient/degenerate history (no shrink) so the cold-start is conservative-neutral.
    """
    est = np.asarray([x for x in prior_est_gaps if x is not None and np.isfinite(x)], dtype=float)
    real = np.asarray([x for x in prior_realized_edges if x is not None and np.isfinite(x)], dtype=float)
    n = min(est.size, real.size)
    if n < 15:
        return 1.0
    ve = float(np.var(est[-n:], ddof=1))
    if ve <= 0:
        return 1.0
    return float(min(1.0, max(0.0, np.var(real[-n:], ddof=1) / ve)))


def corrected_edge(gap: float, prior_est_gaps, prior_realized_edges) -> float:
    """Shrinkage-corrected (centered-posterior) expected edge: m + lambda*(gap - m).

    De-biases the regression-to-the-mean from selecting on a noisy HAR forecast. m = mean(prior estimated
    gaps); lambda from :func:`reliability_lambda`. Use this (NOT the raw gap) as both the gate quantity and
    the Kelly mean. All priors strictly past (no-lookahead is the caller's contract).
    """
    est = np.asarray([x for x in prior_est_gaps if x is not None and np.isfinite(x)], dtype=float)
    if est.size < 15:
        return float(gap)  # cold start: no correction
    m = float(np.mean(est))
    lam = reliability_lambda(prior_est_gaps, prior_realized_edges)
    return m + lam * (float(gap) - m)


def entry_decision(
    corrected_edge_vol: float,
    cost_hurdle: float,
    iv_percentile_rank: float | None,
    *,
    dte_days: int | None = None,
    event_flag: bool = False,
    params: EodParams = EodParams(),
) -> dict:
    """Open a delta-neutral long-vol overnight position iff the SHRUNK edge clears the ABSOLUTE cost hurdle.

    Gate: enter iff corrected_edge_vol > cost_hurdle (both annualized vol; cost_hurdle from
    vol_gate.structure_cost_hurdle = frictions/vega x cushion), AND not a scheduled-event night, AND the
    option has >= min_dte_days to expiry (skip expiry-eve pin risk). Structure: ATM STRADDLE by default; a
    slight-OTM (+/-1 strike) STRANGLE when vol is very cheap (iv_percentile_rank <= strangle_iv_pct).
    ``iv_percentile_rank`` is a [0,1] RANK of close ATM IV vs its trailing history, NOT an IV level.
    Returns {enter, structure, edge, reason}; ``structure`` always populated (intent even on a no-go).
    """
    structure = (
        "strangle"
        if (iv_percentile_rank is not None and iv_percentile_rank <= params.strangle_iv_pct)
        else "straddle"
    )
    base = {"structure": structure, "edge": float(corrected_edge_vol)}
    if event_flag:
        return {"enter": False, "reason": "scheduled high-gap event night excluded", **base}
    if dte_days is not None and dte_days < params.min_dte_days:
        return {"enter": False, "reason": f"DTE {dte_days} < {params.min_dte_days} (expiry-eve excluded)", **base}
    if corrected_edge_vol <= cost_hurdle:
        return {"enter": False, "reason": f"edge {corrected_edge_vol:.4f} <= hurdle {cost_hurdle:.4f}", **base}
    return {
        "enter": True,
        "reason": f"edge {corrected_edge_vol:.4f} > hurdle {cost_hurdle:.4f}; long {structure}",
        **base,
    }


def exit_decision(
    minutes_since_open: int,
    params: EodParams = EodParams(),
    *,
    theta_dominates: bool = False,
    pnl_fraction: float | None = None,
    take_profit_frac: float | None = None,
) -> dict:
    """Exit on the FIRST of: drive-decay crossover (theta_dominates) or the fixed exit_minute backstop.

    min(crossover, T) is the finite-horizon argmax of E[mark] for the one-shot convex payoff (theta_dominates
    = marginal gamma rent <= theta, from risk.greeks_envelope, is the pathwise net-vol-rate<0 signal; the
    fixed exit_minute is the exact argmax when the net rate stays positive to T). NO price stop. The optional
    fixed-fraction take-profit truncates the convex right tail (keep OFF; exposed for live tuning only).
    """
    if take_profit_frac is not None and pnl_fraction is not None and pnl_fraction >= take_profit_frac:
        return {"exit": True, "reason": f"take-profit +{pnl_fraction:.0%} >= {take_profit_frac:.0%} (tail-truncating)"}
    if theta_dominates:
        return {"exit": True, "reason": "drive-decay: marginal gamma rent <= theta (E[mark] argmax)"}
    if minutes_since_open >= params.exit_minute:
        return {"exit": True, "reason": f"square-off {params.exit_minute}min after open (fixed backstop)"}
    return {"exit": False, "reason": "hold through the opening-vol window"}


def entry_greeks_ok(
    vega_entry: float, theta_bleed: float, gamma_rent: float, net_delta: float, *, delta_max: float = 0.15
) -> dict:
    """Entry-time greeks check: long-vega present and the structure is ~delta-neutral (reuses greeks_envelope).

    Rejects entry if vega is non-positive (not actually long vol), theta already dominates gamma rent at
    entry (no opening drive to harvest), or |net_delta| exceeds delta_max (not a clean magnitude bet).
    """
    if vega_entry <= 0:
        return {"ok": False, "reason": "non-positive vega at entry (not long vol)"}
    env = greeks_envelope(
        vega_entry, vega_entry, theta_bleed, gamma_rent, net_delta,
        vega_floor_frac=0.5, delta_max=delta_max,
    )
    if env["theta_dominates"]:
        return {"ok": False, "reason": "theta already dominates gamma rent at entry (no drive)"}
    if env["delta_breach"]:
        return {"ok": False, "reason": f"|net_delta| > {delta_max} (not delta-neutral)"}
    return {"ok": True, "reason": "long-vega, delta-neutral, gamma rent covers theta"}


def position_lots(
    mean_ret: float,
    var_ret: float,
    n_trades: int,
    *,
    capital: float,
    premium_per_lot: float,
    margin_per_lot: float,
    committed_premium: float = 0.0,
    params: EodParams = EodParams(),
) -> int:
    """Quarter-to-half-Kelly lots sized on PREMIUM-at-risk, capped by AGGREGATE premium budget and margin.

    ``mean_ret``/``var_ret`` are CONDITIONAL (gated-day) per-trade RETURN moments = P&L / premium-at-risk
    (so f = fractional_kelly is a dimensionless fraction of the premium-at-risk unit). lots =
    floor(f*capital/premium_per_lot). Margin is a SEPARATE hard cap (not the Kelly denominator). The 5%
    premium cap is AGGREGATE: it subtracts ``committed_premium`` (premium already at risk in open overnight
    positions) so total premium-at-risk <= max_premium_frac*capital holds across the book.
    """
    if premium_per_lot <= 0 or margin_per_lot <= 0 or capital <= 0:
        return 0
    f = fractional_kelly(mean_ret, var_ret, n_trades, params.lam_kelly)
    kelly_lots = int(np.floor(f * capital / premium_per_lot))
    margin_cap = int(np.floor(capital / margin_per_lot))
    remaining_premium = params.max_premium_frac * capital - max(0.0, committed_premium)
    premium_cap = int(np.floor(max(0.0, remaining_premium) / premium_per_lot))
    return int(max(0, min(kelly_lots, margin_cap, premium_cap)))


def risk_gate(
    day_pnl: float, kill_rupees: float, open_positions: int, params: EodParams = EodParams()
) -> dict:
    """Pre-trade risk gate: daily kill-switch (drawdown) + overnight-concentration cap.

    NOTE: the kill-switch acts on realized intraday P&L and is structurally blind to an overnight gap; the
    event/DTE exclusion in :func:`entry_decision` and the aggregate premium cap are the overnight controls.
    """
    if kill_switch(day_pnl, kill_rupees):
        return {"allow": False, "reason": f"kill-switch: day P&L {day_pnl:.0f} <= -{abs(kill_rupees):.0f}"}
    if open_positions >= params.max_overnight_positions:
        return {"allow": False, "reason": f"overnight-position cap ({open_positions}/{params.max_overnight_positions})"}
    return {"allow": True, "reason": "within drawdown + concentration limits"}
