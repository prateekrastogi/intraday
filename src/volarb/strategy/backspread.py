"""Call/put RATIO BACKSPREAD construction for the jump-continuation strategy.

SELL n_short near-money + BUY n_long > n_short further-OTM options of the SAME type (calls for an
up-jump, puts for a down-jump) -> net LONG convexity in the continuation direction. Per the design
decision we PRIORITIZE CONVEXITY and allow a small net DEBIT: the long wings must carry real gamma/vega
(HARD gate), even at a small debit D (wrong-way loss capped at -D).

Payoff (call backspread, up-jump, Ks<Kl), net premium PAID D (>0 debit, <0 credit):
    S_T <= Ks      : -D                              (wrong-way down: lose the small debit)
    Ks < S_T <= Kl : -D - n_s*(S_T-Ks)               (middle zone; worst at S_T=Kl)
    S_T > Kl       : -D - n_s*(S_T-Ks) + n_l*(S_T-Kl) -> slope (n_l-n_s)>0 (convex continuation)
    MaxLoss = D + n_s*(Kl-Ks) at S_T=Kl ; upper breakeven = Kl + MaxLoss/(n_l-n_s)
Put backspread mirrors (Ks>Kl, longs further OTM down).

EXECUTION ORDER: legs are emitted LONGS-FIRST so the live broker books the long hedge before the short
and charges the reduced (hedged) margin on the short. NSE index options are European/cash-settled (no
early assignment); the short near-money leg is the one most likely ITM after a continuation, so the
strategy squares off before expiry (avoids exercise STT).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from volarb.pricing.greeks import DEFAULT_VEGA_FLOOR, iv_greeks, vega_per_vol


@dataclass(frozen=True)
class Leg:
    """One option leg. ``qty`` is signed lots/units: +long, -short."""

    right: str          # "C" or "P"
    strike: float
    qty: int
    entry_price: float


@dataclass(frozen=True)
class Backspread:
    direction: str                  # "call" (up-jump) or "put" (down-jump)
    legs: tuple                     # (long_leg, short_leg) -- LONGS FIRST (execution order)
    n_short: int
    n_long: int
    short_strike: float
    long_strike: float
    debit: float                    # net premium paid per ratio-unit, premium points (>0 debit, <0 credit)
    max_loss: float                 # D + n_s*|Ks-Kl| (premium points/ratio-unit); *lot_size = rupees
    upper_breakeven: float
    net_delta: float
    net_gamma: float
    net_vega: float                 # per-1.0-vol
    net_theta: float
    reason: str


@dataclass(frozen=True)
class BackspreadParams:
    delta_long_min: float = 0.10    # long wing must have |delta| >= this (not a worthless lottery)
    debit_max_frac: float = 0.50    # net debit cap as a fraction of the short premium (small)
    n_short: int = 1
    max_n_long: int = 5
    vega_floor: float = DEFAULT_VEGA_FLOOR


def _greeks_chain(chain, forward, t_years, is_call):
    out = {}
    for strike, px in chain:
        g = iv_greeks(px, forward, strike, t_years, is_call)
        if g is not None:
            out[float(strike)] = (float(px), g)
    return out


def construct_backspread(direction, chain, forward, t_years, params=BackspreadParams()):
    """Select (Ks, Kl, n_short:n_long) for the continuation direction; return a Backspread or None (abstain).

    ``direction``: "call" (up-jump) / "put" (down-jump). ``chain``: iterable of (strike, mid_price) for THAT
    right. Convexity is a HARD gate (net_gamma>0, net_vega>0, |long delta|>=delta_long_min); a small net
    debit (<= debit_max_frac * short premium) is allowed. Abstains (None) when the chain can't deliver a
    convex structure -- expect frequent no-trades on short-dated weeklies (favors the monthly).
    """
    is_call = direction == "call"
    q = _greeks_chain(chain, forward, t_years, is_call)
    if len(q) < 2:
        return None
    strikes = sorted(q)
    if is_call:
        ks = max([k for k in strikes if k <= forward] or [strikes[0]])   # ATM/just-ITM
        long_side = [k for k in strikes if k > ks]                       # OTM up
    else:
        ks = min([k for k in strikes if k >= forward] or [strikes[-1]])
        long_side = [k for k in strikes if k < ks]                       # OTM down
    ps, gs = q[ks]
    ns = params.n_short
    best = None
    for kl in long_side:
        pl, gl = q[kl]
        if gl["vega"] < params.vega_floor or abs(gl["delta"]) < params.delta_long_min:
            continue                                                     # worthless-lottery wing -> skip
        if gl["gamma"] <= 0 or gl["vega"] <= 0:
            continue
        nl = max(ns + 1, int(math.floor(ns * max(gs["gamma"] / gl["gamma"], gs["vega"] / gl["vega"]))) + 1)
        if nl > params.max_n_long:
            continue
        debit = nl * pl - ns * ps                                        # net premium paid (premium points)
        if debit > params.debit_max_frac * ps:                           # cap the debit (small)
            continue
        net_delta = nl * gl["delta"] - ns * gs["delta"]
        net_gamma = nl * gl["gamma"] - ns * gs["gamma"]
        net_vega = nl * vega_per_vol(gl) - ns * vega_per_vol(gs)
        if not (net_gamma > 0 and net_vega > 0):
            continue
        if (is_call and net_delta <= 0) or (not is_call and net_delta >= 0):
            continue                                                     # must be directional in the move
        width = abs(kl - ks)
        max_loss = debit + ns * width                                    # worst at S=Kl (signed debit)
        net_theta = nl * gl["theta"] - ns * gs["theta"]
        # objective: max continuation slope (n_l-n_s); tie -> smaller width, then smaller n_l
        key = (nl - ns, -width, -nl)
        if best is None or key > best[0]:
            best = (key, kl, nl, pl, debit, max_loss, net_delta, net_gamma, net_vega, net_theta)
    if best is None:
        return None
    _, kl, nl, pl, debit, max_loss, nd, ng, nv, nt = best
    ube = kl + max_loss / (nl - ns) if (nl - ns) > 0 else float("inf")
    if not is_call:
        ube = kl - max_loss / (nl - ns) if (nl - ns) > 0 else float("-inf")  # lower breakeven for puts
    right = "C" if is_call else "P"
    legs = (Leg(right, kl, +nl, pl), Leg(right, ks, -ns, ps))            # LONGS FIRST, then short
    return Backspread(direction, legs, ns, nl, ks, kl, debit, max_loss, ube, nd, ng, nv, nt,
                      f"{direction} backspread {ns}x{nl} Ks={ks:g} Kl={kl:g} debit={debit:.1f} maxloss={max_loss:.1f}")


def payoff(bs: Backspread, s_t: float, lot_size: int = 1) -> float:
    """Terminal payoff (rupees) of the backspread at underlying ``s_t`` (×lot_size)."""
    ns, nl, ks, kl = bs.n_short, bs.n_long, bs.short_strike, bs.long_strike
    if bs.direction == "call":
        val = -bs.debit - ns * max(s_t - ks, 0.0) + nl * max(s_t - kl, 0.0)
    else:
        val = -bs.debit - ns * max(ks - s_t, 0.0) + nl * max(kl - s_t, 0.0)
    return val * lot_size


def max_loss_rupees(bs: Backspread, lot_size: int) -> float:
    """The structure's defined worst-case loss in rupees (>=0; 0 if the structure cannot lose)."""
    return max(bs.max_loss, 0.0) * lot_size
