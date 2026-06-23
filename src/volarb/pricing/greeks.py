"""IV / greeks via NautilusTrader Black-76 -- the SINGLE inverter (reuse, not vollib).

REUSE nautilus_trader.core.nautilus_pyo3.imply_vol_and_greeks (Rust LetsBeRational): Black-76 with
b=0 and s = the FUTURES forward (index-options-on-futures convention; NO dividend hack). The same
inverter runs in research, backtest and live -> the RV-IV edge axis is computed identically (no drift).
"""

from __future__ import annotations

from nautilus_trader.core.nautilus_pyo3 import imply_vol_and_greeks

YEAR_MINUTES = 365.0 * 24 * 60  # calendar-minute annualization -- be consistent fit<->live
DEFAULT_R = 0.065  # India short rate ~6.5%; tiny effect on weekly IV (small r*T)
DEFAULT_VEGA_FLOOR = 0.5  # below this vega, IV inversion is unstable -> drop the strike


def years_to_expiry(minutes: float) -> float:
    """Calendar-minute time-to-expiry in years (floored so t>0 for the solver)."""
    return max(float(minutes), 1.0) / YEAR_MINUTES


def iv_greeks(price, forward, strike, t_years, is_call, r=DEFAULT_R):
    """Black-76 IV + greeks for one option. None if degenerate (t<=0, price<=intrinsic, solver fails)."""
    intrinsic = max(forward - strike, 0.0) if is_call else max(strike - forward, 0.0)
    if t_years <= 0 or price <= intrinsic + 1e-6:
        return None
    try:
        g = imply_vol_and_greeks(
            s=forward, r=r, b=0.0, is_call=is_call, k=strike, t=t_years, price=price
        )
    except Exception:
        return None
    return {"vol": g.vol, "delta": g.delta, "gamma": g.gamma, "vega": g.vega, "theta": g.theta}


def atm_iv(quotes, forward, t_years, r=DEFAULT_R, vega_floor=DEFAULT_VEGA_FLOOR):
    """ATM IV at K=forward. quotes: iterable of (strike, price, is_call). Vega-filter, then linearly
    interpolate the (strike, vol) points to K=forward. None if <2 usable points."""
    pts = []
    for k, price, is_call in quotes:
        g = iv_greeks(price, forward, k, t_years, is_call, r)
        if g is not None and g["vega"] >= vega_floor:
            pts.append((float(k), g["vol"]))
    if len(pts) < 2:
        return None
    pts.sort()
    ks = [p[0] for p in pts]
    vs = [p[1] for p in pts]
    if forward <= ks[0]:
        return vs[0]
    if forward >= ks[-1]:
        return vs[-1]
    for i in range(1, len(ks)):
        if ks[i] >= forward:
            w = (forward - ks[i - 1]) / (ks[i] - ks[i - 1])
            return vs[i - 1] * (1 - w) + vs[i] * w
    return None
