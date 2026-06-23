"""IV / greeks over NautilusTrader's built-in Black-76 -- the SINGLE inverter (reuse, not vollib).

REUSE: nautilus_trader.core.nautilus_pyo3.imply_vol_and_greeks (Rust LetsBeRational). Black-76 via b=0
with s = the NIFTY FUTURES forward. The SAME implementation runs in research, backtest and live, so the
RV-IV edge axis is computed identically everywhere (no two-inverter drift). Verified on the VM:
  imply_vol_and_greeks(s=F, r=0.065, b=0.0, is_call=..., k=K, t=..., price=...) -> .vol/.delta/.gamma/.vega/.theta
"""
# TODO (thin wrapper + guards -- NO hand-rolled solver, NO vollib):
#   - invert per (strike, minute): s = futures forward, t = minutes_to_expiry / 525600, r ~ 0.065
#   - GUARDS: drop rows below a vega floor; floor t (~1e-3 yr); skip price <= intrinsic; drop last ~30-45 min of expiry day
#   - ATM IV = vega-filtered smile interpolated in strike to K=F, per minute per expiry
#   - term slope = ATMIV(e2) - ATMIV(e1); cross-check level vs INDVIX. tests/test_greeks.py
