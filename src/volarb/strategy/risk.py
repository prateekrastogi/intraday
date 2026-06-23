"""Exit + risk-management for the intraday long-vol straddle (PLAN step 4b).

Implements the Leung & Li (2015, IJTAF 18(3); arXiv:1411.5062) optimal mean-reversion
exit math applied to the MEAN-REVERTING RV-IV edge state (NOT the random-walk straddle
price -- Carr-Madan / Glynn-Iglehart), plus fractional-Kelly sizing and the model-free
risk floor (kill-switch, greeks envelope, same-day square-off).

Equation references are to arXiv:1411.5062 (verified against the source PDF):
  OU dX = mu(theta - X)dt + sigma dB                                 (2.1)
  generator L u = r u ; fundamental solutions F, G                   (3.3)/(3.4)
  take-profit, no stop:        F(b*) = (b* - c) F'(b*)               (4.3)
  no-trade / immediate-liq threshold  L* = (mu*theta + r*c)/(mu+r)   (4.1), Cor 5.2
  take-profit with stop-loss L:  Eq (5.5)                            (5.5)
  Prop 5.3: b*_L strictly DECREASES as L increases
  Prop 4.3: b* (no stop) INCREASES in transaction cost c
  Remark 4.7 (mu->0, Brownian):  b* = c + sigma / sqrt(2r)

The reward in the paper is h(x)=x-c (you receive the spread value on exit). Our straddle
exit yields the OPTION P&L, so the state X here is the position's accumulated edge / mark
(in spread units), and the spread->P&L map is the flagged transfer heuristic to validate OOS.

Numerical note: the F/G integrals are well-conditioned for r/mu ~ O(1). For an intraday
problem with very fast mean-reversion (r/mu -> 0) the u^(r/mu-1) singularity stiffens; a
parabolic-cylinder (scipy.special.pbdv) representation is the robust upgrade (TODO).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import brentq
from scipy.special import pbdv
from statsmodels.tsa.stattools import adfuller

__all__ = [
    "OUParams",
    "fit_ou",
    "F",
    "Fp",
    "G",
    "Gp",
    "l_star",
    "take_profit",
    "take_profit_stoploss",
    "bm_take_profit",
    "fractional_kelly",
    "kelly_lots",
    "kill_switch",
    "greeks_envelope",
    "exit_decision",
]


# ----------------------------------------------------------------------------- OU fit
@dataclass(frozen=True)
class OUParams:
    mu: float  # mean-reversion speed (>0)
    theta: float  # long-run mean
    sigma: float  # volatility (>0)
    ok: bool  # True if a valid mean-reverting fit (mu>0, sigma>0)


def fit_ou(x: np.ndarray, dt: float) -> OUParams:
    """MLE of the OU params from a spread series sampled at spacing dt (Leung-Li Eq 2.2).

    Closed-form via the AR(1) representation x_i = a + b x_{i-1} + eps:
        b = e^{-mu dt}  ->  mu = -ln(b)/dt ;  theta = a/(1-b) ;  sigma from residual var.
    Returns ok=False (NOT mean-reverting) if b not in (0,1) -- then use kill-switch only
    (Glynn-Iglehart: no threshold on a non-mean-reverting state).
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size < 30:
        return OUParams(np.nan, np.nan, np.nan, ok=False)
    xp, xc = x[:-1], x[1:]
    b, a = np.polyfit(xp, xc, 1)  # xc ~ b*xp + a
    if not (0.0 < b < 1.0):  # b<=0 or b>=1 => no stationary mean reversion
        return OUParams(np.nan, np.nan, np.nan, ok=False)
    mu = -np.log(b) / dt
    theta = a / (1.0 - b)
    resid = xc - (b * xp + a)
    var_resid = float(np.var(resid, ddof=2)) if resid.size > 2 else float(np.var(resid))
    sigma2 = var_resid * 2.0 * mu / (1.0 - b * b)  # invert sigma~^2 = sigma^2(1-e^{-2mu dt})/2mu
    sigma = float(np.sqrt(max(sigma2, 1e-18)))
    # stationarity gate: a finite random-walk sample still fits b in (0,1); require the
    # unit-root null to be REJECTED (ADF) before treating the spread as mean-reverting.
    try:
        stationary = adfuller(x, regression="c", autolag="AIC")[1] <= 0.10
    except Exception:
        stationary = False
    return OUParams(float(mu), float(theta), sigma, ok=bool(stationary))


# ------------------------------------------------------- F, G fundamental solutions
def _k(p: OUParams) -> float:
    return np.sqrt(2.0 * p.mu / p.sigma**2)


def _quad0inf(fn) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        val, _ = quad(fn, 0.0, np.inf, limit=200)
    return val


def F(x: float, p: OUParams, r: float) -> float:
    """Increasing convex fundamental solution of L u = r u (Eq 3.3)."""
    k, pw = _k(p), r / p.mu - 1.0
    return _quad0inf(lambda u: u**pw * np.exp(k * (x - p.theta) * u - 0.5 * u * u))


def Fp(x: float, p: OUParams, r: float) -> float:
    """F'(x) > 0 (differentiate Eq 3.3: bring down k*u -> power r/mu)."""
    k, pw = _k(p), r / p.mu
    return k * _quad0inf(lambda u: u**pw * np.exp(k * (x - p.theta) * u - 0.5 * u * u))


def G(x: float, p: OUParams, r: float) -> float:
    """Decreasing convex fundamental solution of L u = r u (Eq 3.4)."""
    k, pw = _k(p), r / p.mu - 1.0
    return _quad0inf(lambda u: u**pw * np.exp(k * (p.theta - x) * u - 0.5 * u * u))


def Gp(x: float, p: OUParams, r: float) -> float:
    """G'(x) < 0."""
    k, pw = _k(p), r / p.mu
    return -k * _quad0inf(lambda u: u**pw * np.exp(k * (p.theta - x) * u - 0.5 * u * u))


# --------------------------------------------------------------- thresholds
def l_star(p: OUParams, r: float, c: float) -> float:
    """No-trade / immediate-liquidation threshold L* = (mu*theta + r*c)/(mu+r) (Eq 4.1)."""
    return (p.mu * p.theta + r * c) / (p.mu + r)


def _scan_root(fn, lo: float, hi: float, n: int = 60):
    """Find the first sign-change root of fn on [lo, hi] (robust bracketing) or None."""
    xs = np.linspace(lo, hi, n)
    prev_x, prev_f = xs[0], fn(xs[0])
    for cur_x in xs[1:]:
        cur_f = fn(cur_x)
        if prev_f == 0.0:
            return prev_x
        if prev_f * cur_f < 0.0:
            return float(brentq(fn, prev_x, cur_x))
        prev_x, prev_f = cur_x, cur_f
    return None


def take_profit(p: OUParams, r: float, c: float, method: str = "pc") -> float | None:
    """Optimal take-profit b* with NO stop-loss: solves F(b) = (b-c)F'(b)  (Eq 4.3)."""
    ff, fpf, _, _ = _funcs(method)
    hi = _scan_hi(p, c)
    return _scan_root(lambda b: ff(b, p, r) - (b - c) * fpf(b, p, r), c + 1e-6, hi)


def take_profit_stoploss(
    p: OUParams, r: float, c: float, ll: float, method: str = "pc"
) -> float | None:
    """Optimal take-profit b*_L WITH stop-loss at ll: solves Eq (5.5), valid for ll < L*.

    Returns None if ll >= L* (Cor 5.2: delay region empty -> liquidate immediately / don't enter).
    """
    Ls = l_star(p, r, c)
    if ll >= Ls:
        return None
    ff, fpf, gf, gpf = _funcs(method)
    FL, GL = ff(ll, p, r), gf(ll, p, r)

    def res(b):
        Fb, Gb, Fpb, Gpb = ff(b, p, r), gf(b, p, r), fpf(b, p, r), gpf(b, p, r)
        lhs = ((ll - c) * Gb - (b - c) * GL) * Fpb + ((b - c) * FL - (ll - c) * Fb) * Gpb
        return lhs - (Gb * FL - GL * Fb)

    hi = _scan_hi(p, c)
    return _scan_root(res, Ls + 1e-6, hi)


def bm_take_profit(c: float, sigma: float, r: float) -> float:
    """Brownian-motion degenerate take-profit b* = c + sigma/sqrt(2r)  (Remark 4.7, mu->0)."""
    return c + sigma / np.sqrt(2.0 * r)


# --------------------------------------------------------------- sizing (fractional Kelly)
def fractional_kelly(
    mean_pnl: float, var_pnl: float, n_trades: int, lam_base: float = 0.5
) -> float:
    """Fractional-Kelly bet fraction on a thin, ESTIMATED edge.

    full Kelly f = mean/var ; shrink by lam_base AND by max(0, 1 - 1/t^2), t = mean/SE(mean),
    SE = sqrt(var/n). No edge / weak t-stat -> 0. Never exceeds full Kelly; never negative.
    lam_base=0.5 (half-Kelly) default; pass 0.25 (quarter-Kelly) for low-confidence signals
    (matches the vol_research conditional-quarter-Kelly decision).
    """
    if var_pnl <= 0 or n_trades < 2:
        return 0.0
    f_full = mean_pnl / var_pnl
    se = np.sqrt(var_pnl / n_trades)
    t = mean_pnl / se if se > 0 else 0.0
    shrink = max(0.0, 1.0 - 1.0 / (t * t)) if t != 0 else 0.0
    f_used = lam_base * shrink * f_full
    return float(max(0.0, min(f_used, f_full)))  # floor 0 (no edge), cap full Kelly


def kelly_lots(
    mean_pnl: float,
    var_pnl: float,
    n_trades: int,
    risk_capital: float,
    margin_per_lot: float,
    lam_base: float = 0.5,
) -> int:
    """Position size in whole lots from the fractional-Kelly fraction (rounded DOWN, >=0)."""
    if margin_per_lot <= 0 or risk_capital <= 0:
        return 0
    f = fractional_kelly(mean_pnl, var_pnl, n_trades, lam_base)
    return int(max(0, np.floor(f * risk_capital / margin_per_lot)))


# --------------------------------------------------------------- model-free risk floor
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
    """Intraday greeks limits -> action flags (long-straddle envelope).

    vega floor: edge structurally gone once vega < frac*entry (decayed out of the long-vol regime);
    theta envelope: theta bleed exceeding marginal gamma rent is the drive-decay exit;
    delta band: |net_delta| over delta_max -> trim to keep the Carr-Madan edge mapping valid.
    """
    return {
        "vega_dead": vega_now < vega_floor_frac * vega_entry,
        "theta_dominates": theta_bleed > gamma_rent,
        "delta_breach": abs(net_delta) > delta_max,
    }


def exit_decision(
    x_edge: float, b_star: float | None, ll: float, marginal_carry: float, minutes_to_squareoff: int
) -> dict:
    """Combine the exit triggers (precedence: square-off > stop > take-profit > drive-decay).

    x_edge          : current mean-reverting edge state (RV-IV spread / position mark).
    b_star          : take-profit threshold (take_profit_stoploss); None => no profitable hold.
    ll              : stop level on the edge (from the kill-switch mapping).
    marginal_carry  : G_t = gamma rent - theta - amortized cost (drive-decay / thesis check).
    """
    if minutes_to_squareoff <= 0:
        return {"exit": True, "reason": "same-day square-off (terminal condition)"}
    if x_edge <= ll:
        return {"exit": True, "reason": f"stop on edge: x {x_edge:.4f} <= L {ll:.4f}"}
    if b_star is not None and x_edge >= b_star:
        return {"exit": True, "reason": f"take-profit: x {x_edge:.4f} >= b* {b_star:.4f}"}
    if marginal_carry <= 0.0:
        return {"exit": True, "reason": "drive-decay: gamma rent no longer covers theta+cost"}
    return {"exit": False, "reason": "hold: edge alive, above stop, below take-profit"}


# ---------------- parabolic-cylinder representation of F,G (robust as r/mu -> 0) ----------------
# F,G also solve L u = r u via parabolic-cylinder (Weber) functions D_v. With
#   zeta = (x-theta)*sqrt(2 mu)/sigma,  nu = r/mu:
#   F ∝ e^{zeta^2/4} D_{-nu}(-zeta)  (increasing) ,  G ∝ e^{zeta^2/4} D_{-nu}(zeta)  (decreasing).
# The constant Gamma(nu) prefactor of the integral form (3.3)/(3.4) is dropped: the threshold
# equations (4.3)/(5.5) pair one F-type with one G-type in every term, so they are invariant to it.
# Well-conditioned as r/mu -> 0 (D_0 = e^{-z^2/4}), where the integral's u^{r/mu-1} singularity stiffens.
# scipy.special.pbdv(v, x) -> (D_v(x), D_v'(x)), so derivatives are exact.


def _pc(x, p: OUParams, r: float):
    kappa = np.sqrt(2.0 * p.mu) / p.sigma
    zeta = (x - p.theta) * kappa
    nu = r / p.mu
    dm, dpm = pbdv(-nu, -zeta)  # D_{-nu}(-zeta), D'_{-nu}(-zeta)
    dpz, dppz = pbdv(-nu, zeta)  # D_{-nu}( zeta), D'_{-nu}( zeta)
    e = np.exp(0.25 * zeta * zeta)
    return zeta, kappa, e, dm, dpm, dpz, dppz


def F_pc(x: float, p: OUParams, r: float) -> float:
    """Increasing fundamental solution via parabolic-cylinder D (robust for small r/mu)."""
    zeta, kappa, e, dm, dpm, dpz, dppz = _pc(x, p, r)
    return float(e * dm)


def Fp_pc(x: float, p: OUParams, r: float) -> float:
    """F'(x) via D: e^{zeta^2/4} kappa (zeta/2 D_{-nu}(-zeta) - D'_{-nu}(-zeta))."""
    zeta, kappa, e, dm, dpm, dpz, dppz = _pc(x, p, r)
    return float(e * kappa * (0.5 * zeta * dm - dpm))


def G_pc(x: float, p: OUParams, r: float) -> float:
    """Decreasing fundamental solution via parabolic-cylinder D."""
    zeta, kappa, e, dm, dpm, dpz, dppz = _pc(x, p, r)
    return float(e * dpz)


def Gp_pc(x: float, p: OUParams, r: float) -> float:
    """G'(x) via D: e^{zeta^2/4} kappa (zeta/2 D_{-nu}(zeta) + D'_{-nu}(zeta))."""
    zeta, kappa, e, dm, dpm, dpz, dppz = _pc(x, p, r)
    return float(e * kappa * (0.5 * zeta * dpz + dppz))


def _funcs(method: str):
    """Return (F, F', G, G') for the chosen backend: 'pc' (parabolic-cylinder) or 'quad' (integral)."""
    if method == "pc":
        return F_pc, Fp_pc, G_pc, Gp_pc
    if method == "quad":
        return F, Fp, G, Gp
    raise ValueError(f"unknown method {method!r} (use 'pc' or 'quad')")


def _scan_hi(p: OUParams, c: float, z_max: float = 10.0) -> float:
    """Upper bracket for the b* root-scan, bounded in STANDARDIZED units (zeta <= z_max) so the
    parabolic-cylinder evaluation stays conditioned even for fast mean reversion (large mu)."""
    return p.theta + z_max * p.sigma / np.sqrt(2.0 * p.mu) + abs(c)
