"""Lee-Mykland (2008) jump test on 1-min log-returns -- a REGIME feature, not a trigger.

The Lee-Mykland statistic asks whether the return at a single instant is too large to be
explained by the *local* diffusive volatility just before it. The local volatility is
estimated from bipower variation over a trailing window (bipower variation is robust to
jumps -- it pairs adjacent absolute returns, so a single jump cannot inflate it), giving

    sigma_hat_t^2 = (1 / (K - 2)) * sum_{i} |r_{i-1}| * |r_i|     over a trailing window K

    L_t = r_t / sigma_hat_t                                       (test statistic)

Under the null of no jump, the maximum of |L_t| over n observations follows a Gumbel law.
A jump is flagged when the standardised excess exceeds the Gumbel quantile for level alpha:

    (|L_t| - C_n) / S_n  >  beta*       with   beta* = -log(-log(1 - alpha))

using the Lee-Mykland constants (c = sqrt(2/pi)):

    C_n = (2 log n)^{1/2} / c  -  (log(pi) + log(log n)) / (2 c (2 log n)^{1/2})
    S_n = 1 / (c (2 log n)^{1/2})

Window-length rule. Lee-Mykland recommend K ~ a multiple of sqrt(observations-per-day) so
that K -> infinity while the window's time span -> 0 (local-volatility consistency without
leaking the jump into sigma_hat). For a 375-row NSE session that is K ~ sqrt(375) ~ 19;
we default to a slightly larger K=27 (~ smoothing over ~half an hour) for a steadier local
sigma. `n` in the Gumbel constants is the number of valid L-statistics being maxed over.

Reference: S. S. Lee & P. A. Mykland (2008), "Jumps in Financial Markets: A New
Nonparametric Test and Jump Dynamics", Review of Financial Studies 21(6).

Per the strategy spec this is used as a regime / feature on NIFTY-futures 1-min returns,
NEVER as a reactive trigger. Pure numpy + scipy for the Gumbel quantile; no data lake.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import gumbel_r

__all__ = [
    "BIPOWER_C",
    "local_bipower_sigma",
    "lee_mykland",
    "default_window",
]

# c = E[|Z|] for a standard normal = sqrt(2/pi). It appears ONLY in the Gumbel constants
# C_n/S_n -- NOT in sigma_hat. Lee-Mykland's sigma_hat^2 = (1/(K-2)) sum |r||r| deliberately
# keeps the c^2 bias, and C_n/S_n are derived for exactly that biased estimator; dividing
# sigma_hat by c^2 would mis-calibrate the threshold (~165x too conservative).
BIPOWER_C: float = np.sqrt(2.0 / np.pi)


def default_window(n_obs: int, multiplier: float = 1.4) -> int:
    """Trailing window K via the Lee-Mykland sqrt-of-observations rule.

    K ~ multiplier * sqrt(n_obs) (then clamped to >= 3 so the bipower sum is defined).
    For a 375-row NSE session this gives ~27. Exposed so callers can size the window to
    their bar count rather than hard-coding it.
    """
    if n_obs <= 0:
        raise ValueError(f"n_obs must be positive, got {n_obs}")
    k = int(round(multiplier * np.sqrt(n_obs)))
    return max(k, 3)


def local_bipower_sigma(log_returns: np.ndarray, K: int) -> np.ndarray:
    """Rolling local volatility (sigma_hat_t) from trailing bipower variation.

    sigma_hat_t^2 = (1/(K-2)) * sum of adjacent |r| products over the K returns ending *just
    before* t (the Lee-Mykland estimator -- the c^2 bias is left in; C_n/S_n correct for it).
    The window ends at t-1 (it excludes r_t itself) so the statistic L_t = r_t / sigma_hat_t
    does not standardise the candidate return by a sigma that already contains it -- this is
    what gives the test power against a jump at t. Bipower variation pairs adjacent returns,
    so one large jump inside the window contaminates only two products and cannot blow up the
    local-vol estimate.

    Args:
        log_returns: 1-D array of 1-min log-returns within a single session. NaNs (e.g. an
            overnight/roll gap that must never enter) propagate as NaN sigma locally.
        K: trailing window length in returns (>= 3); see :func:`default_window`.

    Returns:
        sigma_hat: float array, same length as ``log_returns``. Entries where fewer than
        ``K`` prior returns exist (the leading edge) are NaN by construction (no-lookahead:
        the window is strictly the K returns *before* each index).
    """
    r = np.asarray(log_returns, dtype=float).ravel()
    n = r.size
    if K < 3:
        raise ValueError(f"K must be >= 3 for a bipower window, got {K}")

    # Adjacent absolute-return products: bp[i] = |r[i-1]| * |r[i]|, indexed at i.
    abs_r = np.abs(r)
    bp = np.full(n, np.nan)
    bp[1:] = abs_r[:-1] * abs_r[1:]

    sigma = np.full(n, np.nan)
    # sigma_hat at t uses the K returns ending at t-1 -> the K-1 bipower products bp over
    # (t-K+1 .. t-1). We need at least K returns of history strictly before t.
    for t in range(K, n):
        window_bp = bp[t - K + 1 : t]  # K-1 products, all strictly before t
        # Lee-Mykland sigma_hat^2 = sum |r||r| / (K-2). The c=sqrt(2/pi) bias is left IN;
        # C_n/S_n correct for it. Do NOT divide by c^2 here (that mis-calibrates the test).
        finite = window_bp[np.isfinite(window_bp)]
        if finite.size == 0:
            continue
        sigma[t] = np.sqrt(finite.sum() / (K - 2))
    return sigma


def lee_mykland(
    log_returns: np.ndarray,
    K: int | None = None,
    alpha: float = 0.01,
) -> dict[str, np.ndarray | float]:
    """Lee-Mykland jump test over a series of 1-min log-returns.

    Computes the local-volatility-standardised statistic L_t and flags jumps where the
    Gumbel-standardised excess exceeds the level-``alpha`` critical value. This is a
    regime/feature output (which bars carry jumps, how big), NOT a reactive trade trigger.

    Args:
        log_returns: 1-D 1-min log-returns within a single session.
        K: trailing bipower window (>= 3). Defaults to :func:`default_window` of the series
            length (the sqrt-of-observations rule).
        alpha: test level (significance). Smaller alpha -> stricter threshold, fewer flags.

    Returns:
        dict with:
          ``L``        : float array of L_t = r_t / sigma_hat_t (NaN on the leading edge).
          ``sigma_hat``: float array of local volatility (from :func:`local_bipower_sigma`).
          ``jump_flag``: bool array, True where bar t is flagged a jump.
          ``threshold``: float, the absolute-L critical value C_n + S_n * beta* (the
                         |L_t| level above which a jump fires; handy for diagnostics).
          ``C_n``, ``S_n``, ``beta_star``: the Gumbel constants used.
          ``n``        : int, number of finite L statistics the Gumbel law is taken over.
    """
    r = np.asarray(log_returns, dtype=float).ravel()
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if K is None:
        K = default_window(r.size)

    sigma = local_bipower_sigma(r, K)
    with np.errstate(divide="ignore", invalid="ignore"):
        L = r / sigma  # NaN where sigma is NaN (leading edge) or 0

    finite = np.isfinite(L)
    n = int(finite.sum())
    if n < 1:
        # Not enough history to standardise anything; nothing can be flagged.
        return {
            "L": L,
            "sigma_hat": sigma,
            "jump_flag": np.zeros(r.size, dtype=bool),
            "threshold": np.inf,
            "C_n": np.nan,
            "S_n": np.nan,
            "beta_star": np.nan,
            "n": n,
        }

    # Lee-Mykland Gumbel constants. n is the count of L statistics being maxed over.
    log_n = np.log(n)
    sqrt_2logn = np.sqrt(2.0 * log_n)
    c = BIPOWER_C
    C_n = sqrt_2logn / c - (np.log(np.pi) + np.log(log_n)) / (2.0 * c * sqrt_2logn)
    S_n = 1.0 / (c * sqrt_2logn)

    # beta* is the upper-(alpha) quantile of the standard Gumbel: P(xi > beta*) = alpha.
    # gumbel_r.ppf(1 - alpha) == -log(-log(1 - alpha)).
    beta_star = float(gumbel_r.ppf(1.0 - alpha))

    threshold = C_n + S_n * beta_star  # the |L_t| level at which a jump is declared

    jump_flag = np.zeros(r.size, dtype=bool)
    excess = np.full(r.size, np.nan)
    excess[finite] = (np.abs(L[finite]) - C_n) / S_n
    jump_flag[finite] = excess[finite] > beta_star

    return {
        "L": L,
        "sigma_hat": sigma,
        "jump_flag": jump_flag,
        "threshold": float(threshold),
        "C_n": float(C_n),
        "S_n": float(S_n),
        "beta_star": beta_star,
        "n": n,
    }
