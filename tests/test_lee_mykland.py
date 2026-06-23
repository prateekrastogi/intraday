"""Synthetic tests for the Lee-Mykland jump test.

No real lake/creds/network. Asserts the MATH: the local bipower sigma tracks volatility, a
planted jump is flagged (and a calm stretch is not), a ~5.5-sigma jump fires (which the old
/c^2 normalization bug would MISS), and the per-series false-positive rate under a pure
Gaussian diffusion is on the order of alpha (the bug made it ~165x too conservative).
"""

from __future__ import annotations

import numpy as np

from volarb.signals.lee_mykland import default_window, lee_mykland, local_bipower_sigma


def _calm(n: int, seed: int = 0, sigma: float = 1e-3) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, sigma, size=n)


def test_default_window_sqrt_rule():
    assert default_window(375) == max(3, round(1.4 * np.sqrt(375)))
    assert default_window(1) >= 3


def test_local_sigma_tracks_volatility():
    sig = local_bipower_sigma(_calm(400, sigma=2e-3), K=27)
    finite = sig[np.isfinite(sig)]
    assert finite.size > 0
    assert 5e-4 < np.median(finite) < 6e-3  # near the true 2e-3, within a loose band


def test_planted_jump_is_flagged_and_calm_is_not():
    r = _calm(400, seed=1, sigma=1e-3)
    r[250] = 0.02  # a ~20-sigma return: unambiguous jump
    out = lee_mykland(r, alpha=0.01)
    assert bool(out["jump_flag"][250]) is True
    assert not out["jump_flag"][50:200].any()  # no flags in a clean stretch away from the jump


def test_five_sigma_jump_fires_only_after_normalization_fix():
    # ~5.5-sigma jump: with the paper-faithful sigma_hat, |L| ~ 6.9 > threshold ~6.0 (alpha=0.01)
    # so it fires. The old /c^2 bug inflated sigma_hat by 1/c, dropping |L| to ~5.5 < 6.0 and
    # MISSING this jump -- so this test guards the normalization fix.
    r = _calm(400, seed=2, sigma=1e-3)
    r[300] = 5.5e-3
    assert bool(lee_mykland(r, alpha=0.01)["jump_flag"][300]) is True


def test_null_per_series_flag_rate_is_order_alpha():
    # Pure Gaussian diffusion (no jumps): the Gumbel threshold targets the MAX over n bars, so
    # the PER-SERIES flag probability is on the order of alpha. The old /c^2 bug was ~165x too
    # conservative -> almost no series ever flag. Per-series rate (not per-bar) is the right metric.
    alpha = 0.05
    n_series = 150
    flagged = sum(
        bool(lee_mykland(_calm(400, seed=200 + s, sigma=1e-3), alpha=alpha)["jump_flag"].any())
        for s in range(n_series)
    )
    rate = flagged / n_series
    assert 0.02 < rate < 0.5  # order alpha; the c^2 bug gives ~0 here (fails low)
