from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import gumbel_r

from volarb.signals.lee_mykland import (
    BIPOWER_C,
    default_window,
    lee_mykland,
    local_bipower_sigma,
)


def _calm(n, seed=0, sigma=1e-3):
    return np.random.default_rng(seed).normal(0.0, sigma, size=n)


# --- INPUT VALIDATION (failure modes none of the existing tests exercise) ---


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5, 2.0])
def test_alpha_out_of_unit_interval_raises(bad):
    with pytest.raises(ValueError):
        lee_mykland(_calm(100), alpha=bad)


@pytest.mark.parametrize("bad_k", [2, 1, 0, -5])
def test_window_below_three_raises(bad_k):
    with pytest.raises(ValueError):
        local_bipower_sigma(_calm(100), bad_k)


@pytest.mark.parametrize("bad_n", [0, -1, -375])
def test_default_window_nonpositive_raises(bad_n):
    with pytest.raises(ValueError):
        default_window(bad_n)


# --- LEADING-EDGE STRUCTURE: sigma_hat is NaN until K returns of history exist ---


def test_leading_edge_is_nan_exactly_to_K():
    K = 27
    sig = local_bipower_sigma(_calm(120, sigma=2e-3), K)
    assert np.all(np.isnan(sig[:K]))  # no sigma before K returns of history
    assert np.isfinite(sig[K])  # first finite sigma at index K
    assert np.all(np.isfinite(sig[K:]))  # finite thereafter (clean diffusion)


def test_L_is_nan_on_leading_edge_and_never_flagged_there():
    K = 27
    out = lee_mykland(_calm(120, seed=3, sigma=1e-3), K=K, alpha=0.01)
    assert np.all(np.isnan(out["L"][:K]))
    assert not out["jump_flag"][:K].any()


# --- EXACT sigma_hat FORMULA on a hand-computable series (pins the (1/(K-2)) bipower sum) ---


def test_sigma_hat_matches_closed_form_on_constant_abs_returns():
    # |r| == a everywhere -> every bipower product == a^2; window has K-1 of them.
    # sigma_hat^2 = (K-1) * a^2 / (K-2).
    a = 1e-3
    K = 5
    r = np.full(20, a)  # constant positive returns -> |r|=a
    sig = local_bipower_sigma(r, K)
    expected = np.sqrt((K - 1) * a * a / (K - 2))
    assert sig[K] == pytest.approx(expected, rel=1e-12)
    assert sig[-1] == pytest.approx(expected, rel=1e-12)


# --- BIPOWER ROBUSTNESS: a single jump inside the window must not blow up local sigma ---


def test_single_jump_barely_perturbs_local_sigma():
    # Bipower pairs adjacent |r|, so one jump contaminates only TWO products. The trailing
    # sigma a few bars later (jump already out of window) is essentially unchanged.
    K = 27
    r = _calm(200, seed=9, sigma=1e-3)
    clean = local_bipower_sigma(r, K)
    r2 = r.copy()
    r2[100] = 0.05  # ~50-sigma jump
    dirty = local_bipower_sigma(r2, K)
    # window ending well before the jump is identical
    assert np.isclose(clean[90], dirty[90], equal_nan=True)
    # window starting after the jump has fully scrolled past it is identical again
    assert np.isclose(clean[100 + K + 2], dirty[100 + K + 2], equal_nan=True)


# --- DEGENERATE / EMPTY-WINDOW INPUTS (NaN gaps, all-zero, too-short) ---


def test_all_zero_returns_produce_no_flags_and_no_crash():
    out = lee_mykland(np.zeros(150), alpha=0.01)
    assert not out["jump_flag"].any()  # 0/0 -> NaN L, never flagged
    assert out["n"] == 0  # no finite L statistics
    assert out["threshold"] == np.inf  # nothing standardisable


def test_nan_gap_propagates_locally_then_recovers():
    # An overnight/roll gap encoded as NaN must NOT enter sigma_hat as a number; it makes the
    # windows that touch it NaN, but later windows (gap scrolled out) recover.
    K = 5
    r = _calm(60, seed=4, sigma=1e-3)
    r[30] = np.nan
    sig = local_bipower_sigma(r, K)
    assert np.isfinite(sig[30 + K + 2])
    out = lee_mykland(r, K=K, alpha=0.01)
    assert out["n"] >= 1


def test_too_short_series_returns_empty_no_flag_contract():
    out = lee_mykland(np.array([1e-3, -2e-3]), K=3, alpha=0.01)
    assert out["n"] == 0
    assert out["threshold"] == np.inf
    assert np.isnan(out["C_n"]) and np.isnan(out["S_n"]) and np.isnan(out["beta_star"])
    assert out["jump_flag"].shape == (2,) and not out["jump_flag"].any()


# --- GUMBEL CONSTANTS: exact identities + alpha monotonicity of the threshold ---


def test_gumbel_constants_match_paper_closed_form():
    out = lee_mykland(_calm(400, seed=5, sigma=1e-3), K=27, alpha=0.01)
    n = out["n"]
    c = BIPOWER_C
    log_n = np.log(n)
    sqrt_2logn = np.sqrt(2.0 * log_n)
    C_n = sqrt_2logn / c - (np.log(np.pi) + np.log(log_n)) / (2.0 * c * sqrt_2logn)
    S_n = 1.0 / (c * sqrt_2logn)
    beta = -np.log(-np.log(1.0 - 0.01))
    assert out["C_n"] == pytest.approx(C_n, rel=1e-12)
    assert out["S_n"] == pytest.approx(S_n, rel=1e-12)
    assert out["beta_star"] == pytest.approx(beta, rel=1e-12)
    assert out["beta_star"] == pytest.approx(gumbel_r.ppf(0.99), rel=1e-12)
    assert out["threshold"] == pytest.approx(C_n + S_n * beta, rel=1e-12)


def test_bipower_c_is_sqrt_two_over_pi():
    assert BIPOWER_C == pytest.approx(np.sqrt(2.0 / np.pi), rel=1e-15)


def test_threshold_strictly_decreasing_in_alpha():
    r = _calm(400, seed=6, sigma=1e-3)
    ths = [lee_mykland(r, K=27, alpha=a)["threshold"] for a in (0.001, 0.01, 0.05, 0.2)]
    assert all(hi > lo for hi, lo in zip(ths, ths[1:]))  # looser alpha -> lower bar


def test_smaller_alpha_never_flags_more_bars():
    # Monotone power: a stricter level cannot flag a bar a looser level missed.
    r = _calm(400, seed=8, sigma=1e-3)
    r[123] = 6e-3
    r[300] = 4.5e-3
    strict = lee_mykland(r, K=27, alpha=0.001)["jump_flag"]
    loose = lee_mykland(r, K=27, alpha=0.10)["jump_flag"]
    assert np.all(loose[strict])  # every strict flag is also a loose flag


# --- c^2-FIX DISCRIMINATOR made explicit: the buggy |L| would be c*L and would MISS ---


def test_c2_bug_would_miss_the_borderline_jump():
    r = _calm(400, seed=2, sigma=1e-3)
    r[300] = 5.5e-3
    out = lee_mykland(r, alpha=0.01)
    L = abs(out["L"][300])
    thr = out["threshold"]
    assert L > thr  # paper-faithful estimator fires
    assert BIPOWER_C * L < thr  # the /c^2 estimator (|L| scaled by c) would miss
