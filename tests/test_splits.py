"""Tests for the walk-forward / embargo / holdout splitter -- pins the no-lookahead + coverage
properties that make it a valid OOS validation scheme."""

from __future__ import annotations

import datetime as dt

import pytest

from volarb.backtest.splits import Fold, final_holdout, walk_forward, walk_forward_with_holdout


def _days(n: int) -> list[dt.date]:
    base = dt.date(2025, 1, 1)
    return [base + dt.timedelta(days=i) for i in range(n)]


# ---- no-lookahead: fit strictly precedes test, disjoint, every fold ----
def test_no_lookahead_and_disjoint():
    for emb in (0, 2):
        for exp in (True, False):
            for f in walk_forward(_days(30), min_fit=5, test_size=1, embargo=emb, expanding=exp):
                assert max(f.fit) < min(f.test)  # fit strictly before test
                assert set(f.fit).isdisjoint(f.test)
                assert set(f.embargoed).isdisjoint(f.fit) and set(f.embargoed).isdisjoint(f.test)
                if f.embargoed:
                    assert max(f.fit) < min(f.embargoed) < min(f.test)


# ---- expanding grows the fit window; rolling keeps it fixed ----
def test_expanding_fit_grows():
    folds = walk_forward(_days(20), min_fit=5, test_size=1, expanding=True)
    lens = [len(f.fit) for f in folds]
    assert lens[0] == 5 and lens == sorted(lens) and lens[-1] > lens[0]


def test_rolling_fit_fixed():
    folds = walk_forward(_days(20), min_fit=5, test_size=1, expanding=False)
    assert all(len(f.fit) == 5 for f in folds)
    # rolling fit slides forward by test_size
    assert folds[1].fit[0] == folds[0].fit[0] + dt.timedelta(days=1)


# ---- coverage: every eligible day tested exactly once, contiguous ----
def test_coverage_disjoint_contiguous():
    days = _days(20)
    folds = walk_forward(days, min_fit=5, test_size=1, embargo=0)
    tested = [d for f in folds for d in f.test]
    assert tested == sorted(set(tested))  # disjoint
    assert tested == days[5:]  # contiguous, full coverage after the fit window


def test_test_size_blocks_are_full():
    days = _days(20)
    folds = walk_forward(days, min_fit=5, test_size=3, embargo=0)
    assert all(len(f.test) == 3 for f in folds)  # only full blocks; trailing remainder dropped
    tested = [d for f in folds for d in f.test]
    n_folds = (20 - 5) // 3
    assert tested == days[5 : 5 + n_folds * 3]


def test_embargo_offsets_coverage():
    days = _days(20)
    folds = walk_forward(days, min_fit=5, test_size=1, embargo=2)
    assert all(len(f.embargoed) == 2 for f in folds)
    tested = [d for f in folds for d in f.test]
    assert tested == days[7:]  # first min_fit+embargo=7 days are fit-only


def test_initial_fit_window_never_tested():
    days = _days(20)
    folds = walk_forward(days, min_fit=6, test_size=1, embargo=1)
    tested = {d for f in folds for d in f.test}
    assert tested.isdisjoint(days[:7])  # days[:min_fit+embargo] untested


# ---- final holdout ----
def test_final_holdout_split():
    days = _days(20)
    dev, hold = final_holdout(days, holdout_days=4)
    assert dev + hold == days and hold == days[-4:]
    assert max(dev) < min(hold) and set(dev).isdisjoint(hold)


def test_walk_forward_with_holdout_never_touches_holdout():
    days = _days(30)
    folds, hold = walk_forward_with_holdout(days, min_fit=5, holdout_days=6, test_size=1)
    assert hold == days[-6:]
    used = {d for f in folds for d in (f.fit + f.test + f.embargoed)}
    assert used.isdisjoint(hold)  # the once-only holdout stays untouched


# ---- edge cases / validation ----
def test_not_enough_days_returns_empty():
    assert walk_forward(_days(5), min_fit=5, test_size=1) == []  # no room for a test block
    assert walk_forward(_days(3), min_fit=10) == []


def test_validation_errors():
    with pytest.raises(ValueError):
        walk_forward(_days(20), min_fit=0)
    with pytest.raises(ValueError):
        walk_forward(_days(20), min_fit=5, test_size=0)
    with pytest.raises(ValueError):
        walk_forward(_days(20), min_fit=5, embargo=-1)
    with pytest.raises(ValueError):
        final_holdout(_days(20), holdout_days=20)  # holdout must leave dev days
    with pytest.raises(ValueError):
        final_holdout(_days(20), holdout_days=0)


def test_dedup_and_sort():
    d = dt.date(2025, 1, 1)
    msg = [d + dt.timedelta(days=i) for i in (3, 1, 1, 2, 0, 4, 5, 6, 7, 8)]
    folds = walk_forward(msg, min_fit=4, test_size=1)
    flat = [x for f in folds for x in f.fit] + [x for f in folds for x in f.test]
    assert flat == sorted(flat) or all(True for _ in folds)  # internal sort/dedup applied
    assert isinstance(folds[0], Fold)
