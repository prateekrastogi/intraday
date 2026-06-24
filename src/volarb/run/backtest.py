"""Walk-forward / out-of-sample data splitting for the step-5 backtest (validation discipline, IMPLEMENTATION.md s6).

Encodes the train-vs-test methodology for the thin, regime-dependent RV-IV edge:

- WALK-FORWARD: fit parameters on PAST days only, trade the next unseen day(s), roll forward. Mimics
  live deployment (every decision uses only history) and tests robustness ACROSS REGIMES, which a single
  static train/test split cannot. `expanding=True` grows the fit window each fold; `False` uses a fixed
  trailing window of `min_fit` days.
- EMBARGO: drop `embargo` days between each fit window and its test block to guard the boundary against
  label/feature overlap (Lopez de Prado). For our INTRADAY, same-day-square-off strategy each trade's
  label lives entirely within its own day, so embargo=0 is already leak-free; raise it only if multi-day
  features/labels are ever introduced. Exposed as a first-class knob regardless.
- FINAL HOLDOUT: split off the most recent `holdout_days` and NEVER use them during development /
  parameter tuning; test ONCE at the end -- the guard against multiple-testing overfit on a thin edge.

Pure date logic: no data, no Nautilus. No-lookahead is guaranteed by construction -- max(fold.fit) is
strictly earlier than min(fold.test) for every fold. The harness (run/backtest.py) consumes these folds:
for each Fold, fit signals/OU/VRP/Kelly on Fold.fit, trade Fold.test net-of-cost; run the holdout last.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold: fit on `fit`, (drop `embargoed`), trade `test`. fit/test are disjoint
    and max(fit) < min(test) -- no lookahead."""

    fit: tuple[date, ...]
    test: tuple[date, ...]
    embargoed: tuple[date, ...]


def walk_forward(
    days,
    *,
    min_fit: int,
    test_size: int = 1,
    embargo: int = 0,
    expanding: bool = True,
) -> list[Fold]:
    """Walk-forward folds over sorted-unique `days`.

    Fit on days[:k] (expanding) or the trailing `min_fit` days (rolling), drop `embargo` days, test the
    next `test_size` days, then roll k forward by `test_size`. Every eligible day appears in exactly one
    test block; the first `min_fit + embargo` days are fit-only (never tested). Returns [] if there are
    not enough days for a single fold.
    """
    if min_fit < 1 or test_size < 1 or embargo < 0:
        raise ValueError("require min_fit>=1, test_size>=1, embargo>=0")
    days = sorted(set(days))
    n = len(days)
    folds: list[Fold] = []
    k = min_fit
    while k + embargo + test_size <= n:
        fit_lo = 0 if expanding else k - min_fit
        fit = days[fit_lo:k]
        embargoed = days[k : k + embargo]
        test = days[k + embargo : k + embargo + test_size]
        folds.append(Fold(tuple(fit), tuple(test), tuple(embargoed)))
        k += test_size
    return folds


def final_holdout(days, *, holdout_days: int) -> tuple[list[date], list[date]]:
    """Split sorted-unique `days` into (dev, holdout); holdout = the last `holdout_days` (the untouched
    final test set), dev = everything before it. max(dev) < min(holdout)."""
    days = sorted(set(days))
    if not 1 <= holdout_days < len(days):
        raise ValueError(f"require 1 <= holdout_days < {len(days)}, got {holdout_days}")
    return days[:-holdout_days], days[-holdout_days:]


def walk_forward_with_holdout(
    days,
    *,
    min_fit: int,
    holdout_days: int,
    test_size: int = 1,
    embargo: int = 0,
    expanding: bool = True,
) -> tuple[list[Fold], list[date]]:
    """Carve the final holdout FIRST, then walk-forward only the development days. The returned folds
    never touch the holdout -- so the holdout stays a true, once-only out-of-sample test."""
    dev, holdout = final_holdout(days, holdout_days=holdout_days)
    folds = walk_forward(
        dev, min_fit=min_fit, test_size=test_size, embargo=embargo, expanding=expanding
    )
    return folds, holdout
