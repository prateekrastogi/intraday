"""Backtest runner for the directional ITM ORB strategy on NIFTY-weekly + Crude.

Runs each instrument full-sample and on a held-out train/test split (TEST = the last ~year, never used to
choose parameters). Prints n / win% / EV-per-trade / per-trade Sharpe / final equity (20%/trade) / max drawdown
/ top-3-trade share of PnL (the fat-tail tell) / real-mark coverage. No-lookahead; real 1-min data only.

    uv run --no-sync python -m volarb.run.backtest            # both instruments
    uv run --no-sync python -m volarb.run.backtest NIFTY-weekly
"""

from __future__ import annotations

import datetime as dt
import sys

from volarb.strategy import orb
from volarb.strategy.config import INSTRUMENTS

SPLIT = dt.date(2025, 6, 1)  # held-out test boundary (the last ~year is never used to tune)


def _line(tag, m, cov=None):
    if not m:
        print(f"  {tag:14s} (no trades)")
        return
    c = f" cov {cov:3.0f}%" if cov is not None else ""
    print(f"  {tag:14s} n={m['n']:4d} win={m['win']:2.0f}% EV={m['ev']:+5.1f}%/tr "
          f"Sharpe={m['sharpe']:+.2f} eq={m['final_eq']:.2f}x DD={m['max_dd']:+.0f}%{c}")
    # winner-concentration: how much of the edge is a few tail winners
    print(f"  {'':14s} top1/top3 of PnL={m['top1_pnl']:.0f}%/{m['top3_pnl']:.0f}%  "
          f"EV net of top1={m['ev_drop1']:+.1f}%/tr, net of top3={m['ev_drop3']:+.1f}%/tr")


def backtest(cfg):
    B = orb.build(cfg)
    trades, cov = orb.run(cfg, B)
    nd = len(B["days"])
    ndtr = sum(1 for d in B["days"] if d < SPLIT)
    ndte = sum(1 for d in B["days"] if d >= SPLIT)
    tr = [t for t in trades if t[0] < SPLIT]
    te = [t for t in trades if t[0] >= SPLIT]
    print(f"\n### {cfg.name}  ({B['days'][0]}..{B['days'][-1]}, {nd} days, hold={cfg.hold}, stop={int(cfg.stop*100)}%)")
    _line("full", orb.metrics(trades, nd), cov)
    _line("train(<split)", orb.metrics(tr, ndtr))
    _line("TEST(>=split)", orb.metrics(te, ndte))


def main():
    names = sys.argv[1:] or list(INSTRUMENTS)
    print("Directional ITM opening-range-breakout | real 1-min data | true 1-min -12% stop | net spread")
    for nm in names:
        if nm not in INSTRUMENTS:
            print(f"unknown instrument {nm!r}; choices: {list(INSTRUMENTS)}")
            continue
        backtest(INSTRUMENTS[nm])
    print("\nLow win rate (~15-30%) is by design: cut losers small, ride rare big winners (convex/fat-tailed).")


if __name__ == "__main__":
    main()
