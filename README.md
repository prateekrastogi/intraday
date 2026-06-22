# volarb — intraday NIFTY straddle/strangle vol-arb engine

Implements the long-volatility / **realized-vs-implied SPREAD** strategy from `vol_research`
(see `vol_research/IMPLEMENTATION.md`): enter an unhedged intraday NIFTY ATM straddle / slight-OTM strangle
on a HAR-CJ + Lee-Mykland signal, **only when forecast realized vol beats the option's implied vol by enough
to clear the variance risk premium + the full NSE cost stack.**

**Read `CLAUDE.md` (dev rules) and `PLAN.md` (build order, stack, validation gate) first.**

## Principle: quality open-source first, minimal custom code
Reuse vetted libraries on every critical path; write only the Breeze adapter, the signal logic, the strategy
class, and the NSE cost model. See `CLAUDE.md`.

## Stack
NautilusTrader (engine + backtest↔live) · vollib (pricing/IV) · statsmodels (HAR) · breeze-connect (broker+data).

## Run (uv)
```
uv sync
uv run python -m volarb.run.backtest   # walk-forward OOS on the Parquet data catalog
uv run python -m volarb.run.live        # paper/live via Breeze adapter (SAME strategy class)
```
