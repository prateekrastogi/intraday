# arbitrage — directional ITM ORB engine (development rules)

This repo implements the one **validated** edge from 3 years of real 1-minute OOS backtesting (see `README.md`):
an intraday **opening-range breakout** → buy a deep-ish **ITM (~delta 0.65) option** in the breakout direction →
**cut losses fast at a true 1-minute −12% intraday stop, ride winners ~5 trading days.** Convex / fat-tailed
(low win rate, rare big winners). The cut-loss-fast stop *is* the edge. Runs on NIFTY-weekly and MCX Crude.

> The original long-vol straddle/strangle (realized-vs-implied spread) thesis was **refuted** by the same
> backtesting and removed. Don't reintroduce vol-forecast signals (HAR-RV, Lee-Mykland), straddle/strangle/
> backspread structures, or the OU exit — they were tested and bled.

## CORE PRINCIPLE (non-negotiable): QUALITY open-source first, minimal custom code
Trading-infra bugs cost real money. **Reuse battle-tested, actively-maintained, permissively-licensed libraries**
on every critical path; write only what no quality library covers — here: the Breeze data adapter, the ORB
signal, the strike/stop strategy logic, and the NSE cost model. An abandoned/niche library is a liability —
write our own (well-tested) instead.

### Reuse — adopt, don't reimplement
| Concern | Library |
|---|---|
| Dataframes / IO / Parquet | pandas / numpy / pyarrow |
| Broker + data (NSE + MCX) | **breeze-connect** (ICICI Breeze) — official, MIT; 1-min underlying + per-strike options incl. expired; MCX via `exchange_code="MCX"`. |

The strategy is a lightweight pandas/numpy backtest (`strategy/orb.py`) on the real 1-min Parquet lakes — it uses
**moneyness, not a pricing model** (no greeks needed). A NautilusTrader engine path is NOT required for this
research-grade implementation; promote to Nautilus only if/when going live (then keep one strategy class
backtest↔live).

### Write ourselves — ONLY these
- **Breeze adapter** (`adapters/breeze/`): daily session-token auth, rate limiter, resumable 1-min backfill.
- **ORB signal + strategy** (`strategy/orb.py`, `strategy/config.py`): opening-range breakout (OR scaled to ~8%
  of session), ITM strike selection, true 1-minute intraday stop, N-day hold capped at expiry.
- **NSE cost model** (`costs/nse_fees.py`): full STT / txn / SEBI / GST / stamp / brokerage / bid-ask stack.

## Rules for Claude (enforce on every change)
1. Before writing any non-trivial component, search for a QUALITY library first; adopt only if it clears the bar.
2. Keep the custom surface area **minimal and well-tested** (pytest). No-lookahead by construction; net-of-cost.
3. **Match comparisons** (same hold period when comparing moneyness; same instrument at equal expiry, etc.) and
   **report winner-robustness** (drop-top-N, OOS held-out test) — the edge is fat-tailed and easy to fool yourself on.
4. Pin dependency versions (uv). `uv run --no-sync` to run. **Never commit secrets** — `BREEZE_SESSION_TOKEN`
   lives only in `.env` (gitignored); commit/push only when explicitly asked.
