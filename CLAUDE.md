# arbitrage — vol-arb engine (development rules)

This repo implements the long-volatility / **realized-vs-implied** strategy from `vol_research`:
an unhedged intraday NIFTY straddle / slight-OTM strangle, entered on a jump + HAR-RV signal **only
when forecast realized vol exceeds the option's implied vol by enough to clear costs.**

## CORE PRINCIPLE (non-negotiable): QUALITY open-source first, minimal custom code
**Trading-infra bugs cost real money.** We **reuse battle-tested open-source** on every critical path and
write as little of our own code as possible — but the bar is **quality, not just "open-source":** only
mature, actively-maintained, widely-adopted, well-tested, permissively-licensed libraries qualify.
**An abandoned or niche library is a liability — for those, write our own (well-tested) instead.**
Writing our own **adapter / glue / signal logic is expected** where no quality library fits (e.g. a Kite adapter).

### Reuse — adopt these vetted libraries (do NOT reimplement):
| Concern | Library | Why it clears the quality bar |
|---|---|---|
| Engine (backtest **+** live, one code path) | **NautilusTrader** (pin a v1.x) | options-native (`OptionContract`) + true backtest↔live parity; 24k★, daily commits. *No NSE adapter — we write our own **Breeze** adapter (pure Python).* |
| Pricing / IV / greeks | **vollib** (`vollib.black`, Black-76) | MIT, LetsBeRational solver. **NOT** mibian (abandoned, GPL) or py_vollib_vectorized (5yr unmaintained, ITM bug). |
| HAR-RV | **statsmodels** (OLS + HAC SEs) | HAR is OLS w/ Newey-West. **No GARCH / `arch`** — HAR is the decided forecaster (per vol_research). |
| Broker + data (NSE) | **breeze-connect** (ICICI Breeze) | official, MIT; free 1-sec underlying + best free per-strike options. *Caveat: WS/session reliability — the adapter must be reconnect-robust and handle the daily session token.* |
| Dataframes / IO / storage | pandas / polars / pyarrow / duckdb | standard |

**Avoid (failed the quality/fit bar):** GARCH/`arch` (not needed — HAR decided), mibian, py_vollib_vectorized, vectorbt/pro (array paradigm fights options; non-permissive license), OpenAlgo as a default layer (extra AGPL hop for a single broker — call breeze-connect directly), lumibot, backtrader.

### Write ourselves — ONLY these (no quality lib covers them):
- **Breeze adapter** for NautilusTrader (data + execution) — pure Python; daily session-token auth; robust WS reconnect.
- **Signal logic**: HAR-RV RV forecast + Lee-Mykland jump flag (on NIFTY **futures**, 1-min) + the **RV−IV spread** model (*the edge axis*).
- **Strategy class** (Nautilus Strategy subclass) + the **spread-gated entry** (enter only when forecast RV beats IV + the regime VRP + costs).
- **NSE cost model** (full STT / txn / SEBI / GST / stamp / brokerage / bid-ask stack) — **build FIRST** (per IMPLEMENTATION.md §7).
- **Config** + a thin vollib wrapper.

## Rule for Claude (enforce on every change)
1. **Before writing any non-trivial component, search for a QUALITY library first** — check last release, commit activity, adoption, test coverage, license. Adopt only if it clears that bar; an abandoned/niche lib is worse than our own well-tested code.
2. **If you write custom code, justify in the PR/commit why no library fits.** Default is compose/configure, not build.
3. Keep the **custom surface area minimal and well-tested** (pytest; e.g. put-call-parity checks on pricing, golden-file tests on the cost model).
4. Pin dependency versions (uv); prefer libraries that are actively maintained and permissively licensed.

*(Stack finalized via a quality-graded research pass; see `PLAN.md` for the build order and the realized-vs-implied validation gate.)*
