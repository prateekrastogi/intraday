# PLAN — volarb (intraday NIFTY straddle/strangle vol-arb)

*Implements the strategy from `vol_research`; **aligned to `vol_research/IMPLEMENTATION.md`**. Built as thin
glue over quality open-source (see `CLAUDE.md`).*

## North star — the reframe (IMPLEMENTATION.md §0)
Forecast accuracy ≠ edge. Option IV *is* the market's RV forecast, so a good RV model mostly reproduces
consensus. **The edge = the RV-forecast − IV *spread*, net of the variance risk premium and costs** — and it
only appears in narrow pockets (short-dated, index/liquid, intraday, where IV under-prices realized and you act
before the surface re-prices). **Build everything to target the SPREAD, not the forecast.**

> ⚠️ **Honest scope (§5):** the edge is *thin on NIFTY* (efficient surface + heavy STT/spread drag), strongest
> on *BTC daily*. The NIFTY OOS net-of-cost test (step 5) may not clear — that is the point of validating.

## Scope guardrails (§5 — what we do NOT trade)
Trade **magnitude, never direction**. **In-scope:** short-dated + NIFTY index + intraday + where IV under-prices. **Out-of-scope (do not build toward):** monthly-expiry intraday, single-stock options, directional bets, clustering already priced by IV (the equity weekly/30-day trap), and **leverage** (scales risk, adds no edge). The intraday edge is a **speed edge** — act before the short-dated IV re-prices.

## Quality stack (grades in CLAUDE.md)
- **Engine + backtest↔live:** NautilusTrader (pin v1.x) — options-native, *same strategy class* sim↔live.
- **Pricing / IV / greeks:** NautilusTrader built-in `imply_vol_and_greeks` (Rust LetsBeRational, Black-76 via b=0) -- one inverter across research+backtest+live.
- **HAR-RV:** statsmodels (OLS + HAC). **No GARCH.**
- **Broker + data:** breeze-connect (ICICI Breeze) — 1-sec underlying + per-strike options (incl. expired).
  *WS/session reliability is the known risk; the adapter must be reconnect-robust.*

## Write ourselves (no quality lib covers these)
1. **Breeze adapter for Nautilus** (data + execution) — pure Python; daily session token; robust WS reconnect. *Highest-effort item.*
2. **HAR-RV forecast** + **Lee-Mykland jump flag** (NIFTY **futures**, 1-min; 5-min RV components). Jump = regime/feature, **not a reactive trigger** (a detected jump has already re-priced the option — position *ahead* of jump-prone regimes).
3. **RV−IV spread model** — *the edge axis*; benchmark IV = India VIX / ATM IV; de-bias by regime VRP; trade only the residual that beats IV.
4. **Strategy** (Nautilus Strategy subclass): straddle / slight-OTM strangle entry/exit/sizing, gated on the spread.
5. **NSE cost model** — built FIRST.
6. Thin wrapper over Nautilus `imply_vol_and_greeks` + IV guards (vega floor, t-floor, intrinsic filter, ATM smile-interp).

*Optional (≤4–6 param ceiling, only if they beat OOS net-of-cost): HARQ (measurement-error), signed jumps + realized semivariance, term-structure / IV-percentile gating (Johnson 2017 SLOPE — inverted term structure / low IV-percentile). Skip rough-vol, ML/DL, exotic jump machinery.*

## NSE cost model (§4 — build FIRST, "or the backtest lies")
- STT **0.10% of premium on sell** (→ **0.15% Apr-2026**); **0.125% of intrinsic on ITM exercise → never let an ITM option expire; square off.**
- NSE txn ≈ 0.03553% premium; SEBI 0.0001%; **GST 18%** on (brokerage+txn+SEBI); stamp 0.003% buy; brokerage ~₹20/order.
- **Model the bid-ask spread explicitly** (tight NIFTY ATM, wide OTM) — dominates short-dated/frequent. Round-trip ≈ 0.8% premium ATM, 30%+ far-OTM → **avoid far-OTM scalping.** Golden-file tested.

## Build order (IMPLEMENTATION.md §7)
0. ~~Repo + uv deps + pin Nautilus~~ -- DONE (uv.lock committed; Oracle A1 bootstrapped, uv synced, imports OK).
1. ~~NSE cost model~~ -- **DONE** (date-aware calculator + 6 golden tests: costs/nse_fees.py, tests/test_costs.py; Nautilus FeeModel wrapper deferred to backtest integration).
2. ~~Breeze data adapter~~ -- **DONE & validated on live Breeze** (auth + history[IST/OHLCV/OI] + ratelimit + robust multi-instrument backfill: index + front-month futures + India VIX + 2 weekly expiries × ATM-band × {CE,PE}, expiry-day widening, discover-don't-compute expiries, retry/backoff, clean budget-stop, resumable manifest; 24 tests). Produces the raw long parquet lake. *Nautilus catalog writer (data/catalog.py) + research panel (data/panel.py) fold into steps 3-5.*
3. ~~HAR-RV forecast + Lee-Mykland~~ -- **IMPLEMENTED** (signals/{har_rv,lee_mykland,semivariance,quarticity}.py + data/panel.py + pricing/greeks.py via Nautilus Black-76; unit-tested on synthetic data, 122 tests). Offline FIT on real pulled data = step 5.
4. ~~RV−IV spread signal~~ -- **IMPLEMENTED** (signals/{spread,surface}.py + strategy/vol_gate.py; de-biased by regime VRP, Johnson SLOPE gate, cost-hurdle from nse_fees, RV/IV unit-reconciled). Walk-forward OOS validation = step 5.
4b. **Exit & risk-management model** -- **IMPLEMENTED** (strategy/risk.py: Leung-Li F/G + b* solvers [Eq 4.3/5.5], L* no-trade filter [Eq 4.1/Cor 5.2], OU MLE + ADF stationarity gate, fractional-Kelly, kill-switch/greeks-envelope/exit-decision; 16 tests verify vs the paper propositions -- L* exact, Prop 5.3 b*_L decreasing meeting L*, Cor 5.2 no-trade, Remark 4.7 BM degenerate. Exit runs on the mean-reverting RV-IV edge, not the price; no price stop on long gamma. OOS validation vs fixed-time + kill-switch baselines folds into step 5.) (the exit policy drives most of the return variance, so backtesting entries against a placeholder exit is meaningless -- this GATES step 5). Covers: optimal-stopping EXIT timing (gamma-gain vs theta-bleed race), profit-booking / take-profit under a mean-reverting edge, stop-loss design (caveat: a naive price-stop can HURT a long-gamma / positive-convexity position), fractional-Kelly sizing on a thin estimation-uncertain edge, intraday gamma/vega/theta limits, and drawdown / daily-loss kill-switch. Grounded in the verified literature reading list (deep-research).
5. **Walk-forward OOS backtest on REAL Breeze option data**, vs (a) blind straddle (b) opening-momentum baseline. **← THE GATE.** *(Validation split DONE -- backtest/splits.py: walk-forward + embargo + final-holdout, 12 tests. Harness wiring [Nautilus catalog + signals + risk + baselines] pending the full lake.)*
6. If it clears costs OOS → **paper (Breeze)** → small live on a **fixed-IP VPS, SEBI-compliant.**

## Validation discipline (§6)
Walk-forward, OOS, **net-of-cost**; judge on **Sortino/Calmar** (long-vol is positive-skew/lumpy), not Sharpe
alone; must beat **both** the blind straddle and the opening-momentum baseline OOS; current results are
**BS-modeled off real underlying paths** — the binding next step is **real Breeze option prices**;
multiple-testing awareness; add a feature only if it improves **OOS net-of-cost** Sharpe (never in-sample R²; must survive walk-forward *across regimes*).

## Compliance & risk (§3)
SEBI retail-algo (full enforcement Apr-2026): static-IP whitelist + 2FA + client keys; **<10 orders/sec exempt**;
**fixed-IP VPS**; SEBI-registered domestic broker (ICICI/Breeze); daily Breeze session token. **Risk:** size far
below Kelly; **hard daily-loss kill-switch** (mandatory, unhedged short-dated options); square-off timer.

## Status
Research-grade, **not a validated edge.** The make-or-break is **step 5** (real-option-data OOS net-of-cost on
NIFTY); everything before is plumbing. *(NautilusTrader also ships crypto adapters → BTC-daily, where the edge
is strongest per §5, is testable later with minimal glue.)*
