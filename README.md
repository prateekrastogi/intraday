# volarb — intraday options research (NIFTY · stocks · MCX)

> This repo began as an intraday NIFTY **long-volatility** straddle/strangle vol-arb engine (the realized-vs-implied
> *spread* thesis). Three years of real 1-minute OOS backtesting **refuted that thesis** and
> surfaced a different, validated edge. **This README is the current state of findings.** The old straddle build
> plan (`PLAN.md`) is superseded and removed; engine/stack notes are at the bottom; dev rules live in `CLAUDE.md`.

## TL;DR
- **Long-vol has no harvestable edge on real data.** Straddle / strangle / ratio-backspread all bleed the
  variance-risk-premium + theta + spread — on index *and* stocks, intraday *and* daily, under every gate tried.
  Apparent "compression / sell-into-expansion" wins were **weekly-expiry gamma artifacts** that evaporate on the
  correct instrument. Buying vol *after* a move pays inflated IV; the move is already priced in.
- **The one validated edge is DIRECTIONAL, not vol:** an intraday **opening-range breakout** → buy a **deep-ish ITM
  (~δ0.65) option** in the breakout direction → **cut losses fast at −12% (true 1-min intraday stop), ride winners
  ~5 trading days.** Win rate ~20%, but a handful of +70–300% winners drive a positive, **convex (fat-right-tail)** EV.
- **The cut-loss-fast stop *is* the edge** — strip it and the ITM goes negative. It's an *exit-discipline* edge that
  rides best on a leveraged, downside-capped ITM option; not an options-structure trick.

## Deployable configurations (real 1-min data · net of spread · no-lookahead · OOS)
| Vehicle | Config | Sharpe (per-trade) | EV/trade | max DD | Note |
|---|---|---|---|---|---|
| **10-stock monthly basket** | ITM δ0.65, hold 5d | **0.91** | +2.6% | **−13%** | diversification cuts DD ~3× vs single — smoothest, most robust |
| NIFTY weekly | ITM δ0.65, hold 5d | ~0.7–0.9 | +7–9% | ~−40% | biggest per-trade punch (weekly gamma); single instrument |
| RELIANCE monthly | ITM, hold 5d | ~0.3 | +3% | — | ≈ NIFTY-monthly — instrument barely matters at equal expiry |

## Winner concentration — the fat-tail tell
The edge is convex (cut losers small, ride rare big winners), so a handful of trades carry most of the PnL. The
honest test is **how much EV/trade survives once you drop the top winners** — printed by `run.backtest`. Full sample:

| Vehicle | EV/trade | top-1 / top-3 share of PnL | EV net of top-1 | EV net of top-3 |
|---|---|---|---|---|
| NIFTY-weekly | +8.3% | 46% / 127% | +4.5% | **−2.3%** |
| Crude | +5.5% | 98% / 210% | **+0.1%** | −6.7% |
| 10-stock basket (pooled, 788 tr) | +2.6% | 32% / 59% | positive | **positive** |

Single instruments are **one-to-three-trade lotteries**: drop the top 3 and NIFTY-weekly turns negative, and
Crude's *entire* edge is a single trade (net of top-1 ≈ 0). The **diversified basket is the least concentrated** —
full-sample top-3 = 59% vs NIFTY's 127% / Crude's 210% — because pooling ~10× the trades samples the winner tail
better. **But it is not robust either:** the 59% is the *full 3-year* number; on the held-out ~1-year OOS slice
(≈260 trades) the basket's top-3 was also **≈102% of PnL** (verified), so drop-top-3 is negative *there* too.
Diversification *reduces* concentration but does **not** make a thin window drop-top-3-proof — **no vehicle here is.**
Treat every Sharpe as an optimistic upper bound; the edge is only trustworthy across **many independent trades**
(the full-sample basket), never a single year or a single instrument.

## Findings across the fields tested

**Moneyness** (apples-to-apples, *matched* hold N=5, 10-stock basket): **ITM > ATM > OTM**, monotonic on Sharpe AND
drawdown — ITM 0.91/−13%, ATM 0.75/−21%, OTM 0.63/−24%. Going OTM trades reliability for nothing. *Always match the
hold when comparing moneyness — different N confounds moneyness with hold period.*

**Hold period:** N=5 (trading days) beats N=3 at every moneyness (let winners run); N=3 barely clears zero.

**Expiry:** weekly ≫ monthly (~2.5× the Sharpe) — short-dated gamma converts the continuation to intrinsic faster.
**Only the index has weeklies** (NSE single-stock options are monthly-only), so stocks/basket are structurally
capped at the weaker monthly regime.

**Diversification (10-stock basket):** the genuine benefit is on **drawdown** (−13% portfolio vs −40% single) and
**tail-concentration** (top-3 trades = 59% of PnL vs 103% single), *not* a Sharpe-ratio multiple (an earlier
"1.75×" claim was an activity-fraction artifact, refuted by adversarial verification). ~Half the basket names lose;
the basket works by **averaging idiosyncratic winners**, not universal edge.

**Fat-tail / lottery nature:** ~80% of trades hit the stop; EV rides on rare monster winners. On thin samples
(one stock, ~30 OOS trades) you **cannot** separate edge from luck — the basket (~260 OOS trades) is the first
statistically meaningful confirmation. Treat single-instrument Sharpes as optimistic upper bounds.

**Risk overlays** (causal, pre-registered, no-snoop): only a **drawdown brake** (de-risk in deep drawdowns) helps
robustly across ITM/ATM/OTM (cuts max DD at ~neutral Sharpe). Tail-cutters (take-profit cap, equity-MA filter)
**degrade** the convex edge; a consecutive-loss filter looked great on one variant but was a **snooping trap**
(failed on the others — the cross-variant consistency check exposed it). *You can cap the drawdown; you cannot cap
the winners or sit out.*

**MCX commodities** (most-liquid: Crude, NatGas — verified live): the ORB-ITM edge does **not** clearly transfer.
NatGas is a reliable negative (Sharpe −1.2, equity → 0.37×, −67% DD over 3yr). Crude looks positive (~0.7) but on
only ~34 trades over a recent ~16-month window (shallow Breeze crude-option history) → **anecdotal**. MCX options
liquidity: Crude + NatGas dominate; **gold options are effectively dead** (liquidity is in gold *futures*); silver
only via the mini. (The ORB opening range must be scaled to session length — ~8% — else it over-fires on the
14.9-hour MCX session.)

**Dead ends (confirmed no edge):** long-vol straddle/strangle/backspread (compression); compression /
sell-into-expansion; overnight→open vol harvest (open-spread-fragile, regime-dependent); intraday-momentum on
*futures* (STT eats it); gamma-regime gating; the 20-day *daily* Donchian breakout (the *intraday* ORB is what
carries the edge — same instruments score ~0.1 on the daily breakout vs ~0.8 on the intraday one).

## Leveraged futures vs. convex options — and the gap-market boundary

We tested the natural objection: *the edge is the cut-loss-fast stop, so why pay option theta — just run the
same ORB on a leveraged future?* The honest comparison requires **matched risk**: size both vehicles by the
*same* standard, pre-registered, no-snoop rule (vol-targeting 10/15/20%, fixed-fractional 1–2% per trade,
fractional Kelly), on the *same* paired ORB trades, with the future's underlying stop set to the option's −12%
premium stop ÷ its embedded leverage (ω≈32 NIFTY, ≈16 Crude). Result (independently re-implemented, exact match):

- **Under matched sizing, final equity ranks by per-trade Sharpe + convexity — leverage adds nothing.** The
  earlier full-capital high-leverage "ruin" (10–50× ending <1× final equity) was an **over-sizing artifact**: at
  any standard sizing rule *neither* vehicle ruins. Leverage is Sharpe-invariant; only the return *shape* matters.
- **NIFTY: the option wins every rule — but the edge is inside the noise.** At the matched stop (0.375%),
  continuous-future Sharpe 0.61 < option 0.70 and the option is more convex (skew +3.55 vs +2.64), so the option
  ends richer (vol-target-15% 1.32× vs 1.27×; half-Kelly 2.30× vs 1.71×). *But* the paired-bootstrap
  Sharpe-difference 90% CI is [−0.27, +0.42] (straddles zero, P(option wins)≈66%) and the win leans on one +321%
  trade (drop it → half-Kelly 2.30×→1.34×). A prior "continuous NIFTY future 0.83 > 0.70" figure was a
  tighter-non-matched-stop (0.24%=ω50) / full-signal-set artifact and does **not** reproduce on the matched set.
- **Crude: the future wins every rule, and the win is statistically real.** Continuous-future Sharpe 1.01 (gappy
  0.83) ≫ crude-option 0.39; paired-bootstrap P(future > option) = 1.00 (cont) / 0.98 (gappy), CI excludes zero;
  robust to dropping the future's own best trades. *But* this is largely the crude **option being a bad wrapper**
  (50% per-trade vol, a −100% total-loss trade, thin/illiquid MCX premium) rather than the future being special —
  and n=35 from a recent shallow window, so the ranking is reliable in-sample but the magnitude is tail-specific.

**The gap-market boundary (why the option's cap is structural).** An option caps loss at the premium
**regardless of whether the market is open** — gap-proof by construction. A linear future's loss is uncapped
through an overnight/weekend gap, and no stop fills inside a gap. So the verdict splits on market hours *and* on
how good the option wrapper is on that instrument:

| Market | Gap risk | Best vehicle (matched sizing) |
|---|---|---|
| **NIFTY weekly (overnight + weekend)** | uncapped gap-through | **option** — cheap weekly convexity (skew +3.55) + premium cap; the future never beats it under matched sizing, though only inside the noise band |
| **MCX Crude (overnight)** | uncapped gap-through | **future** — even the *realistic gappy* future beats the thin/expensive crude option under every sizing rule; the option wrapper is the weak link here |
| **24/7 continuous (crypto perps)** | none — stops fill cleanly | tight-stop future is most competitive (no gap to cap), but only where its Sharpe ≥ the option's |
| **US index futures (ES/NQ, ~23/5)** | weekend + flash-crash gaps remain | not truly continuous — the gap-cap argument still applies |

Bottom line: under risk-matched, no-snoop sizing the answer is **instrument-specific, set by Sharpe + convexity,
not by the vehicle per se**. Where the option is a cheap, convex wrapper (NIFTY weekly) it wins and its
premium-cap is a free structural bonus; where the option is thin and expensive (MCX crude) the leverage-equivalent
future wins outright. A leveraged future earns no Sharpe premium from leverage — it only wins when the option
wrapper is the weaker vehicle, and even then a closing market re-exposes it to the uncapped gap risk the option
eliminates by construction.

## Methodology / data-ethics standards
True 1-minute intraday stops (gap-through aware) · real option marks (intrinsic floor only for genuinely-missing,
which is conservative) · split/bonus back-adjustment + cross-ex-date exclusion · **matched-hold apples-to-apples** ·
pre-registered variants with **held-out test splits** (no threshold grid-search) · **adversarial independent
re-implementation** of every headline result. That verification caught real errors and kept the findings honest —
a delta-fill winner fabrication, a weekly-expiry compression artifact, the diversification-ratio artifact, and the
consec-loss snooping trap. Winner-robustness is always reported (drop-top-N, leave-one-stock-out, block-bootstrap CI).

## Data assets (real, on the research VM under `~/marketdata/`)
- `nifty3y` — 3yr NIFTY 1-min cash + weekly & monthly options (40k+ strike files) + wide daily store.
- `reliance3y` — 3yr RELIANCE 1-min cash + options.
- `stockbasket/` — 10 large-caps (HDFCBANK, ICICIBANK, INFY, TCS, SBIN, AXISBANK, LT, ITC, BHARTIARTL, KOTAKBANK):
  1-min cash (757d) + 1-min options at ITM/ATM/OTM (43k+ leaf files) + daily band store.
- `commodity/{CRUDE,NATGAS}` — 3yr 1-min front-month futures + ITM/ATM/OTM options (MCX via Breeze).

Breeze data caveats: daily server-side rate limit (shared per token; refresh = ICICI login → `BREEZE_SESSION_TOKEN`
in `.env`); proprietary stock codes; MCX via `exchange_code="MCX"`, codes `CRUDE`/`NATGAS` (futures & options expire
on different dates); crude option 1-min history is shallow/recent.

## Engine / stack
NautilusTrader (engine + IV/greeks) · statsmodels (HAR) · breeze-connect (broker + 1-min data incl. expired
options) · **uv** for env. Quality-open-source-first; write only the Breeze adapter, signals, strategy, and NSE
cost model. See `CLAUDE.md`.
```
uv run --no-sync python -m volarb.run.backtest   # walk-forward OOS on the Parquet catalog
```
Research harnesses (NIFTY/basket/commodity ITM-ATM-OTM 3-way, overlays, verification) live as standalone scripts;
the strategy-layer promotion to the engine is pending a final deployable-config decision.
