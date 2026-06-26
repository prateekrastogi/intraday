"""Directional ITM opening-range-breakout strategy -- the one validated edge.

Signal  : intraday breakout of the first ~8%-of-session opening range, confirmed by a strong close
          (day-high > OR-high AND close > OR-high AND (close-dayLow)/dayRange > 0.7; mirror for down).
Structure: buy ONE deep-ish ITM (~delta 0.65, `moneyness` in-the-money) option in the breakout direction,
          snapped to the nearest available strike on the signal day (guarantees a real entry mark).
Entry   : the held strike's last-bar close on the signal day.
Stop    : TRUE 1-minute intraday stop at -`stop`. Each hold day: open <= threshold -> fill at open
          (overnight gap-through, can exceed the stop); else intraday low touches -> fill at -stop. THE EDGE.
Winner  : if never stopped, exit at the held strike's last-bar close on day N (`hold` trading days).
Expiry  : front expiry with >= `hold` TRADING days to run (so the option survives the hold); the hold is
          additionally capped at the expiry day. Real option marks; intrinsic floor only if a mark is missing.

Parametrised by Inst (config.py): the SAME code runs NIFTY-weekly and MCX Crude. Pure pandas/numpy + real
1-min Parquet; no pricing model (moneyness, not greeks); no lookahead (signal acts at its own day's close).
"""

from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict

import numpy as np
import pandas as pd

from volarb.strategy.config import Inst

ANN = 252


def _num(df, col):
    return pd.to_numeric(df[col], errors="coerce").to_numpy() if col in df.columns else None


def _bars(path, cfg: Inst):
    df = pd.read_parquet(path)
    if "datetime" in df.columns:  # commodity / stock files carry a datetime column
        t = pd.to_datetime(df["datetime"])
        m = (t.dt.time >= cfg.session_start) & (t.dt.time <= cfg.session_end)
        df = df.loc[m].assign(_t=t[m]).sort_values("_t")
    return df  # nifty3y files have no datetime column: already session-ordered


def or_feat(path, cfg: Inst):
    """(OR-high, OR-low, day-high, day-low, day-close) or None."""
    df = _bars(path, cfg)
    hi, lo, cl = _num(df, "high"), _num(df, "low"), _num(df, "close")
    if hi is None or len(hi) < cfg.min_bars:
        return None
    if "_t" in df.columns:
        om = (df["_t"].dt.time <= cfg.or_end).to_numpy()
        if not om.any():  # no bars in the opening-range window (odd/half session) -> skip the day
            return None
        orh, orl = np.nanmax(hi[om]), np.nanmin(lo[om])
    else:
        orh, orl = np.nanmax(hi[: cfg.or_minutes]), np.nanmin(lo[: cfg.or_minutes])
    return float(orh), float(orl), float(np.nanmax(hi)), float(np.nanmin(lo)), float(cl[-1])


def opt_bars(path, cfg: Inst):
    """(open0, day-low, last-close) for an option's 1-min file, or None."""
    df = _bars(path, cfg)
    op, lo, cl = _num(df, "open"), _num(df, "low"), _num(df, "close")
    if cl is None or not np.isfinite(cl).any():
        return None
    o0 = float(op[0]) if op is not None and np.isfinite(op[0]) else float(cl[np.isfinite(cl)][0])
    return o0, float(np.nanmin(lo)), float(cl[np.isfinite(cl)][-1])


def build(cfg: Inst):
    """Load the lake into the day/expiry/strike index needed for a backtest run."""
    root = __import__("pathlib").Path(cfg.root)
    intr = {}
    for f in sorted((root / "underlying").glob("*.parquet")):
        v = or_feat(f, cfg)
        if v:
            intr[dt.date.fromisoformat(f.stem)] = v
    EDS: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    allexp: set = set()
    for f in (root / "options").glob("expiry=*/right=*/strike=*/*.parquet"):
        pr = f.parts
        e = dt.date.fromisoformat([p for p in pr if p.startswith("expiry=")][0].split("=")[1])
        r = [p for p in pr if p.startswith("right=")][0].split("=")[1]
        k = int([p for p in pr if p.startswith("strike=")][0].split("=")[1])
        EDS[e][dt.date.fromisoformat(f.stem)][r].add(k)
        allexp.add(e)
    bymonth: dict = defaultdict(list)
    for e in allexp:
        bymonth[(e.year, e.month)].append(e)
    monthly = {max(v) for v in bymonth.values()}
    return {"opt": root / "options", "intr": intr, "days": sorted(intr), "EDS": EDS, "monthly": monthly}


def signal(intr_d):
    orh, orl, dh, dl, c = intr_d
    rng = dh - dl if dh > dl else 1e9
    if dh > orh and c > orh and (c - dl) / rng > 0.7:
        return 1
    if dl < orl and c < orl and (dh - c) / rng > 0.7:
        return -1
    return 0


def run(cfg: Inst, B=None):
    """Walk the lake, return [(exit_date, net_return), ...] and the real-mark coverage %."""
    B = B or build(cfg)
    intr, days, EDS, monthly, opt = B["intr"], B["days"], B["EDS"], B["monthly"], B["opt"]
    n = len(days)

    def front(d, i):
        def tdte(e):
            return sum(1 for j in range(i + 1, n) if days[j] <= e)
        cand = [e for e in EDS if d in EDS[e] and e > d and EDS[e][d].get("CE") and EDS[e][d].get("PE")
                and tdte(e) >= cfg.hold]
        if cfg.expiry == "monthly":
            cand = [e for e in cand if e in monthly]
        return min(cand) if cand else None

    def ofile(e, r, k, d):
        return opt / f"expiry={e.isoformat()}" / f"right={r}" / f"strike={int(k)}" / f"{d.isoformat()}.parquet"

    trades, cov, i = [], [0, 0], 20
    while i < n:
        d = days[i]
        s = signal(intr[d])
        if s == 0:
            i += 1
            continue
        spot, r = intr[d][4], ("CE" if s > 0 else "PE")
        e = front(d, i)
        if e is None:
            i += 1
            continue
        avail = EDS[e][d].get(r)
        if not avail:
            i += 1
            continue
        target = spot * (1 - cfg.moneyness) if s > 0 else spot * (1 + cfg.moneyness)
        k = min(avail, key=lambda x: abs(x - target))
        end = min(i + cfg.hold, n - 1)
        while end > i and days[end] > e:  # never hold past expiry
            end -= 1
        ef = ofile(e, r, k, d)
        if not ef.exists():
            i += 1
            continue
        eb = opt_bars(ef, cfg)
        cov[1] += 1
        cov[0] += 1
        if eb is None or eb[2] <= 0:
            i += 1
            continue
        p0 = eb[2]
        thr = (1 - cfg.stop) * p0
        ret = None
        for j in range(i + 1, end + 1):
            mf = ofile(e, r, k, days[j])
            m = opt_bars(mf, cfg) if mf.exists() else None
            cov[1] += 1
            cov[0] += int(m is not None)
            if m is not None:
                o0, low, _ = m
                if o0 <= thr:
                    ret = o0 / p0 - 1
                    break
                if low <= thr:
                    ret = -cfg.stop
                    break
        if ret is None:
            dd = days[end]
            mf = ofile(e, r, k, dd)
            m = opt_bars(mf, cfg) if mf.exists() else None
            px = m[2] if m is not None else (max(intr[dd][4] - k, 0.0) if r == "CE" else max(k - intr[dd][4], 0.0))
            ret = px / p0 - 1
        trades.append((days[end], ret - cfg.spread))
        i = end + 1
    return trades, 100 * cov[0] / max(1, cov[1])


def metrics(trades, n_days, size=0.20):
    """Per-trade stats + compounded equity (fixed `size` of capital per trade) + max drawdown."""
    a = np.array([r for _, r in trades])
    if len(a) == 0:
        return None
    eq, peak, mdd = 1.0, 1.0, 0.0
    for r in a:
        eq *= 1 + size * r
        peak = max(peak, eq)
        mdd = min(mdd, eq / peak - 1)
    srt = np.sort(a)[::-1]
    asc = srt[::-1]
    tot = a.sum() if a.sum() != 0 else 1e-9
    return {
        "n": len(a), "win": 100 * (a > 0).mean(), "ev": a.mean() * 100,
        "sharpe": (a.mean() / a.std() * math.sqrt(len(a) / (n_days / ANN))) if a.std() > 0 else 0.0,
        "final_eq": eq, "max_dd": mdd * 100,
        # fat-tail / winner-concentration: top-N share of total PnL, and EV/trade after dropping the top-N winners
        "top1_pnl": 100 * srt[0] / tot, "top3_pnl": 100 * srt[:3].sum() / tot,
        "ev_drop1": asc[:-1].mean() * 100 if len(a) > 1 else 0.0,
        "ev_drop3": asc[:-3].mean() * 100 if len(a) > 3 else 0.0,
    }
