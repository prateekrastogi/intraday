"""Breeze backfill orchestrator: per-day front-weekly ATM band -> partitioned parquet.

Strategy-shaped pull (IMPLEMENTATION.md sec.2/5 -- intraday, front weekly expiry, near-ATM):
  per trading day -> nearest weekly expiry (DISCOVERED, not computed: NSE moved the NIFTY weekly
  expiry weekday repeatedly across 2023-26) -> that day's ATM +/- n strikes x {call,put}, 1-min.

Index/holiday days return empty (Status 200) and self-skip. Resumable via a JSONL manifest.
Rate-limited (RateLimiter pacing) and Status-5 aware (classify_response). All fetch windows are
IST literal via ist_window (Breeze treats from/to as IST, ignores the 'Z').
"""
from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .history import atm_strike_band, candles_to_frame, ist_window
from .ratelimit import DailyBudgetExceeded, RateLimiter, classify_response

EXPIRY_WEEKDAYS = (0, 1, 2, 3)   # Mon..Thu -- NSE has used all of these for NIFTY weekly, 2023-26
RIGHTS = ("call", "put")


def _expiry_req(d: date) -> str:
    """Breeze request format for an expiry date (06:00 IST literal; the time is cosmetic)."""
    return f"{d.isoformat()}T06:00:00.000Z"


def candidate_expiries(day: date, horizon_days: int = 8) -> list[date]:
    """Plausible expiry dates >= day within horizon, on weekdays NSE has used (earliest first)."""
    return [day + timedelta(days=k)
            for k in range(horizon_days + 1)
            if (day + timedelta(days=k)).weekday() in EXPIRY_WEEKDAYS]


def _call(bz, limiter: RateLimiter, sleep, **kw):
    """One get_historical_data_v2 call, paced + Status-5 aware. Returns (classification, response)."""
    resp = None
    for _ in range(3):
        limiter.acquire()                       # paces; raises DailyBudgetExceeded at budget
        resp = bz.get_historical_data_v2(**kw)
        cls = classify_response(resp)
        if cls == "rate_minute":
            sleep(60); continue                 # server-side per-minute limit -> back off, retry
        if cls == "rate_day":
            raise DailyBudgetExceeded("server-side daily limit hit")
        return cls, resp
    return "rate_minute", resp


def discover_expiry(bz, day: date, limiter, probe_strike: int, cache: dict, sleep) -> date | None:
    """Front weekly expiry >= day: probe candidates at probe_strike, earliest non-empty wins.

    Cache reuses the last expiry only while day <= it; once day passes the expiry the front rolls
    to the next one, so we re-discover (the expiry weekday is NOT fixed across 2023-26).
    """
    last = cache.get("last")
    if last is not None and day <= last:
        return last
    f, t = ist_window(day, "09:15", "09:16")    # narrow probe window (1-2 candles)
    found = None
    for c in candidate_expiries(day):
        cls, _ = _call(bz, limiter, sleep, interval="1minute", from_date=f, to_date=t,
                       stock_code="NIFTY", exchange_code="NFO", product_type="options",
                       expiry_date=_expiry_req(c), right="call", strike_price=str(int(probe_strike)))
        if cls == "ok":
            found = c
            break
    cache["last"] = found
    return found


def _load_manifest(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["k"] for line in path.read_text().splitlines() if line.strip()}


def _mark(path: Path, done: set[str], key: str) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps({"k": key}) + "\n")
    done.add(key)


def _write(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def backfill(bz, *, start: date, end: date, out_dir, n_strikes: int = 4, step: int = 50,
             limiter: RateLimiter | None = None, sleep=time.sleep) -> dict:
    """Pull [start, end]: underlying index 1-min + front-weekly ATM+/-n option 1-min -> parquet. Resumable."""
    out = Path(out_dir)
    limiter = limiter or RateLimiter()
    manifest = out / "manifest.jsonl"
    out.mkdir(parents=True, exist_ok=True)
    done = _load_manifest(manifest)
    day_exp: dict = {}
    for k in done:
        if k.startswith("exp|"):
            _, dd, ee = k.split("|", 2)
            day_exp[dd] = ee
    expiry_cache: dict = {}
    stats = {"days": 0, "und": 0, "opt": 0, "empty": 0, "skip": 0, "holidays": 0}

    d = start
    while d <= end:
        if d.weekday() >= 5:                     # weekend
            d += timedelta(days=1); continue

        # --- underlying (also gives the day's spot for ATM) ---
        und_key = f"und|{d}"
        und_path = out / "underlying" / f"{d}.parquet"
        if und_key in done:
            stats["skip"] += 1
            und = pd.read_parquet(und_path) if und_path.exists() else pd.DataFrame()
        else:
            f, t = ist_window(d)
            _, resp = _call(bz, limiter, sleep, interval="1minute", from_date=f, to_date=t,
                            stock_code="NIFTY", exchange_code="NSE", product_type="cash")
            und = candles_to_frame(resp)
            if und.empty:                        # holiday / no session
                stats["holidays"] += 1; _mark(manifest, done, und_key); d += timedelta(days=1); continue
            _write(und_path, und); _mark(manifest, done, und_key); stats["und"] += 1
        if und.empty:
            d += timedelta(days=1); continue

        spot = float(und.iloc[0]["open"])
        atm = round(spot / step) * step
        marker = day_exp.get(str(d))
        if marker is not None:                      # expiry already discovered in a prior run
            expiry = None if marker == "none" else date.fromisoformat(marker)
        else:
            expiry = discover_expiry(bz, d, limiter, atm, expiry_cache, sleep)
            _mark(manifest, done, f"exp|{d}|{expiry.isoformat() if expiry else 'none'}")
            day_exp[str(d)] = expiry.isoformat() if expiry else "none"
        if expiry is None:
            d += timedelta(days=1); continue

        # --- options: front-weekly ATM +/- n strikes x {call, put} ---
        for strike in atm_strike_band(spot, n=n_strikes, step=step):
            for right in RIGHTS:
                key = f"opt|{d}|{expiry}|{right}|{strike}"
                if key in done:
                    stats["skip"] += 1; continue
                f, t = ist_window(d)
                _, resp = _call(bz, limiter, sleep, interval="1minute", from_date=f, to_date=t,
                                stock_code="NIFTY", exchange_code="NFO", product_type="options",
                                expiry_date=_expiry_req(expiry), right=right, strike_price=str(int(strike)))
                frame = candles_to_frame(resp)
                if not frame.empty:
                    rc = "CE" if right == "call" else "PE"
                    fp = out / "options" / f"expiry={expiry}" / f"right={rc}" / f"strike={int(strike)}" / f"{d}.parquet"
                    _write(fp, frame); stats["opt"] += 1
                else:
                    stats["empty"] += 1
                _mark(manifest, done, key)

        stats["days"] += 1
        d += timedelta(days=1)
    return stats
