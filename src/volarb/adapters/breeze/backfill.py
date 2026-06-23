"""Breeze backfill orchestrator (robust, multi-instrument): per trading day pull a TIME-CONSISTENT
cross-section -> partitioned parquet, resumable.

Per trading day:
  - NIFTY INDEX 1-min            (underlying/{date}.parquet)   -- sanity + ATM/moneyness reference
  - NIFTY front-month FUTURES    (futures/{date}.parquet)      -- true-volume underlying + Black-76 forward
  - INDIA VIX  (INDVIX)          (vix/{date}.parquet)          -- RV-IV regime benchmark
  - FRONT weekly options         ATM+/-n (ATM+/-expiry_day_n on its own expiry day) x {CE,PE}
  - 2nd weekly options           ATM+/-n x {CE,PE}             -- term structure (Johnson SLOPE)

All expiries (weekly x2, monthly futures) are DISCOVERED by probing (NSE expiry weekday drifted 2023-26),
cached in-run AND persisted to the manifest for zero-call resume. Index/holiday days return empty
(Status 200) and self-skip. Robust to transient errors (retry+backoff, never marked done -> retried next
run) and to the daily budget (clean stop with stopped_on_budget=True so the next token resumes). All
windows are IST literal via ist_window. Reuses candles_to_frame/ist_window/atm_strike_band unchanged.
"""
from __future__ import annotations

import calendar
import json
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .history import atm_strike_band, candles_to_frame, ist_window
from .ratelimit import DailyBudgetExceeded, RateLimiter, classify_response

EXPIRY_WEEKDAYS = (0, 1, 2, 3)        # Mon..Thu -- NSE has used all of these for NIFTY weekly, 2023-26
FUT_EXPIRY_WEEKDAYS = (1, 3)          # monthly futures: Tuesday (current) + Thursday (historical)
RIGHTS = ("call", "put")
_RETRIES = 4


def _expiry_req(d: date) -> str:
    """Breeze request format for an expiry date (06:00 IST literal; the time is cosmetic)."""
    return f"{d.isoformat()}T06:00:00.000Z"


def candidate_expiries(day: date, horizon_days: int = 8) -> list[date]:
    """Plausible WEEKLY expiry dates >= day within horizon, on weekdays NSE has used (earliest first)."""
    return [day + timedelta(days=k)
            for k in range(horizon_days + 1)
            if (day + timedelta(days=k)).weekday() in EXPIRY_WEEKDAYS]


def _last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    d = date(year, month, calendar.monthrange(year, month)[1])
    while d.weekday() != weekday:
        d -= timedelta(days=1)
    return d


def futures_expiry_candidates(day: date) -> list[date]:
    """Plausible MONTHLY futures expiries >= day (this month + next, Tue/Thu), earliest first."""
    nxt = (day.year + 1, 1) if day.month == 12 else (day.year, day.month + 1)
    cands = {_last_weekday_of_month(y, m, wd)
             for (y, m) in ((day.year, day.month), nxt) for wd in FUT_EXPIRY_WEEKDAYS}
    return sorted(c for c in cands if c >= day)


def _call(bz, limiter: RateLimiter, sleep, **kw):
    """One get_historical_data_v2 call: paced, retried w/ backoff, Status-5 aware.
    Returns (classification, response): 'ok'|'empty' on success, 'error' after retries (caller skips
    marking -> retried next run). Raises DailyBudgetExceeded so the run can stop cleanly."""
    last = ("error", {"Status": -1, "Error": "no attempt"})
    for attempt in range(_RETRIES):
        limiter.acquire()                                   # paces; raises DailyBudgetExceeded at budget
        try:
            resp = bz.get_historical_data_v2(**kw)
        except Exception as e:                              # network/SDK blip -> backoff + retry
            last = ("error", {"Status": -1, "Error": repr(e)}); sleep(2 * (attempt + 1)); continue
        cls = classify_response(resp)
        if cls == "rate_minute":
            sleep(60); continue                             # server-side per-minute limit
        if cls == "rate_day":
            raise DailyBudgetExceeded("server-side daily limit hit")
        if cls == "error":
            last = (cls, resp); sleep(2 * (attempt + 1)); continue
        return cls, resp                                    # 'ok' or 'empty'
    return last


def discover_weekly_expiry(bz, day, limiter, probe_strike, sleep, *, after=None):
    """Earliest weekly option expiry that returns data on `day` (strictly > `after` if given), else None."""
    f, t = ist_window(day, "09:15", "09:16")                # narrow probe window
    start = (after + timedelta(days=1)) if after else day
    for c in candidate_expiries(start):
        cls, _ = _call(bz, limiter, sleep, interval="1minute", from_date=f, to_date=t,
                       stock_code="NIFTY", exchange_code="NFO", product_type="options",
                       expiry_date=_expiry_req(c), right="call", strike_price=str(int(probe_strike)))
        if cls == "ok":
            return c
    return None


def discover_futures_expiry(bz, day, limiter, sleep):
    """Earliest monthly futures expiry that returns data on `day`, else None."""
    f, t = ist_window(day, "09:15", "09:16")
    for c in futures_expiry_candidates(day):
        cls, _ = _call(bz, limiter, sleep, interval="1minute", from_date=f, to_date=t,
                       stock_code="NIFTY", exchange_code="NFO", product_type="futures",
                       expiry_date=_expiry_req(c))
        if cls == "ok":
            return c
    return None


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


def _fetch_store(bz, limiter, sleep, manifest, done, key, path, **kw):
    """Fetch one series -> store + mark. Returns (status, frame). 'error' is NOT marked (retried)."""
    if key in done:
        return "skip", (pd.read_parquet(path) if path.exists() else pd.DataFrame())
    cls, resp = _call(bz, limiter, sleep, **kw)
    if cls == "error":
        return "error", None                                # leave unmarked -> retried next run
    frame = candles_to_frame(resp)
    if frame.empty:
        _mark(manifest, done, key); return "empty", frame
    _write(path, frame); _mark(manifest, done, key); return "ok", frame


def _persisted(done: set[str], prefix: str) -> dict[str, str]:
    """Recover {date_str: value} from manifest markers like '<prefix>|{date}|{value}'."""
    out = {}
    for k in done:
        if k.startswith(prefix + "|"):
            _, dd, vv = k.split("|", 2)
            out[dd] = vv
    return out


def _resolve_expiry(marker: str | None):
    if marker is None:
        return None, False                                  # not yet discovered
    return (None if marker == "none" else date.fromisoformat(marker)), True


def backfill(bz, *, start: date, end: date, out_dir, n_strikes: int = 4, expiry_day_n: int = 6,
             step: int = 50, second_expiry: bool = True, limiter: RateLimiter | None = None,
             sleep=time.sleep, log=print) -> dict:
    """Pull [start, end]: index + futures + VIX + front & 2nd weekly ATM-band options -> parquet.
    Resumable + robust. Stops cleanly on the daily budget (stats['stopped_on_budget'])."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    limiter = limiter or RateLimiter()
    manifest = out / "manifest.jsonl"
    done = _load_manifest(manifest)
    day_w0 = _persisted(done, "expw0")
    day_w1 = _persisted(done, "expw1")
    day_fut = _persisted(done, "expfut")
    c_w0: dict = {}; c_w1: dict = {}; c_fut: dict = {}      # in-run discovery caches
    stats = {"days": 0, "und": 0, "fut": 0, "vix": 0, "opt": 0,
             "empty": 0, "skip": 0, "error": 0, "holidays": 0, "stopped_on_budget": False}

    try:
        d = start
        while d <= end:
            if d.weekday() >= 5:                            # weekend
                d += timedelta(days=1); continue

            # --- underlying index (spot + holiday detector) ---
            st, und = _fetch_store(bz, limiter, sleep, manifest, done,
                                   f"und|{d}", out / "underlying" / f"{d}.parquet",
                                   interval="1minute", **_ix(d))
            stats["und"] += st == "ok"; stats["skip"] += st == "skip"; stats["error"] += st == "error"
            if st == "error":                               # transient -> retry whole day next run
                d += timedelta(days=1); continue
            if st == "empty" or und is None or und.empty:   # holiday / no session
                stats["holidays"] += st == "empty"; d += timedelta(days=1); continue

            spot = float(und.iloc[0]["open"])
            atm = round(spot / step) * step

            # --- front-month futures (monthly expiry discovered/persisted) ---
            fm, have = _resolve_expiry(day_fut.get(str(d)))
            if not have:
                last = c_fut.get("last")
                fm = last if (last is not None and d <= last) else discover_futures_expiry(bz, d, limiter, sleep)
                c_fut["last"] = fm
                _mark(manifest, done, f"expfut|{d}|{fm.isoformat() if fm else 'none'}")
                day_fut[str(d)] = fm.isoformat() if fm else "none"
            if fm is not None:
                st, _ = _fetch_store(bz, limiter, sleep, manifest, done,
                                     f"fut|{d}", out / "futures" / f"{d}.parquet",
                                     interval="1minute", **_fut(d, fm))
                stats["fut"] += st == "ok"; stats["error"] += st == "error"

            # --- India VIX ---
            st, _ = _fetch_store(bz, limiter, sleep, manifest, done,
                                 f"vix|{d}", out / "vix" / f"{d}.parquet",
                                 interval="1minute", **_vix(d))
            stats["vix"] += st == "ok"; stats["error"] += st == "error"

            # --- discover front (w0) + 2nd (w1) weekly expiries (persisted) ---
            e1, have = _resolve_expiry(day_w0.get(str(d)))
            if not have:
                last = c_w0.get("last")
                e1 = last if (last is not None and d <= last) else discover_weekly_expiry(bz, d, limiter, atm, sleep)
                c_w0["last"] = e1
                _mark(manifest, done, f"expw0|{d}|{e1.isoformat() if e1 else 'none'}")
                day_w0[str(d)] = e1.isoformat() if e1 else "none"

            e2 = None
            if second_expiry and e1 is not None:
                e2, have = _resolve_expiry(day_w1.get(str(d)))
                if not have:
                    e2 = c_w1[e1] if e1 in c_w1 else discover_weekly_expiry(bz, d, limiter, atm, sleep, after=e1)
                    c_w1[e1] = e2
                    _mark(manifest, done, f"expw1|{d}|{e2.isoformat() if e2 else 'none'}")
                    day_w1[str(d)] = e2.isoformat() if e2 else "none"

            # --- options: front weekly (widen on its own expiry day) + 2nd weekly ---
            for expiry, n in ((e1, expiry_day_n if e1 == d else n_strikes), (e2, n_strikes)):
                if expiry is None:
                    continue
                for strike in atm_strike_band(spot, n=n, step=step):
                    for right in RIGHTS:
                        key = f"opt|{d}|{expiry}|{right}|{strike}"
                        if key in done:
                            stats["skip"] += 1; continue
                        cls, resp = _call(bz, limiter, sleep, interval="1minute", **_opt(d, expiry, right, strike))
                        if cls == "error":
                            stats["error"] += 1; continue   # leave unmarked -> retried next run
                        frame = candles_to_frame(resp)
                        if frame.empty:
                            stats["empty"] += 1
                        else:
                            rc = "CE" if right == "call" else "PE"
                            _write(out / "options" / f"expiry={expiry}" / f"right={rc}" / f"strike={int(strike)}" / f"{d}.parquet", frame)
                            stats["opt"] += 1
                        _mark(manifest, done, key)

            stats["days"] += 1
            if stats["days"] % 25 == 0:
                log(f"  ...{stats['days']} days done @ {d} | calls={limiter.calls_made} | {stats}")
            d += timedelta(days=1)
    except DailyBudgetExceeded:
        stats["stopped_on_budget"] = True
        log(f"DAILY BUDGET reached at {d} after {limiter.calls_made} calls -- resume with a fresh token. {stats}")
    return stats


def _ix(d):
    f, t = ist_window(d)
    return dict(from_date=f, to_date=t, stock_code="NIFTY", exchange_code="NSE", product_type="cash")


def _fut(d, fm):
    f, t = ist_window(d)
    return dict(from_date=f, to_date=t, stock_code="NIFTY", exchange_code="NFO",
                product_type="futures", expiry_date=_expiry_req(fm))


def _vix(d):
    f, t = ist_window(d)
    return dict(from_date=f, to_date=t, stock_code="INDVIX", exchange_code="NSE", product_type="cash")


def _opt(d, expiry, right, strike):
    f, t = ist_window(d)
    return dict(from_date=f, to_date=t, stock_code="NIFTY", exchange_code="NFO", product_type="options",
                expiry_date=_expiry_req(expiry), right=right, strike_price=str(int(strike)))
