"""Orchestrator tests with a FakeBreeze (no live session): multi-instrument pull + robustness."""

from datetime import date

from volarb.adapters.breeze.backfill import (
    backfill,
    candidate_expiries,
    futures_expiry_candidates,
    _last_weekday_of_month,
)
from volarb.adapters.breeze.ratelimit import RateLimiter


def _NO(*_):
    return None


def _FAST(**kw):
    return RateLimiter(min_interval=0.0, sleep=_NO, clock=lambda: 0.0, **kw)


def _rows(day, spot, oi=None, vol=10):
    out = []
    for mn in ("09:15:00", "09:16:00"):
        r = {
            "datetime": f"{day} {mn}",
            "open": spot,
            "high": spot,
            "low": spot,
            "close": spot,
            "volume": vol,
        }
        if oi is not None:
            r["open_interest"] = oi
        out.append(r)
    return {"Status": 200, "Success": out, "Error": None}


_EMPTY = {"Status": 200, "Success": [], "Error": None}


class FakeBreeze:
    """index/VIX -> 2 rows (empty on holidays); futures valid iff expiry=last-Tue-of-month;
    options valid iff expiry is a Tuesday AND strike within ATM(24000)+/-6 band."""

    def __init__(self, holidays=()):
        self.holidays = set(holidays)
        self.calls = []

    def get_historical_data_v2(self, **kw):
        self.calls.append(kw)
        day = kw["from_date"][:10]
        ex, pt, stock = kw["exchange_code"], kw.get("product_type", ""), kw.get("stock_code")
        if ex == "NSE":  # cash: index or VIX
            if day in self.holidays:
                return _EMPTY
            return _rows(day, 13.0 if stock == "INDVIX" else 24000.0)
        if ex == "NFO" and pt == "futures":
            exp = date.fromisoformat(kw["expiry_date"][:10])
            if exp == _last_weekday_of_month(exp.year, exp.month, 1):  # last Tuesday = monthly
                return _rows(day, 24115.0, oi=16_000_000, vol=56_000)
            return _EMPTY
        if ex == "NFO" and pt == "options":
            exp = date.fromisoformat(kw["expiry_date"][:10])
            strike = float(kw["strike_price"])
            if exp.weekday() == 1 and 23700 <= strike <= 24300:  # Tuesday + within ATM+/-6
                return _rows(day, strike / 100.0, oi=5000, vol=1000)
            return _EMPTY
        return _EMPTY


class FlakyBreeze(FakeBreeze):
    """Raises on the first `fail_first` calls (network/SDK blip), then behaves normally."""

    def __init__(self, fail_first=2, **kw):
        super().__init__(**kw)
        self.fail_first = fail_first
        self.n = 0

    def get_historical_data_v2(self, **kw):
        self.n += 1
        if self.n <= self.fail_first:
            raise ConnectionError("transient blip")
        return super().get_historical_data_v2(**kw)


def test_candidate_expiries_weekdays_only_earliest_first():
    c = candidate_expiries(date(2026, 6, 1))
    assert c[0] == date(2026, 6, 1) and c == sorted(c)
    assert all(x.weekday() in (0, 1, 2, 3) and x >= date(2026, 6, 1) for x in c)


def test_futures_expiry_candidates_month_end_only():
    c = futures_expiry_candidates(date(2026, 6, 1))
    assert date(2026, 6, 30) in c  # last Tuesday of June 2026 (current regime)
    assert date(2026, 6, 25) in c  # last Thursday (historical regime, also probed)
    assert c == sorted(c) and all(x >= date(2026, 6, 1) for x in c)


def test_backfill_pulls_all_instruments_and_two_expiries(tmp_path):
    bz = FakeBreeze()
    s = backfill(
        bz,
        start=date(2026, 6, 1),
        end=date(2026, 6, 2),
        out_dir=tmp_path,
        limiter=_FAST(),
        sleep=_NO,
        log=_NO,
    )
    assert s["und"] == 2 and s["fut"] == 2 and s["vix"] == 2
    # day1(Mon): front 18 + 2nd 18 = 36 ; day2(Tue=front-expiry): front widened 26 + 2nd 18 = 44
    assert s["opt"] == 80
    for series in ("underlying", "futures", "vix"):
        assert (tmp_path / series / "2026-06-01.parquet").exists()
    assert (
        tmp_path
        / "options"
        / "expiry=2026-06-02"
        / "right=CE"
        / "strike=24000"
        / "2026-06-01.parquet"
    ).exists()
    assert (
        tmp_path
        / "options"
        / "expiry=2026-06-09"
        / "right=PE"
        / "strike=24000"
        / "2026-06-01.parquet"
    ).exists()  # 2nd expiry


def test_backfill_widens_front_on_its_expiry_day(tmp_path):
    bz = FakeBreeze()
    backfill(
        bz,
        start=date(2026, 6, 1),
        end=date(2026, 6, 2),
        out_dir=tmp_path,
        n_strikes=4,
        expiry_day_n=6,
        limiter=_FAST(),
        sleep=_NO,
        log=_NO,
    )
    wing = tmp_path / "options" / "expiry=2026-06-02" / "right=CE" / "strike=23700"  # ATM-6
    assert (wing / "2026-06-02.parquet").exists()  # widened on the expiry day
    assert not (wing / "2026-06-01.parquet").exists()  # NOT on a normal day (band was +/-4)


def test_backfill_resumes_zero_calls(tmp_path):
    backfill(
        FakeBreeze(),
        start=date(2026, 6, 1),
        end=date(2026, 6, 2),
        out_dir=tmp_path,
        limiter=_FAST(),
        sleep=_NO,
        log=_NO,
    )
    bz2 = FakeBreeze()
    s = backfill(
        bz2,
        start=date(2026, 6, 1),
        end=date(2026, 6, 2),
        out_dir=tmp_path,
        limiter=_FAST(),
        sleep=_NO,
        log=_NO,
    )
    assert len(bz2.calls) == 0
    assert s["skip"] > 0 and s["opt"] == 0 and s["und"] == 0 and s["fut"] == 0 and s["vix"] == 0


def test_backfill_skips_holiday(tmp_path):
    bz = FakeBreeze(holidays={"2026-06-01"})
    s = backfill(
        bz,
        start=date(2026, 6, 1),
        end=date(2026, 6, 1),
        out_dir=tmp_path,
        limiter=_FAST(),
        sleep=_NO,
        log=_NO,
    )
    assert s["holidays"] == 1 and s["fut"] == 0 and s["vix"] == 0 and s["opt"] == 0
    assert not (tmp_path / "options").exists()


def test_backfill_retries_transient_error(tmp_path):
    bz = FlakyBreeze(fail_first=2)  # first 2 calls raise -> _call backs off and retries
    s = backfill(
        bz,
        start=date(2026, 6, 1),
        end=date(2026, 6, 1),
        out_dir=tmp_path,
        limiter=_FAST(),
        sleep=_NO,
        log=_NO,
    )
    assert s["und"] == 1  # underlying recovered despite the blip
    assert (tmp_path / "underlying" / "2026-06-01.parquet").exists()


def test_backfill_stops_cleanly_on_budget(tmp_path):
    bz = FakeBreeze()
    s = backfill(
        bz,
        start=date(2026, 6, 1),
        end=date(2026, 6, 5),
        out_dir=tmp_path,
        limiter=_FAST(daily_budget=5),
        sleep=_NO,
        log=_NO,
    )
    assert s["stopped_on_budget"] is True  # caught, no exception escaped
    assert isinstance(s, dict)


# --------------------------------------------------------------------------------------
# GAP-FILLING TESTS (added by math/coverage audit)
# These pin properties the docstring claims but the original suite only "ran":
# weekly+2nd expiry ROLL across a week, discover-None, second_expiry=False,
# budget-stop -> resume-to-completion convergence, exhausted-retry cross-run resume,
# mid-week holiday, and the futures month-end cache reuse.
# --------------------------------------------------------------------------------------

import json  # noqa: E402
import os  # noqa: E402


def _markers(tmp_path, prefix):
    """Return sorted manifest markers with the given prefix."""
    man = tmp_path / "manifest.jsonl"
    if not man.exists():
        return []
    keys = [json.loads(line)["k"] for line in man.read_text().splitlines() if line.strip()]
    return sorted(k for k in keys if k.startswith(prefix))


class NoOptBreeze(FakeBreeze):
    """index/VIX/futures behave; options ALWAYS empty -> no weekly expiry is ever discovered."""

    def get_historical_data_v2(self, **kw):
        self.calls.append(kw)
        ex, pt = kw["exchange_code"], kw.get("product_type", "")
        if ex == "NFO" and pt == "options":
            return _EMPTY
        return super().get_historical_data_v2(**kw)


# --- 1. WEEKLY + 2nd EXPIRY ROLL across a full week (the headline docstring claim) ---
def test_backfill_rolls_weekly_and_second_expiry_across_week(tmp_path):
    """Mon/Tue use front=06-02; from Wed (after front expired) front rolls to 06-09 and
    the 2nd weekly rolls 06-09 -> 06-16. Pins the in-run cache `d <= last` roll rule."""
    bz = FakeBreeze()
    backfill(
        bz,
        start=date(2026, 6, 1),
        end=date(2026, 6, 5),
        out_dir=tmp_path,
        limiter=_FAST(),
        sleep=_NO,
        log=_NO,
    )
    assert _markers(tmp_path, "expw0") == [
        "expw0|2026-06-01|2026-06-02",
        "expw0|2026-06-02|2026-06-02",
        "expw0|2026-06-03|2026-06-09",  # rolled the day AFTER expiry
        "expw0|2026-06-04|2026-06-09",
        "expw0|2026-06-05|2026-06-09",
    ]
    assert _markers(tmp_path, "expw1") == [
        "expw1|2026-06-01|2026-06-09",
        "expw1|2026-06-02|2026-06-09",
        "expw1|2026-06-03|2026-06-16",  # 2nd weekly rolled in lock-step
        "expw1|2026-06-04|2026-06-16",
        "expw1|2026-06-05|2026-06-16",
    ]
    # all three distinct weekly expiries actually got option parquet written
    written = {p.split("expiry=")[1].split(os.sep)[0] for p in _glob_parquets(tmp_path / "options")}
    assert {"2026-06-02", "2026-06-09", "2026-06-16"} <= written


def _glob_parquets(root):
    import glob

    return glob.glob(str(root / "**" / "*.parquet"), recursive=True)


# --- 2. FUTURES month-end cache: reused all month, never re-discovered ---
def test_backfill_futures_expiry_pinned_to_month_end_all_week(tmp_path):
    bz = FakeBreeze()
    backfill(
        bz,
        start=date(2026, 6, 1),
        end=date(2026, 6, 5),
        out_dir=tmp_path,
        limiter=_FAST(),
        sleep=_NO,
        log=_NO,
    )
    futm = _markers(tmp_path, "expfut")
    assert all(m.endswith("2026-06-30") for m in futm)  # last Tuesday of June, every day
    assert len(futm) == 5


# --- 3. discover returns None -> 'none' persisted, options skipped, no 2nd-expiry probe ---
def test_backfill_no_weekly_expiry_records_none_and_skips_options(tmp_path):
    bz = NoOptBreeze()
    s = backfill(
        bz,
        start=date(2026, 6, 1),
        end=date(2026, 6, 1),
        out_dir=tmp_path,
        limiter=_FAST(),
        sleep=_NO,
        log=_NO,
    )
    assert s["opt"] == 0 and s["und"] == 1 and s["fut"] == 1
    assert _markers(tmp_path, "expw0") == ["expw0|2026-06-01|none"]
    assert _markers(tmp_path, "expw1") == []  # e1 is None -> 2nd expiry never probed
    assert not (tmp_path / "options").exists()


# --- 4. second_expiry=False: only the front weekly is pulled, no w1 marker ---
def test_backfill_second_expiry_disabled(tmp_path):
    bz = FakeBreeze()
    s = backfill(
        bz,
        start=date(2026, 6, 1),
        end=date(2026, 6, 1),
        out_dir=tmp_path,
        n_strikes=4,
        second_expiry=False,
        limiter=_FAST(),
        sleep=_NO,
        log=_NO,
    )
    # Mon front only: ATM+/-4 = 9 strikes x 2 rights = 18, no 2nd expiry
    assert s["opt"] == 18
    assert _markers(tmp_path, "expw1") == []
    assert not (tmp_path / "options" / "expiry=2026-06-09").exists()


# --- 5. BUDGET STOP mid-day -> resume completes everything -> 3rd run zero calls ---
def test_backfill_budget_stop_then_resume_to_completion(tmp_path):
    s1 = backfill(
        FakeBreeze(),
        start=date(2026, 6, 1),
        end=date(2026, 6, 5),
        out_dir=tmp_path,
        limiter=_FAST(daily_budget=40),  # spent mid first day
        sleep=_NO,
        log=_NO,
    )
    assert s1["stopped_on_budget"] is True

    bz2 = FakeBreeze()  # fresh token, ample budget -> finishes the range
    s2 = backfill(
        bz2,
        start=date(2026, 6, 1),
        end=date(2026, 6, 5),
        out_dir=tmp_path,
        limiter=_FAST(daily_budget=4500),
        sleep=_NO,
        log=_NO,
    )
    assert s2["stopped_on_budget"] is False and s2["days"] == 5

    bz3 = FakeBreeze()  # everything persisted -> nothing left to do
    s3 = backfill(
        bz3,
        start=date(2026, 6, 1),
        end=date(2026, 6, 5),
        out_dir=tmp_path,
        limiter=_FAST(),
        sleep=_NO,
        log=_NO,
    )
    assert len(bz3.calls) == 0 and s3["opt"] == 0 and s3["und"] == 0


# --- 6. EXHAUSTED retries: day left UNMARKED (not just recovered within a run) and resumes ---
def test_backfill_exhausted_retry_leaves_day_unmarked_then_resumes(tmp_path):
    bz = FlakyBreeze(fail_first=4)  # >= _RETRIES -> underlying never succeeds this run
    s = backfill(
        bz,
        start=date(2026, 6, 1),
        end=date(2026, 6, 1),
        out_dir=tmp_path,
        limiter=_FAST(),
        sleep=_NO,
        log=_NO,
    )
    assert s["und"] == 0 and s["error"] == 1 and s["days"] == 0
    assert _markers(tmp_path, "und") == []  # NOT marked -> eligible for retry
    assert not (tmp_path / "underlying" / "2026-06-01.parquet").exists()

    bz2 = FlakyBreeze(fail_first=0)  # healthy now
    s2 = backfill(
        bz2,
        start=date(2026, 6, 1),
        end=date(2026, 6, 1),
        out_dir=tmp_path,
        limiter=_FAST(),
        sleep=_NO,
        log=_NO,
    )
    assert s2["und"] == 1
    assert (tmp_path / "underlying" / "2026-06-01.parquet").exists()


# --- 7. MID-WEEK holiday: skipped + counted, surrounding days still pulled, holiday writes nothing ---
def test_backfill_midweek_holiday_skipped_neighbors_pulled(tmp_path):
    bz = FakeBreeze(holidays={"2026-06-02"})  # Tue holiday between Mon and Wed
    s = backfill(
        bz,
        start=date(2026, 6, 1),
        end=date(2026, 6, 3),
        out_dir=tmp_path,
        limiter=_FAST(),
        sleep=_NO,
        log=_NO,
    )
    assert s["holidays"] == 1
    assert s["und"] == 2  # Mon + Wed pulled
    assert (tmp_path / "underlying" / "2026-06-01.parquet").exists()
    assert (tmp_path / "underlying" / "2026-06-03.parquet").exists()
    assert not (tmp_path / "underlying" / "2026-06-02.parquet").exists()
    # holiday day produced no futures/vix/option writes (it self-skips before them)
    assert s["fut"] == 2 and s["vix"] == 2


# --- 8. WEEKEND days are silently skipped (no calls, no day count) ---
def test_backfill_weekend_only_range_makes_no_calls(tmp_path):
    bz = FakeBreeze()
    s = backfill(
        bz,
        start=date(2026, 6, 6),  # Saturday
        end=date(2026, 6, 7),  # Sunday
        out_dir=tmp_path,
        limiter=_FAST(),
        sleep=_NO,
        log=_NO,
    )
    assert len(bz.calls) == 0 and s["days"] == 0 and s["und"] == 0


# --- 9. candidate_expiries horizon boundary: excludes Friday, spans into next week ---
def test_candidate_expiries_excludes_friday_and_spans_horizon():
    c = candidate_expiries(date(2026, 6, 1))  # Monday, horizon 8
    assert date(2026, 6, 5) not in c  # Friday excluded
    assert date(2026, 6, 6) not in c and date(2026, 6, 7) not in c  # weekend excluded
    assert date(2026, 6, 8) in c and date(2026, 6, 9) in c  # next Mon/Tue within horizon
    assert max(c) <= date(2026, 6, 1) + __import__("datetime").timedelta(days=8)


# --- 10. futures_expiry_candidates near month-end pulls in NEXT month's expiries ---
def test_futures_expiry_candidates_rolls_into_next_month():
    c = futures_expiry_candidates(date(2026, 6, 29))  # past June's last Thu(25)/Tue(30)?
    # June last Tuesday = 30 (>= 29) included; June last Thu = 25 (< 29) excluded;
    # July last Tue/Thu included.
    assert date(2026, 6, 30) in c
    assert date(2026, 6, 25) not in c
    assert any(x.month == 7 for x in c)
    assert c == sorted(c)
