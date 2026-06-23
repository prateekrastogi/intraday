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
