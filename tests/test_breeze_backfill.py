"""Orchestrator tests using a FakeBreeze (no live session) -- exercise discovery/band/resume/holiday."""
from datetime import date

from volarb.adapters.breeze.backfill import backfill, candidate_expiries
from volarb.adapters.breeze.ratelimit import RateLimiter

_NOSLEEP = lambda *_: None
_FAST = lambda: RateLimiter(min_interval=0.0, sleep=_NOSLEEP, clock=lambda: 0.0)

_OHLC2 = lambda spot: [
    {"datetime": "D 09:15:00", "open": spot, "high": spot, "low": spot, "close": spot, "volume": 10},
    {"datetime": "D 09:16:00", "open": spot, "high": spot, "low": spot, "close": spot, "volume": 12},
]


class FakeBreeze:
    """Underlying: 2 rows except on holidays. Options: rows only when expiry is a Tuesday & strike in band."""
    def __init__(self, holidays=()):
        self.holidays = set(holidays)
        self.calls = []

    def get_historical_data_v2(self, **kw):
        self.calls.append(kw)
        day = kw["from_date"][:10]
        if kw["exchange_code"] == "NSE":                       # underlying / cash
            if day in self.holidays:
                return {"Status": 200, "Success": [], "Error": None}
            rows = _OHLC2(24000.0)
            for r in rows: r["datetime"] = r["datetime"].replace("D", day)
            return {"Status": 200, "Success": rows, "Error": None}
        # options (NFO): valid only when the chosen expiry is a Tuesday (fake's expiry rule)
        exp = date.fromisoformat(kw["expiry_date"][:10])
        strike = float(kw["strike_price"])
        if exp.weekday() == 1 and 23800 <= strike <= 24200:    # Tuesday + within ATM+/-4 band
            rows = _OHLC2(strike / 100.0)
            for r in rows:
                r["datetime"] = r["datetime"].replace("D", day)
                r["open_interest"] = 5000
            return {"Status": 200, "Success": rows, "Error": None}
        return {"Status": 200, "Success": [], "Error": None}    # wrong expiry -> empty


def test_candidate_expiries_weekdays_only_earliest_first():
    cands = candidate_expiries(date(2026, 6, 1))               # Monday
    assert cands[0] == date(2026, 6, 1)                        # Mon included, earliest first
    assert all(c.weekday() in (0, 1, 2, 3) for c in cands)     # Mon..Thu only
    assert all(c >= date(2026, 6, 1) for c in cands)           # never before `day`
    assert cands == sorted(cands)


def test_backfill_discovers_tuesday_and_writes_band(tmp_path):
    bz = FakeBreeze()
    stats = backfill(bz, start=date(2026, 6, 1), end=date(2026, 6, 1), out_dir=tmp_path,
                     n_strikes=4, step=50, limiter=_FAST(), sleep=_NOSLEEP)
    # Mon 1-Jun -> front weekly = Tue 2-Jun; 9 strikes x 2 rights all in band -> 18 option files
    assert stats["und"] == 1
    assert stats["opt"] == 18
    assert (tmp_path / "underlying" / "2026-06-01.parquet").exists()
    opt = tmp_path / "options" / "expiry=2026-06-02" / "right=CE" / "strike=24000" / "2026-06-01.parquet"
    assert opt.exists()


def test_backfill_resumes_via_manifest(tmp_path):
    bz1 = FakeBreeze()
    backfill(bz1, start=date(2026, 6, 1), end=date(2026, 6, 2), out_dir=tmp_path, limiter=_FAST(), sleep=_NOSLEEP)
    n_first = len(bz1.calls)
    bz2 = FakeBreeze()
    stats2 = backfill(bz2, start=date(2026, 6, 1), end=date(2026, 6, 2), out_dir=tmp_path, limiter=_FAST(), sleep=_NOSLEEP)
    assert len(bz2.calls) == 0                                 # everything already in manifest
    assert stats2["skip"] > 0 and stats2["und"] == 0 and stats2["opt"] == 0
    assert n_first > 0


def test_backfill_skips_holiday(tmp_path):
    bz = FakeBreeze(holidays={"2026-06-01"})
    stats = backfill(bz, start=date(2026, 6, 1), end=date(2026, 6, 1), out_dir=tmp_path, limiter=_FAST(), sleep=_NOSLEEP)
    assert stats["holidays"] == 1
    assert stats["opt"] == 0                                   # no option calls on a holiday
    assert not (tmp_path / "options").exists()
