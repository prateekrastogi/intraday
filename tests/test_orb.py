"""Unit tests for the ORB-ITM strategy core (signal logic, metrics, bar readers) -- synthetic, no real lake."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volarb.strategy import orb
from volarb.strategy.config import CRUDE, NIFTY_WEEKLY, Inst


def test_signal_up_down_none():
    # (OR-high, OR-low, day-high, day-low, close)
    assert orb.signal((100, 98, 110, 97, 109)) == 1     # breaks OR-high, closes strong near the high
    assert orb.signal((100, 98, 101, 88, 89)) == -1     # breaks OR-low, closes weak near the low
    assert orb.signal((100, 98, 110, 90, 100)) == 0     # broke OR-high but closed mid-range -> no signal
    assert orb.signal((100, 98, 100.5, 99.5, 100)) == 0  # never broke the OR


def test_metrics_basic():
    from datetime import date
    trades = [(date(2024, 1, 1), -0.12), (date(2024, 1, 2), -0.12), (date(2024, 1, 3), +0.90)]
    m = orb.metrics(trades, n_days=252, size=0.20)
    assert m["n"] == 3
    assert round(m["win"]) == 33
    assert m["ev"] == pytest.approx((-0.12 - 0.12 + 0.90) / 3 * 100, abs=1e-6)
    # equity: 0.976 * 0.976 * 1.18
    assert m["final_eq"] == pytest.approx(0.976 * 0.976 * 1.18, abs=1e-6)
    assert m["max_dd"] < 0


def test_metrics_empty():
    assert orb.metrics([], 252) is None


def test_or_feat_positional_no_datetime(tmp_path):
    # nifty-style file: no datetime column, session-ordered; OR = first or_minutes bars
    cfg = NIFTY_WEEKLY
    n = 80
    hi = np.full(n, 100.0)
    hi[10], hi[40] = 105.0, 120.0   # OR-high (first 30 bars) = 105; day-high = 120
    lo = np.full(n, 99.0)
    lo[5] = 95.0
    cl = np.full(n, 100.0)
    cl[-1] = 119.0
    f = tmp_path / "2024-01-01.parquet"
    pd.DataFrame({"open": cl, "high": hi, "low": lo, "close": cl}).to_parquet(f)
    orh, orl, dh, dl, c = orb.or_feat(f, cfg)
    assert orh == 105.0 and dh == 120.0 and c == 119.0 and dl == 95.0
    assert orb.signal((orh, orl, dh, dl, c)) == 1


def test_or_feat_datetime_session(tmp_path):
    # crude-style file: datetime column; OR window selected by clock time (<= or_end)
    cfg = CRUDE
    idx = pd.date_range("2024-01-01 09:00", "2024-01-01 12:00", freq="1min")
    hi = np.full(len(idx), 50.0)
    hi[5] = 52.0     # within OR (09:05) -> OR-high 52
    hi[150] = 60.0   # 11:30, after OR -> day-high 60
    lo = np.full(len(idx), 49.0)
    cl = np.full(len(idx), 50.0)
    cl[-1] = 59.5
    f = tmp_path / "2024-01-01.parquet"
    pd.DataFrame({"datetime": idx.astype(str), "open": cl, "high": hi, "low": lo, "close": cl}).to_parquet(f)
    orh, orl, dh, dl, c = orb.or_feat(f, cfg)
    assert orh == 52.0 and dh == 60.0 and c == 59.5


def test_opt_bars(tmp_path):
    f = tmp_path / "2024-01-01.parquet"
    pd.DataFrame({"open": [10.0, 11, 12], "high": [10.0, 11, 12], "low": [9.0, 8, 11], "close": [10.0, 9, 12]}).to_parquet(f)
    o0, low, last = orb.opt_bars(f, NIFTY_WEEKLY)
    assert o0 == 10.0 and low == 8.0 and last == 12.0


def test_min_bars_skips_thin_day(tmp_path):
    f = tmp_path / "2024-01-01.parquet"
    pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]}).to_parquet(f)
    assert orb.or_feat(f, NIFTY_WEEKLY) is None


def test_configs_sane():
    for cfg in (NIFTY_WEEKLY, CRUDE):
        assert isinstance(cfg, Inst)
        assert 0 < cfg.moneyness < 0.1 and cfg.hold >= 1 and 0 < cfg.stop < 1
        assert cfg.session_start < cfg.or_end <= cfg.session_end
