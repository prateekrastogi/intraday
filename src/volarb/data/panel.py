"""Polars data-lake alignment for the intraday vol panel.

This is the *only* alignment/IO module in the repo: it turns the raw per-day
parquet lake into clean, session-aligned polars frames that the signal modules
(HAR-RV, Lee-Mykland, semivariance, spread) consume. All vol/IV math lives in
``volarb.pricing.greeks`` and is reused here -- this module re-implements none of
it.

Lake layout (``root`` points at one store under ``~/marketdata/``)::

    root/underlying/{date}.parquet
    root/futures/{date}.parquet
    root/vix/{date}.parquet
    root/options/expiry={YYYY-MM-DD}/right={CE,PE}/strike={int}/{date}.parquet

Each parquet is a single NSE session: a datetime index (written as a column) plus
``open, high, low, close, volume`` and, for options/futures, ``open_interest``.
A regular session is 375 one-minute bars, 09:15-15:30 IST.

No-lookahead / no-roll discipline enforced here:
  * ``log_ret`` is computed strictly *within* a session, so the first bar is null
    -- no return is ever taken across the overnight gap or a futures monthly roll.
  * Prices are reindexed onto the canonical 375-min grid but never blanket
    forward-filled (that would fabricate zero returns and bias RV down). Only
    open-interest may be ffilled, as it is a stock not a flow.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl

from volarb.pricing.greeks import atm_iv, years_to_expiry

__all__ = [
    "SESSION_OPEN",
    "SESSION_CLOSE",
    "SESSION_MINUTES",
    "trading_days",
    "load_underlying",
    "load_vix",
    "session_grid",
    "atm_iv_series",
    "term_slope",
]

SESSION_OPEN = dt.time(9, 15)
SESSION_CLOSE = dt.time(15, 30)
SESSION_MINUTES = 375  # inclusive 09:15..15:29 close-of-bar; 375 one-min bars

# Near-ATM band: keep strikes within this fraction of the forward when building
# the ATM-IV quote set. Tight enough to stay on the flat at-the-money part of the
# smile (the nearest strikes), where the single-vol vega-weighted inversion is
# robust; the deep wings are deliberately excluded. ~1.5% nets the 2-3 nearest
# 50-pt NIFTY strikes around a ~20k forward.
_ATM_BAND = 0.015


def _date_str(day: dt.date) -> str:
    return day.isoformat()


def _read_session_parquet(path: Path) -> pl.DataFrame:
    """Read one day's parquet and normalise the datetime index to a ``datetime`` column."""
    df = pl.read_parquet(path)
    if "datetime" not in df.columns:
        # pandas writes the index under "__index_level_0__" or the first temporal column.
        ts_cols = [c for c, t in zip(df.columns, df.dtypes) if t in (pl.Datetime, pl.Date)]
        if ts_cols:
            df = df.rename({ts_cols[0]: "datetime"})
        elif df.columns:
            df = df.rename({df.columns[0]: "datetime"})
    return df.with_columns(pl.col("datetime").cast(pl.Datetime("us"))).sort("datetime")


def session_grid(day: dt.date) -> pl.DataFrame:
    """The canonical 375-row one-minute grid for ``day`` (09:15..15:29 IST)."""
    start = dt.datetime.combine(day, SESSION_OPEN)
    return pl.select(
        pl.datetime_range(
            start,
            start + dt.timedelta(minutes=SESSION_MINUTES - 1),
            interval="1m",
            eager=True,
        ).alias("datetime")
    )


def _reindex_to_grid(df: pl.DataFrame, day: dt.date, ffill_oi: bool = False) -> pl.DataFrame:
    """Left-join ``df`` onto the 375-min grid. Prices stay null where missing
    (never blanket-ffilled); ``open_interest`` may be ffilled (stock, not flow)."""
    grid = session_grid(day)
    out = grid.join(df, on="datetime", how="left")
    if ffill_oi and "open_interest" in out.columns:
        out = out.with_columns(pl.col("open_interest").forward_fill())
    return out


def trading_days(root: str | Path, kind: str = "futures") -> list[dt.date]:
    """All session dates present in ``root/{kind}/`` (from ``{date}.parquet`` names)."""
    folder = Path(root) / kind
    days: list[dt.date] = []
    for p in folder.glob("*.parquet"):
        try:
            days.append(dt.date.fromisoformat(p.stem))
        except ValueError:
            continue
    return sorted(days)


def load_underlying(root: str | Path, day: dt.date, kind: str = "futures") -> pl.DataFrame:
    """Load one session of the underlying (futures by default) with a within-session
    ``log_ret`` column. The first bar's ``log_ret`` is null so no return crosses the
    overnight gap or a futures roll. Reindexed to the 375-min grid; OI ffilled."""
    path = Path(root) / kind / f"{_date_str(day)}.parquet"
    raw = _read_session_parquet(path)
    df = _reindex_to_grid(raw, day, ffill_oi=True)
    return df.with_columns(
        (pl.col("close").log() - pl.col("close").log().shift(1)).alias("log_ret")
    )


def load_vix(root: str | Path, day: dt.date) -> pl.DataFrame:
    """Load one session of India VIX as ``[datetime, vix]`` (vix = the close)."""
    path = Path(root) / "vix" / f"{_date_str(day)}.parquet"
    raw = _read_session_parquet(path)
    df = _reindex_to_grid(raw, day, ffill_oi=False)
    return df.select("datetime", pl.col("close").alias("vix"))


def _option_strikes(root: str | Path, expiry: dt.date, right: str) -> list[int]:
    folder = Path(root) / "options" / f"expiry={_date_str(expiry)}" / f"right={right}"
    strikes: list[int] = []
    for p in folder.glob("strike=*"):
        try:
            strikes.append(int(p.name.split("=", 1)[1]))
        except ValueError:
            continue
    return sorted(strikes)


def _load_option_closes(
    root: str | Path, day: dt.date, expiry: dt.date, right: str
) -> pl.DataFrame:
    """Long frame ``[datetime, strike, close, is_call]`` of every strike's close
    for one (day, expiry, right), reindexed to the session grid (prices null-padded)."""
    is_call = right == "CE"
    frames: list[pl.DataFrame] = []
    base = Path(root) / "options" / f"expiry={_date_str(expiry)}" / f"right={right}"
    for strike in _option_strikes(root, expiry, right):
        path = base / f"strike={strike}" / f"{_date_str(day)}.parquet"
        if not path.exists():
            continue
        raw = _read_session_parquet(path)
        frames.append(
            _reindex_to_grid(raw, day, ffill_oi=False).select(
                "datetime",
                pl.lit(strike, dtype=pl.Int64).alias("strike"),
                pl.col("close").cast(pl.Float64),
            )
        )
    if not frames:
        return pl.DataFrame(
            schema={
                "datetime": pl.Datetime("us"),
                "strike": pl.Int64,
                "close": pl.Float64,
                "is_call": pl.Boolean,
            }
        )
    return pl.concat(frames).with_columns(pl.lit(is_call).alias("is_call"))


def _expiry_close_dt(expiry: dt.date) -> dt.datetime:
    return dt.datetime.combine(expiry, SESSION_CLOSE)


def atm_iv_series(
    root: str | Path,
    day: dt.date,
    expiry: dt.date,
    r: float = 0.065,
) -> pl.DataFrame:
    """Per-minute self-built ATM implied-vol curve for ``(day, expiry)``.

    For each minute t with a valid futures close ``F`` (the forward), gather the
    near-ATM CE+PE option closes (strikes within ``_ATM_BAND`` of F) and invert via
    ``volarb.pricing.greeks.atm_iv`` at ``years_to_expiry`` measured from t to 15:30
    on ``expiry`` (calendar minutes). Minutes with <2 usable strike quotes are
    skipped. Returns ``[datetime, atm_iv]`` on a subset of the grid.
    """
    fut = load_underlying(root, day, kind="futures").select("datetime", "close")
    ce = _load_option_closes(root, day, expiry, "CE")
    pe = _load_option_closes(root, day, expiry, "PE")
    opts = pl.concat([ce, pe]) if ce.height or pe.height else ce
    expiry_close = _expiry_close_dt(expiry)

    rows: list[tuple[dt.datetime, float]] = []
    fut_close = dict(zip(fut["datetime"].to_list(), fut["close"].to_list()))
    # Group option quotes by minute once, then price each minute.
    by_minute: dict[dt.datetime, list[tuple[int, float, bool]]] = {}
    if opts.height:
        for ts, strike, close, is_call in zip(
            opts["datetime"], opts["strike"], opts["close"], opts["is_call"]
        ):
            if close is None:
                continue
            by_minute.setdefault(ts, []).append((int(strike), float(close), bool(is_call)))

    for ts in fut["datetime"].to_list():
        forward = fut_close.get(ts)
        if forward is None:
            continue
        quotes = [
            (k, px, call)
            for (k, px, call) in by_minute.get(ts, ())
            if abs(k - forward) <= _ATM_BAND * forward
        ]
        if len(quotes) < 2:
            continue
        minutes_left = (expiry_close - ts).total_seconds() / 60.0
        if minutes_left <= 0:
            continue
        iv = atm_iv(quotes, forward, years_to_expiry(minutes_left), r=r)
        if iv is not None:
            rows.append((ts, iv))

    if not rows:
        return pl.DataFrame(schema={"datetime": pl.Datetime("us"), "atm_iv": pl.Float64})
    return pl.DataFrame(rows, schema=["datetime", "atm_iv"], orient="row")


def term_slope(
    root: str | Path,
    day: dt.date,
    e1: dt.date,
    e2: dt.date,
    r: float = 0.065,
) -> pl.DataFrame:
    """Front-to-next ATM-IV term slope ``[datetime, slope]`` = ATM_IV(e2) - ATM_IV(e1).

    Negative slope = inverted (front IV > next) -- the documented long-vol pocket
    (Johnson 2017). Inner-joined on the minutes where both expiries price."""
    front = atm_iv_series(root, day, e1, r=r).rename({"atm_iv": "iv1"})
    nxt = atm_iv_series(root, day, e2, r=r).rename({"atm_iv": "iv2"})
    return (
        front.join(nxt, on="datetime", how="inner")
        .select("datetime", (pl.col("iv2") - pl.col("iv1")).alias("slope"))
        .sort("datetime")
    )
