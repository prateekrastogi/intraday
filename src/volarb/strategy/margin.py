"""Broker-aligned portfolio margin for the backspread book -- NO Black-Scholes.

A ratio backspread is a DEFINED-RISK hedged structure, so the broker's SPAN+exposure initial margin
~= the structure's own MAX-LOSS (the most the hedged position can lose) + a small exposure add-on,
computed straight from the payoff -- no option repricing. This is the at-expiry worst case; the real
intraday SPAN is LOWER (the long wings keep time value), so sizing off max-loss is conservative/safe.
LIVE: use the broker's actual SPAN number/API. The hedge benefit ("a sell lowers margin") is realized
only with LONGS-FIRST execution (backspread.py emits legs longs-first) -- otherwise the short draws full
naked margin until the longs fill.
"""

from __future__ import annotations

from volarb.strategy.backspread import Backspread, max_loss_rupees


def structure_margin(bs: Backspread, lot_size: int, forward: float, *, exposure_frac: float = 0.0) -> float:
    """Initial margin (rupees) for one hedged backspread = max-loss + exposure_frac * short notional.

    ``exposure_frac`` (default 0.0 = pure defined-risk max-loss) adds the small NSE exposure margin on the
    short notional if you want to be conservative; for a fully-hedged backspread brokers charge ~the SPAN
    scan loss (≈ max-loss) with little/no extra exposure.
    """
    ml = max_loss_rupees(bs, lot_size)
    short_notional = bs.n_short * forward * lot_size
    return ml + max(0.0, exposure_frac) * short_notional


def incremental_margin(margin_book: float, margin_book_plus: float) -> float:
    """Extra margin to add a structure to an existing book = margin(book+structure) - margin(book).

    For independent defined-risk structures this is ~the structure's own margin; when the new structure's
    worst scenario opposes the book's it can be lower (the offset the user wants). Caller supplies both
    margins (each via the broker's number live, or summed max-loss in backtest).
    """
    return max(0.0, margin_book_plus - margin_book)
