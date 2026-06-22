"""NSE options fee+fill model as a Nautilus FeeModel (IMPLEMENTATION.md §4).

# === WRITE OURSELVES (no quality OSS lib covers this) ===
"""
# TODO: STT 0.10->0.15% sell premium + 0.125% ITM-exercise intrinsic; NSE txn 0.03553%; SEBI 0.0001%; GST 18% on (brokerage+txn+SEBI); stamp 0.003% buy; brokerage ~Rs20/order; EXPLICIT bid-ask spread per leg. Backtest costs MUST match live.
