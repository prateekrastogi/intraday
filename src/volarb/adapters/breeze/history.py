"""Breeze historical data ingestion -> Nautilus ParquetDataCatalog (for the backtest).

# === WRITE OURSELVES (the backtest-data half of the data layer) ===
"""
# TODO: pull Breeze historical 1-sec/1-min NIFTY-FUTURES underlying + per-strike option chains
#       INCLUDING EXPIRED contracts (Breeze is the best free source for these), and write them to
#       the Nautilus ParquetDataCatalog that run/backtest.py reads. Distinct from data.py (the LIVE
#       LiveMarketDataClient); THIS catalog is what feeds the walk-forward OOS backtest (the gate).
