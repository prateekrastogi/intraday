"""HAR-CJ realized-vol forecast: RV (d/w/m intraday lags) split into continuous + jump components.

# === ADOPTED: thin wrapper around a vetted library === (statsmodels OLS + HAC)
"""
# TODO: RV from 5-min returns; jump component from lee_mykland; fit offline; online .predict().
