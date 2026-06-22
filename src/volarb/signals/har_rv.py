"""HAR-RV (intraday-adapted) realized-vol forecast - the CORE (per IMPLEMENTATION.md sec.1).
RV from multi-lag intraday windows (e.g. 15/30/60-min), estimated by OLS + HAC SEs.

Optional upgrades - add ONLY if they beat OOS net-of-cost (<=4-6 param ceiling):
  - signed jumps + realized semivariance (Patton-Sheppard 2015) - leverage / index put-skew;
    the index-relevant jump treatment (rather than a plain continuous/jump RV split)
  - HARQ (Bollerslev-Patton-Quaedvlieg 2016) - quarticity-weighted measurement-error correction

# === ADOPTED estimator: statsmodels OLS + HAC ===
"""
# TODO: RV from 5-min returns; HAR lags; fit offline; online .predict().
# Lee-Mykland (signals/lee_mykland) supplies a jump REGIME FEATURE, not a forced continuous/jump split.
