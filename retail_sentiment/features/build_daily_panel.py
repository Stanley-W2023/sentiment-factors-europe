"""
build_daily_panel.py — Construct the daily analysis panel for horizon tests.

Extends the existing daily SAT computation to produce a fully self-contained
daily panel suitable for daily FMB regressions and daily 5×5 portfolio sorts.

Three SAT predictors are constructed at each date t to predict t+1 returns:

  sat_raw  — raw daily SAT = ATV_t × sign(r_t).  Maximum information per day
             but very noisy; captures single-day sentiment shocks.

  sat_5d   — 5-trading-day rolling mean of sat_raw. Smoothed over one week;
             reduces bid-ask bounce noise while retaining near-term signal.

  sat_10d  — 10-trading-day rolling mean of sat_raw. Closest to the monthly
             framework (approximately half a month); most comparable to the
             monthly FMB results.

PE quintiles are assigned daily using the most recent PE observation (already
forward-filled in the upstream daily panel). Since PE moves slowly, daily PE
quintiles are nearly identical to monthly quintiles but allow truly daily sorts.

Daily risk-free rate is approximated as monthly_rf / trading_days_in_month.
This is standard practice in daily asset pricing (see Fama & French daily
factor data documentation). The approximation error is negligible relative to
daily return variance.

Cross-sectional normalisation (winsorise + z-score) is applied daily to each
SAT variant — consistent with the monthly treatment in build_sat.py.

Microstructure note: This module does not implement skip-day prediction or
bid-ask bounce correction. Both the raw daily SAT and the single-day return
predictand share microstructure noise. Users should interpret raw daily results
with caution; the rolling variants (sat_5d, sat_10d) are more robust by
construction. See limitations discussion in the paper.
"""

import numpy as np
import pandas as pd

from features.build_sat import buildDailySATPanel
from config.constants import (
    PE_COL,
    RETURN_COL,
    MARKET_CAP_COL,
    RISK_FREE_COL,
    PE_QUINTILE_COL,
    N_QUINTILES,
    MIN_TURNOVER_CLIP,
)

# ── Column names for the three SAT predictors ─────────────────────────────────
SAT_RAW_COL  = "sat_raw"
SAT_5D_COL   = "sat_5d"
SAT_10D_COL  = "sat_10d"

SAT_RAW_Q_COL = "sat_raw_quintile"
SAT_5D_Q_COL  = "sat_5d_quintile"
SAT_10D_Q_COL = "sat_10d_quintile"

# ── Constants ─────────────────────────────────────────────────────────────────
_WINSOR_LOWER: float = 0.01
_WINSOR_UPPER: float = 0.99
_MIN_STOCKS_PER_DAY: int = 50   # Minimum cross-section size for quintile sort


def buildDailyAnalysisPanel(
    dailyData: pd.DataFrame,
    monthlyRF: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the daily analysis panel for horizon-sensitivity tests.

    Args:
        dailyData : Long-format daily panel from _reshapeToLongFormat in main.py.
                    Required columns: [date, ric, ret_eur, me_eur, pe_raw,
                    volume, shares_outstanding].
        monthlyRF : Monthly risk-free rate panel with columns
                    [year_month, rf_eur]. Period-indexed year_month.

    Returns:
        Daily panel with columns:
          [date, ric, ret_eur, me_eur, pe_trailing,
           sat_raw, sat_5d, sat_10d,
           pe_quintile, sat_raw_quintile, sat_5d_quintile, sat_10d_quintile,
           rf_daily, excess_ret_fwd]
        One row per (date, ric). Only dates with valid sat_raw are retained
        (i.e. after the 252-day AR warm-up window per stock).
    """
    print("  [daily panel] Computing daily SAT variants...")
    daily_sat = _buildDailySATVariants(dailyData)

    print("  [daily panel] Adding PE and market cap...")
    daily_sat = _mergePEAndMarketCap(daily_sat, dailyData)

    print("  [daily panel] Adding daily risk-free rate...")
    daily_sat = _addDailyRF(daily_sat, monthlyRF)

    print("  [daily panel] Assigning daily quintiles...")
    daily_sat = _assignDailyQuintiles(daily_sat)

    print("  [daily panel] Constructing forward returns...")
    daily_sat = _addForwardReturns(daily_sat)

    n_stocks = daily_sat["ric"].nunique()
    n_days   = daily_sat["date"].nunique()
    n_obs    = len(daily_sat)
    print(f"  [daily panel] {n_obs:,} stock-days  |  "
          f"{n_stocks} stocks  |  {n_days:,} trading days")

    return daily_sat


# ── private: SAT variants ─────────────────────────────────────────────────────

def _buildDailySATVariants(dailyData: pd.DataFrame) -> pd.DataFrame:
    """
    Compute sat_raw, sat_5d, and sat_10d for every stock-day.
    Applies cross-sectional winsorisation + z-scoring daily to each variant.
    """
    raw = buildDailySATPanel(dailyData)[["date", "ric", "sat_daily"]]
    raw = raw.rename(columns={"sat_daily": SAT_RAW_COL})
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values(["ric", "date"]).reset_index(drop=True)

    # Rolling variants — computed per stock in chronological order
    raw[SAT_5D_COL]  = (
        raw.groupby("ric")[SAT_RAW_COL]
        .transform(lambda s: s.rolling(5,  min_periods=3).mean())
    )
    raw[SAT_10D_COL] = (
        raw.groupby("ric")[SAT_RAW_COL]
        .transform(lambda s: s.rolling(10, min_periods=5).mean())
    )

    # Cross-sectional normalisation daily for each variant
    for col in [SAT_RAW_COL, SAT_5D_COL, SAT_10D_COL]:
        raw[col] = (
            raw.groupby("date")[col]
            .transform(_winsorize_and_zscore)
        )

    return raw


def _winsorize_and_zscore(s: pd.Series) -> pd.Series:
    """Winsorise at 1/99 pct then z-score. Applied cross-sectionally per day."""
    lo  = s.quantile(_WINSOR_LOWER)
    hi  = s.quantile(_WINSOR_UPPER)
    s   = s.clip(lower=lo, upper=hi)
    mu  = s.mean()
    std = s.std()
    if std < 1e-10:
        return s - mu
    return (s - mu) / std


# ── private: PE and market cap ────────────────────────────────────────────────

def _mergePEAndMarketCap(
    dailySAT: pd.DataFrame,
    dailyData: pd.DataFrame,
) -> pd.DataFrame:
    """Merge pe_raw, me_eur, and ret_eur onto the SAT panel.

    ret_eur is required by _addForwardReturns to build the next-day
    excess return. buildDailySATPanel only returns SAT columns, so we
    must bring the return series back in from the original dailyData.
    """
    cols = ["date", "ric", "pe_raw", "me_eur", RETURN_COL]
    # Guard: RETURN_COL may already be named ret_eur in dailyData
    available = [c for c in cols if c in dailyData.columns]
    pe_me = dailyData[available].copy()
    pe_me["date"] = pd.to_datetime(pe_me["date"])
    pe_me = pe_me.rename(columns={"pe_raw": PE_COL})

    merged = dailySAT.merge(pe_me, on=["date", "ric"], how="left")
    return merged


# ── private: daily RF ─────────────────────────────────────────────────────────

def _addDailyRF(
    panel: pd.DataFrame,
    monthlyRF: pd.DataFrame,
) -> pd.DataFrame:
    """
    Approximate daily RF = monthly_rf / trading_days_in_month.
    Standard in daily asset pricing; approximation error << daily return noise.
    """
    panel = panel.copy()
    panel["year_month"] = pd.to_datetime(panel["date"]).dt.to_period("M")

    # Count actual trading days per month in our panel
    trading_days = (
        panel.groupby("year_month")["date"]
        .nunique()
        .reset_index()
        .rename(columns={"date": "trading_days_in_month"})
    )

    rf = monthlyRF.copy()
    if not isinstance(rf["year_month"].dtype, pd.PeriodDtype):
        rf["year_month"] = pd.PeriodIndex(rf["year_month"], freq="M")

    rf = rf.merge(trading_days, on="year_month", how="left")
    rf["rf_daily"] = rf[RISK_FREE_COL] / rf["trading_days_in_month"].clip(lower=1)

    panel = panel.merge(rf[["year_month", "rf_daily"]], on="year_month", how="left")
    panel = panel.drop(columns=["year_month"])
    return panel


# ── private: daily quintiles ──────────────────────────────────────────────────

def _assignDailyQuintiles(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Assign PE and SAT quintiles cross-sectionally within each trading day.
    Days with fewer than _MIN_STOCKS_PER_DAY eligible stocks are dropped.
    """
    def _qrank(s: pd.Series) -> pd.Series:
        valid = s.dropna()
        if valid.nunique() < N_QUINTILES or len(valid) < _MIN_STOCKS_PER_DAY:
            return pd.Series(np.nan, index=s.index)
        try:
            return pd.qcut(s, q=N_QUINTILES, labels=False, duplicates="drop") + 1
        except ValueError:
            return pd.Series(np.nan, index=s.index)

    # PE quintile — winsorise PE cross-sectionally first
    panel[PE_COL] = (
        panel.groupby("date")[PE_COL]
        .transform(lambda s: s.clip(
            lower=s.quantile(0.01), upper=s.quantile(0.99)
        ))
    )
    # Exclude negative/zero PE (undefined trailing PE)
    panel.loc[panel[PE_COL] <= 0, PE_COL] = np.nan

    panel[PE_QUINTILE_COL]  = panel.groupby("date")[PE_COL].transform(_qrank)
    panel[SAT_RAW_Q_COL]    = panel.groupby("date")[SAT_RAW_COL].transform(_qrank)
    panel[SAT_5D_Q_COL]     = panel.groupby("date")[SAT_5D_COL].transform(_qrank)
    panel[SAT_10D_Q_COL]    = panel.groupby("date")[SAT_10D_COL].transform(_qrank)

    return panel


# ── private: forward returns ──────────────────────────────────────────────────

def _addForwardReturns(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add excess_ret_fwd = r_{i,d+1} - rf_{d+1}.
    Sort variables (SAT, PE) are from day d; return is from day d+1.
    Forward return is winsorised cross-sectionally at 0.5/99.5 pct per day
    to reduce the influence of extreme daily return observations (halts,
    data errors, dividend ex-dates).
    """
    panel = panel.sort_values(["ric", "date"]).copy()

    panel["ret_fwd"]    = panel.groupby("ric")[RETURN_COL].shift(-1)
    panel["rf_fwd"]     = panel.groupby("ric")["rf_daily"].shift(-1)
    panel["excess_ret_fwd"] = panel["ret_fwd"] - panel["rf_fwd"]

    # Winsorise at 0.5/99.5 — tighter than monthly because daily extremes are
    # more often data artefacts (halts, ex-dividend, stale prices)
    panel["excess_ret_fwd"] = (
        panel.groupby("date")["excess_ret_fwd"]
        .transform(lambda s: s.clip(
            lower=s.quantile(0.005), upper=s.quantile(0.995)
        ))
    )

    panel = panel.drop(columns=["ret_fwd", "rf_fwd"])
    return panel