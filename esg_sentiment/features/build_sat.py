"""
build_sat.py — Construct Signed Abnormal Turnover (SAT) for all STOXX constituents.

Pipeline:
  1. Compute daily turnover = Volume / SharesOutstanding.
  2. Fit rolling AR(TURNOVER_AR_ORDER) on log-turnover to extract residuals.
  3. Exponentiate residuals to get Abnormal Turnover (ATV).
  4. Multiply ATV by sign(return) to get daily SAT.
  5. Exclude days within EARNINGS_EXCLUSION_WINDOW_DAYS of earnings announcements.
  6. Aggregate to monthly SAT (mean of daily SAT, min 10 valid days).

Key identification assumption: short-selling friction (Baker & Stein 2004) means
upside volume spikes are disproportionately retail-driven. Signing by return direction
separates retail buying enthusiasm (positive SAT) from retail panic (negative SAT).
"""

import warnings
import numpy as np
import pandas as pd
from config.constants import (
    TURNOVER_AR_ORDER,
    TURNOVER_ROLLING_WINDOW_DAYS,
    MIN_VALID_DAYS_PER_MONTH,
    EARNINGS_EXCLUSION_WINDOW_DAYS,
    MIN_TURNOVER_CLIP,
    SAT_COL,
)

_SAT_DAILY_COL = "sat_daily"


def buildMonthlySAT(
    dailyData: pd.DataFrame,
    earningsCalendar: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build monthly SAT for all stocks in the panel.

    Args:
        dailyData: Long-format panel with columns:
                   [date, ric, volume, shares_outstanding, ret_eur].
        earningsCalendar: DataFrame with columns [ric, announcement_date].

    Returns:
        Monthly panel with columns [year_month, ric, sat_monthly].
    """
    _validateDailyInputs(dailyData)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        daily_sat = (
            dailyData
            .groupby("ric", group_keys=False)
            .apply(_buildDailySATForOneFirm)
        )

    daily_sat_clean = _excludeEarningsWindows(daily_sat, earningsCalendar)

    return _aggregateDailyToMonthly(daily_sat_clean)


def buildDailySATPanel(dailyData: pd.DataFrame) -> pd.DataFrame:
    """
    Build daily SAT panel without earnings exclusion.
    Useful for horizon-sensitivity robustness tests.

    Args:
        dailyData: same format as buildMonthlySAT.

    Returns:
        Daily panel with columns [date, ric, sat_daily, atv, turnover].
    """
    _validateDailyInputs(dailyData)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return (
            dailyData
            .groupby("ric", group_keys=False)
            .apply(_buildDailySATForOneFirm)
            [["date", "ric", _SAT_DAILY_COL, "atv", "turnover"]]
        )



def _buildDailySATForOneFirm(firmData: pd.DataFrame) -> pd.DataFrame:
    """Compute daily SAT series for a single firm."""
    firmData = firmData.sort_values("date").copy()

    firmData["turnover"] = (
        firmData["volume"] / firmData["shares_outstanding"]
    )
    firmData["log_turnover"] = np.log(
        firmData["turnover"].clip(lower=MIN_TURNOVER_CLIP)
    )

    firmData["atv"] = _computeAbnormalTurnover(firmData["log_turnover"])

    firmData["ret_sign"] = np.sign(firmData["ret_eur"])
    firmData.loc[firmData["ret_sign"] == 0, "atv"] = np.nan

    firmData[_SAT_DAILY_COL] = firmData["atv"] * firmData["ret_sign"]

    return firmData


def _computeAbnormalTurnover(logTurnover: pd.Series) -> pd.Series:
    """
    Fit rolling AR(TURNOVER_AR_ORDER) on log-turnover and return
    out-of-sample residuals as ATV = exp(residual) - 1.

    Replaces statsmodels AutoReg with a direct numpy least-squares solve.
    AR(p) is just OLS: design matrix is [1, y_{t-1}, ..., y_{t-p}].

    The daily panel only contains trading days (weekends and bank holidays
    are already excluded upstream), so consecutive rows in the window are
    genuinely consecutive in time. dropna() is therefore safe — any NaN
    in log_turnover reflects a genuine bad data point and that observation
    should simply be excluded from the training set.

    Windows with more than 10% missing values are skipped entirely.
    Near-singular systems (rank-deficient X) are skipped via rank check.
    Prediction is skipped if the final p lags contain any NaN.
    """
    residuals = pd.Series(np.nan, index=logTurnover.index)
    n = len(logTurnover)
    p = TURNOVER_AR_ORDER
    max_missing_fraction = 0.10

    for end_idx in range(TURNOVER_ROLLING_WINDOW_DAYS, n):
        start_idx = end_idx - TURNOVER_ROLLING_WINDOW_DAYS
        window = logTurnover.iloc[start_idx:end_idx]

        missing_fraction = window.isna().mean()
        if missing_fraction > max_missing_fraction:
            continue

        try:
            clean = window.dropna().values
            if len(clean) < p + 1:
                continue

            X = np.column_stack([
                np.ones(len(clean) - p),
                *[clean[p - lag - 1 : len(clean) - lag - 1] for lag in range(p)]
            ])
            y = clean[p:]

            coeffs, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)

            if rank < X.shape[1]:
                continue

            last_lags = np.array([1.0] + [clean[-k] for k in range(1, p + 1)])
            if np.any(np.isnan(last_lags)):
                continue

            predicted = coeffs @ last_lags
            observed = logTurnover.iloc[end_idx]
            residuals.iloc[end_idx] = np.exp(observed - predicted) - 1

        except Exception:
            continue

    return residuals


def _excludeEarningsWindows(
    dailySAT: pd.DataFrame,
    earningsCalendar: pd.DataFrame,
) -> pd.DataFrame:
    """
    Set sat_daily to NaN within EARNINGS_EXCLUSION_WINDOW_DAYS of
    each earnings announcement date, to avoid confounding SAT with
    earnings surprise volume.
    """
    if earningsCalendar.empty:
        return dailySAT

    dailySAT = dailySAT.copy()
    window = pd.Timedelta(days=EARNINGS_EXCLUSION_WINDOW_DAYS)

    for _, row in earningsCalendar.iterrows():
        announcement = pd.Timestamp(row["announcement_date"])
        mask = (
            (dailySAT["ric"] == row["ric"])
            & (dailySAT["date"] >= announcement - window)
            & (dailySAT["date"] <= announcement + window)
        )
        dailySAT.loc[mask, _SAT_DAILY_COL] = np.nan

    return dailySAT


_SAT_WINSOR_LOWER: float = 0.01
_SAT_WINSOR_UPPER: float = 0.99


def _aggregateDailyToMonthly(dailySAT: pd.DataFrame) -> pd.DataFrame:
    """
    Average daily SAT within each (ric, month), then apply two cross-
    sectional normalisation steps within each calendar month:

      1. Winsorise at 1%/99% — prevents single ATV spikes (from the
         exp(residual) transformation) from dominating the distribution.
         Raw ATV = exp(log-turnover residual) - 1 can reach 1000+ for
         large turnover anomalies; std ~ 984 without this step.

      2. Z-score cross-sectionally — makes SAT comparable across
         liquidity regimes and removes scale distortions from high-
         volume names. Required for FMB regressors to be on the same
         scale across months.

    Months with fewer than MIN_VALID_DAYS_PER_MONTH non-NaN days return NaN.
    """
    dailySAT = dailySAT.copy()
    dailySAT["year_month"] = pd.to_datetime(dailySAT["date"]).dt.to_period("M")

    def _safeMean(series: pd.Series) -> float:
        valid = series.dropna()
        if len(valid) < MIN_VALID_DAYS_PER_MONTH:
            return np.nan
        return float(valid.mean())

    monthly = (
        dailySAT
        .groupby(["ric", "year_month"])[_SAT_DAILY_COL]
        .agg(_safeMean)
        .rename(SAT_COL)
        .reset_index()
    )

    # Step 1: cross-sectional winsorisation within each month
    def _winsorize(s: pd.Series) -> pd.Series:
        lo = s.quantile(_SAT_WINSOR_LOWER)
        hi = s.quantile(_SAT_WINSOR_UPPER)
        return s.clip(lower=lo, upper=hi)

    monthly[SAT_COL] = (
        monthly.groupby("year_month")[SAT_COL]
        .transform(_winsorize)
    )

    # Step 2: cross-sectional z-score within each month
    def _zscore(s: pd.Series) -> pd.Series:
        mu  = s.mean()
        std = s.std()
        if std < 1e-10:
            return s - mu
        return (s - mu) / std

    monthly[SAT_COL] = (
        monthly.groupby("year_month")[SAT_COL]
        .transform(_zscore)
    )

    return monthly


def _validateDailyInputs(dailyData: pd.DataFrame):
    """Early return guard: raise if required columns are missing."""
    required = {"date", "ric", "volume", "shares_outstanding", "ret_eur"}
    missing = required - set(dailyData.columns)
    if missing:
        raise ValueError(f"Missing required columns in dailyData: {missing}")