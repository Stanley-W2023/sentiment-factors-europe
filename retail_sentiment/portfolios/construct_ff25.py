"""
construct_ff25.py — Build the 5×5 Fama-French double-sort portfolios.

At each month-end, stocks are independently sorted into quintiles by PE
and SAT. The 25 intersection portfolios are value-weighted (primary) or
equal-weighted (robustness). Returns are computed over month t+1.

Labels: portfolio (k, s) has PE quintile k ∈ {1..5} and SAT quintile
s ∈ {1..5}. Q1 SAT = most bearish; Q5 SAT = most bullish.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config.constants import (
    N_QUINTILES,
    RETURN_COL,
    MARKET_CAP_COL,
    RISK_FREE_COL,
    PE_QUINTILE_COL,
    SAT_QUINTILE_COL,
    NEWEY_WEST_LAGS,
)

_ROLLING_DELTA_CROSS_WINDOW: int = 36  # months


def constructFF25Portfolios(monthlyPanel: pd.DataFrame) -> pd.DataFrame:
    """
    Construct 25 value-weighted portfolios from the monthly panel.

    Args:
        monthlyPanel: Panel with columns
                      [year_month, ric, ret_eur, me_eur, rf_eur,
                       pe_quintile, sat_quintile].

    Returns:
        Long-format DataFrame with columns
        [year_month, pe_quintile, sat_quintile, ret_vw, excess_ret, n_stocks].
    """
    _validatePortfolioInputs(monthlyPanel)
    results = (
        monthlyPanel
        .groupby(["year_month", PE_QUINTILE_COL, SAT_QUINTILE_COL],
                 group_keys=False)
        .apply(_computeOnePortfolioReturn, include_groups=False)
        .reset_index()
    )
    results["excess_ret"] = results["ret_vw"] - results[RISK_FREE_COL]
    return results.drop(columns=[RISK_FREE_COL])


def constructFF25PortfoliosEW(monthlyPanel: pd.DataFrame) -> pd.DataFrame:
    """
    Construct 25 equal-weighted portfolios — robustness check.
    Retail effects are strongest in small/mid caps; EW gives them equal
    weight relative to mega-caps that dominate value-weighting.

    Returns same schema as constructFF25Portfolios with column ret_ew.
    """
    _validatePortfolioInputs(monthlyPanel)
    results = (
        monthlyPanel
        .groupby(["year_month", PE_QUINTILE_COL, SAT_QUINTILE_COL],
                 group_keys=False)
        .apply(_computeOnePortfolioReturnEW, include_groups=False)
        .reset_index()
    )
    results["excess_ret"] = results["ret_ew"] - results[RISK_FREE_COL]
    return results.drop(columns=[RISK_FREE_COL])


def filterSmallMidCap(
    monthlyPanel: pd.DataFrame,
    exclude_top_pct: float = 0.30,
) -> pd.DataFrame:
    """
    Remove the top exclude_top_pct of stocks by market cap each month.
    Retail participation is disproportionately concentrated in smaller names;
    this subsample robustness test should sharpen all spreads.

    Args:
        monthlyPanel   : Full monthly panel.
        exclude_top_pct: Fraction of top market-cap stocks to drop (default 30%).

    Returns:
        Filtered panel with the same schema.
    """
    def _dropTopN(group: pd.DataFrame) -> pd.DataFrame:
        threshold = group[MARKET_CAP_COL].quantile(1 - exclude_top_pct)
        return group[group[MARKET_CAP_COL] <= threshold]

    filtered = (
        monthlyPanel
        .groupby("year_month", group_keys=False)
        .apply(_dropTopN)
        .reset_index(drop=True)
    )
    n_dropped = len(monthlyPanel) - len(filtered)
    print(f"  [small/mid-cap] Dropped {n_dropped:,} large-cap stock-months "
          f"(top {exclude_top_pct*100:.0f}% by market cap)")
    return filtered


def computeSpreads(portfolioReturns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the three primary spread statistics with Newey-West t-statistics.

    For each spread, computes the time series of monthly values and reports:
      mean, NW t-stat, NW p-value.

    Returns DataFrame with columns:
      [stat, mean, t_stat, p_value].
    """
    _validateSpreadInputs(portfolioReturns)

    # Build monthly time series of each spread
    monthly_ts = _buildMonthlySpreadTimeSeries(portfolioReturns)

    records = []
    for stat_col in monthly_ts.columns:
        series = monthly_ts[stat_col].dropna().values
        n = len(series)
        if n < 2:
            records.append({"stat": stat_col, "mean": np.nan,
                            "t_stat": np.nan, "p_value": np.nan})
            continue
        model = sm.OLS(series, np.ones(n)).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": NEWEY_WEST_LAGS},
        )
        records.append({
            "stat":    stat_col,
            "mean":    float(model.params[0]),
            "t_stat":  float(model.tvalues[0]),
            "p_value": float(model.pvalues[0]),
        })

    return pd.DataFrame(records)


def computeRollingDeltaCross(
    portfolioReturns: pd.DataFrame,
    window: int = _ROLLING_DELTA_CROSS_WINDOW,
) -> pd.DataFrame:
    """
    Compute rolling delta_Cross over a trailing window of months.

    delta_Cross_t = [r_{5,5,t} - r_{1,5,t}] - [r_{5,1,t} - r_{1,1,t}]

    Returns DataFrame with columns [year_month, delta_cross_rolling].
    A visible jump around early 2020 supports Hypothesis 4.
    """
    _validateSpreadInputs(portfolioReturns)

    monthly_ts = _buildMonthlySpreadTimeSeries(portfolioReturns)
    dc = monthly_ts[["delta_cross"]].copy()
    dc["delta_cross_rolling"] = (
        dc["delta_cross"]
        .rolling(window=window, min_periods=window // 2)
        .mean()
    )
    dc = dc.reset_index()
    return dc[["year_month", "delta_cross_rolling"]].dropna()


def buildRSFLongShort(portfolioReturns: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the RSF long-short portfolio.
    Long: (PE=5, SAT=5). Short: (PE=1, SAT=1).
    Returns monthly time series [year_month, rsf_return].
    """
    long_leg = portfolioReturns[
        (portfolioReturns[PE_QUINTILE_COL] == N_QUINTILES) &
        (portfolioReturns[SAT_QUINTILE_COL] == N_QUINTILES)
    ].set_index("year_month")["excess_ret"]

    short_leg = portfolioReturns[
        (portfolioReturns[PE_QUINTILE_COL] == 1) &
        (portfolioReturns[SAT_QUINTILE_COL] == 1)
    ].set_index("year_month")["excess_ret"]

    rsf = (long_leg - short_leg).rename("rsf_return").reset_index()
    return rsf


# ── private ───────────────────────────────────────────────────────────────────

def _buildMonthlySpreadTimeSeries(
    portfolioReturns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a time series DataFrame where each column is a monthly spread.
    Index: year_month. Columns: delta_pe_s1..s5, delta_sat_k1..k5, delta_cross.
    """
    records = {}

    for month, grp in portfolioReturns.groupby("year_month"):
        ret = (
            grp.groupby([PE_QUINTILE_COL, SAT_QUINTILE_COL])["excess_ret"]
            .mean()
            .unstack(SAT_QUINTILE_COL)
        )
        row = {}
        for s in range(1, N_QUINTILES + 1):
            if s in ret.columns and 5 in ret.index and 1 in ret.index:
                row[f"delta_pe_s{s}"] = ret.loc[5, s] - ret.loc[1, s]
        for k in range(1, N_QUINTILES + 1):
            if k in ret.index and 5 in ret.columns and 1 in ret.columns:
                row[f"delta_sat_k{k}"] = ret.loc[k, 5] - ret.loc[k, 1]
        if all(x in ret.index for x in [1, 5]) and all(x in ret.columns for x in [1, 5]):
            row["delta_cross"] = (
                (ret.loc[5, 5] - ret.loc[1, 5]) - (ret.loc[5, 1] - ret.loc[1, 1])
            )
        records[month] = row

    ts = pd.DataFrame(records).T
    ts.index.name = "year_month"
    return ts


def _computeOnePortfolioReturn(group: pd.DataFrame) -> pd.Series:
    """Value-weighted return for one (month, PE quintile, SAT quintile) cell."""
    total_me = group[MARKET_CAP_COL].sum()
    if total_me <= 0 or group.empty:
        return pd.Series({
            "ret_vw": np.nan,
            RISK_FREE_COL: group[RISK_FREE_COL].iloc[0] if not group.empty else np.nan,
            "n_stocks": 0,
        })
    weights = group[MARKET_CAP_COL] / total_me
    return pd.Series({
        "ret_vw":      float((weights * group[RETURN_COL]).sum()),
        RISK_FREE_COL: float(group[RISK_FREE_COL].iloc[0]),
        "n_stocks":    len(group),
    })


def _computeOnePortfolioReturnEW(group: pd.DataFrame) -> pd.Series:
    """Equal-weighted return for one (month, PE quintile, SAT quintile) cell."""
    if group.empty:
        return pd.Series({
            "ret_ew": np.nan,
            RISK_FREE_COL: np.nan,
            "n_stocks": 0,
        })
    return pd.Series({
        "ret_ew":      float(group[RETURN_COL].mean()),
        RISK_FREE_COL: float(group[RISK_FREE_COL].iloc[0]),
        "n_stocks":    len(group),
    })


def _validatePortfolioInputs(panel: pd.DataFrame):
    required = {
        "year_month", "ric", RETURN_COL, MARKET_CAP_COL,
        RISK_FREE_COL, PE_QUINTILE_COL, SAT_QUINTILE_COL,
    }
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Missing required columns for portfolio construction: {missing}")


def _validateSpreadInputs(portfolioReturns: pd.DataFrame):
    required = {PE_QUINTILE_COL, SAT_QUINTILE_COL, "excess_ret"}
    missing = required - set(portfolioReturns.columns)
    if missing:
        raise ValueError(f"Missing required columns for spread computation: {missing}")