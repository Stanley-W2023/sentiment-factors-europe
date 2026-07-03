"""
construct_daily_ff25.py — Daily 5×5 double-sort portfolios for horizon tests.

Mirrors construct_ff25.py exactly but operates at daily frequency.
For each SAT variant (raw, 5d, 10d) we produce:
  - 25 value-weighted portfolio returns (one per trading day)
  - Spread statistics: delta_SAT(k), delta_PE(s), delta_Cross
  - RSF long-short daily time series
  - Rolling 60-day delta_Cross (≈ 3 calendar months; analogue of 36-month
    rolling window in the monthly analysis)

Key design decisions:
  - Portfolio formation date d, return measured on d+1 (forward return)
    consistent with the monthly convention (sort at t, return at t+1)
  - Minimum 5 stocks per cell per day to compute a valid portfolio return;
    cells below this threshold return NaN rather than a noisy single-stock
    return
  - Newey-West lags = 22 (≈ one trading month) for time-series mean inference
    on daily lambda series — substantially more conservative than the 6-lag
    monthly default because daily lambdas have much higher autocorrelation
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from features.build_daily_panel import (
    SAT_RAW_Q_COL,
    SAT_5D_Q_COL,
    SAT_10D_Q_COL,
)
from config.constants import (
    N_QUINTILES,
    RETURN_COL,
    MARKET_CAP_COL,
    PE_QUINTILE_COL,
)

# ── Constants ─────────────────────────────────────────────────────────────────
DAILY_NW_LAGS: int    = 22   # ≈ 1 trading month
_MIN_STOCKS_PER_CELL: int = 5
_ROLLING_DC_WINDOW: int   = 60  # trading days ≈ 3 calendar months

# Map human-readable variant names to their quintile column names
SAT_VARIANTS: dict[str, str] = {
    "raw": SAT_RAW_Q_COL,
    "5d":  SAT_5D_Q_COL,
    "10d": SAT_10D_Q_COL,
}


def constructAllDailyPortfolios(
    dailyPanel: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Build 25 VW portfolios for each SAT variant.

    Args:
        dailyPanel: Output of buildDailyAnalysisPanel. Must contain
                    [date, ric, ret_eur, me_eur, excess_ret_fwd,
                     pe_quintile, sat_raw_quintile, sat_5d_quintile,
                     sat_10d_quintile].

    Returns:
        Dict keyed by variant name ("raw", "5d", "10d"). Each value is a
        long-format DataFrame:
          [date, pe_quintile, sat_quintile, ret_vw, excess_ret_fwd_vw,
           n_stocks].
    """
    results = {}
    for variant, sat_q_col in SAT_VARIANTS.items():
        print(f"  [daily portfolios] Building {variant} variant...")
        results[variant] = _buildPortfoliosOneVariant(dailyPanel, sat_q_col)
    return results


def computeDailySpreads(
    portfolios: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Compute spread statistics for all three SAT variants.

    Returns:
        DataFrame with columns:
          [variant, stat, mean, t_stat, p_value, n_days]
        stat values: delta_sat_k1..k5, delta_pe_s1..s5, delta_cross
    """
    records = []
    for variant, port_df in portfolios.items():
        spreads = _computeSpreadStatsOneVariant(port_df)
        spreads["variant"] = variant
        records.append(spreads)
    return pd.concat(records, ignore_index=True)


def buildDailyRSFLongShort(
    portfolios: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Construct daily RSF L/S for each SAT variant.
    Long = PE Q5 (avg over SAT), Short = PE Q1 (avg over SAT).

    Returns:
        DataFrame [date, rsf_raw, rsf_5d, rsf_10d].
    """
    ls_series = {}
    for variant, port_df in portfolios.items():
        long_leg  = (
            port_df[port_df[PE_QUINTILE_COL] == N_QUINTILES]
            .groupby("date")["excess_ret_fwd_vw"].mean()
        )
        short_leg = (
            port_df[port_df[PE_QUINTILE_COL] == 1]
            .groupby("date")["excess_ret_fwd_vw"].mean()
        )
        ls_series[f"rsf_{variant}"] = (long_leg - short_leg)

    result = pd.DataFrame(ls_series).reset_index()
    result = result.rename(columns={"index": "date"})
    return result


def computeRollingDailyDeltaCross(
    portfolios: dict[str, pd.DataFrame],
    window: int = _ROLLING_DC_WINDOW,
) -> pd.DataFrame:
    """
    Rolling delta_Cross over `window` trading days for each SAT variant.

    Returns:
        DataFrame [date, delta_cross_rolling_raw, delta_cross_rolling_5d,
                   delta_cross_rolling_10d].
    """
    rolling = {}
    for variant, port_df in portfolios.items():
        dc = _buildDailySpreadTimeSeries(port_df)[["delta_cross"]].copy()
        dc[f"delta_cross_rolling_{variant}"] = (
            dc["delta_cross"]
            .rolling(window=window, min_periods=window // 2)
            .mean()
        )
        rolling[variant] = dc[[f"delta_cross_rolling_{variant}"]]

    result = pd.concat(rolling.values(), axis=1).reset_index()
    result = result.rename(columns={"index": "date"})
    return result.dropna(subset=[c for c in result.columns if c != "date"],
                         how="all")


# ── private: portfolio construction ──────────────────────────────────────────

def _buildPortfoliosOneVariant(
    dailyPanel: pd.DataFrame,
    satQCol: str,
) -> pd.DataFrame:
    """
    Build 25 VW portfolios using PE_QUINTILE_COL and satQCol as sort keys.
    Formation: day d characteristics → return on day d+1 (excess_ret_fwd).
    """
    needed = ["date", "ric", PE_QUINTILE_COL, satQCol,
              MARKET_CAP_COL, "excess_ret_fwd"]
    sub = dailyPanel[needed].dropna(
        subset=[PE_QUINTILE_COL, satQCol, "excess_ret_fwd"]
    ).copy()

    sub = sub.rename(columns={satQCol: "sat_quintile"})
    sub["pe_quintile"]  = sub[PE_QUINTILE_COL].astype(int)
    sub["sat_quintile"] = sub["sat_quintile"].astype(int)

    results = (
        sub.groupby(["date", "pe_quintile", "sat_quintile"], group_keys=False)
        .apply(_vwReturnOneCell, include_groups=False)
        .reset_index()
    )
    return results


def _vwReturnOneCell(group: pd.DataFrame) -> pd.Series:
    """Value-weighted excess return for one (date, PE-q, SAT-q) cell."""
    if len(group) < _MIN_STOCKS_PER_CELL:
        return pd.Series({
            "excess_ret_fwd_vw": np.nan,
            "n_stocks": len(group),
        })
    total_me = group[MARKET_CAP_COL].sum()
    if total_me <= 0:
        return pd.Series({"excess_ret_fwd_vw": np.nan, "n_stocks": len(group)})
    weights = group[MARKET_CAP_COL] / total_me
    return pd.Series({
        "excess_ret_fwd_vw": float((weights * group["excess_ret_fwd"]).sum()),
        "n_stocks":          len(group),
    })


# ── private: spreads ──────────────────────────────────────────────────────────

def _computeSpreadStatsOneVariant(portDf: pd.DataFrame) -> pd.DataFrame:
    """Compute NW(22) mean and t-stat for each spread in one variant."""
    ts = _buildDailySpreadTimeSeries(portDf)
    records = []
    for col in ts.columns:
        series = ts[col].dropna().values
        n = len(series)
        if n < DAILY_NW_LAGS + 2:
            records.append({"stat": col, "mean": np.nan,
                            "t_stat": np.nan, "p_value": np.nan, "n_days": n})
            continue
        model = sm.OLS(series, np.ones(n)).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": DAILY_NW_LAGS},
        )
        records.append({
            "stat":    col,
            "mean":    float(model.params[0]),
            "t_stat":  float(model.tvalues[0]),
            "p_value": float(model.pvalues[0]),
            "n_days":  n,
        })
    return pd.DataFrame(records)


def _buildDailySpreadTimeSeries(portDf: pd.DataFrame) -> pd.DataFrame:
    """
    Build daily time series of all spread statistics.
    Mirrors _buildMonthlySpreadTimeSeries in construct_ff25.py.
    """
    records = {}
    for day, grp in portDf.groupby("date"):
        ret = (
            grp.groupby(["pe_quintile", "sat_quintile"])["excess_ret_fwd_vw"]
            .mean()
            .unstack("sat_quintile")
        )
        row = {}
        for s in range(1, N_QUINTILES + 1):
            if s in ret.columns and 5 in ret.index and 1 in ret.index:
                row[f"delta_pe_s{s}"] = ret.loc[5, s] - ret.loc[1, s]
        for k in range(1, N_QUINTILES + 1):
            if k in ret.index and 5 in ret.columns and 1 in ret.columns:
                row[f"delta_sat_k{k}"] = ret.loc[k, 5] - ret.loc[k, 1]
        if all(x in ret.index for x in [1, 5]) and \
           all(x in ret.columns for x in [1, 5]):
            row["delta_cross"] = (
                (ret.loc[5, 5] - ret.loc[1, 5])
                - (ret.loc[5, 1] - ret.loc[1, 1])
            )
        records[day] = row

    ts = pd.DataFrame(records).T
    ts.index.name = "date"
    return ts


def printDailySpreadSummary(spreads: pd.DataFrame):
    """Print a formatted comparison table across the three SAT variants."""
    print("\n  [daily spreads] Summary across SAT variants")
    print(f"  {'Stat':<20} {'raw mean':>10} {'raw t':>8} "
          f"{'5d mean':>10} {'5d t':>8} {'10d mean':>10} {'10d t':>8}")
    print("  " + "-" * 80)

    key_stats = (
        [f"delta_sat_k{k}" for k in range(1, 6)]
        + [f"delta_pe_s{s}" for s in range(1, 6)]
        + ["delta_cross"]
    )
    for stat in key_stats:
        row = {"stat": stat}
        for variant in ["raw", "5d", "10d"]:
            sub = spreads[(spreads["variant"] == variant) &
                          (spreads["stat"] == stat)]
            if sub.empty:
                row[f"{variant}_mean"] = np.nan
                row[f"{variant}_t"]    = np.nan
            else:
                row[f"{variant}_mean"] = sub["mean"].iloc[0]
                row[f"{variant}_t"]    = sub["t_stat"].iloc[0]

        def _fmt_t(t):
            if np.isnan(t):
                return "    n/a"
            stars = "***" if abs(t) > 3 else "**" if abs(t) > 2.5 \
                    else "*" if abs(t) > 2 else ""
            return f"{t:7.2f}{stars}"

        def _fmt_m(m):
            return "     n/a" if np.isnan(m) else f"{m*100:8.4f}%"

        print(f"  {stat:<20} "
              f"{_fmt_m(row['raw_mean'])} {_fmt_t(row['raw_t'])}  "
              f"{_fmt_m(row['5d_mean'])} {_fmt_t(row['5d_t'])}  "
              f"{_fmt_m(row['10d_mean'])} {_fmt_t(row['10d_t'])}")