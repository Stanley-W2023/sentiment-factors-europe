"""
time_series_alpha.py — FF6 time-series alpha estimation for all 25 portfolios.

For each (PE quintile, SAT quintile) portfolio, estimates:

  r_{k,s,t} - rf_t = alpha_{k,s}
                     + beta_mkt  * MktRF_t
                     + beta_smb  * SMB_t
                     + beta_hml  * HML_t
                     + beta_rmw  * RMW_t
                     + beta_cma  * CMA_t
                     + beta_umd  * UMD_t
                     + epsilon_t

using OLS with Newey-West standard errors (NEWEY_WEST_LAGS lags).
Requires at least MIN_OBS_FOR_REGRESSION monthly observations.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config.constants import (
    FF_FACTORS,
    NEWEY_WEST_LAGS,
    MIN_OBS_FOR_REGRESSION,
    ESG_QUINTILE_COL,
    SAT_QUINTILE_COL,
)


def estimateTimeSeriesAlpha(
    portfolioReturns: pd.DataFrame,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate FF6 alpha for each of the 25 portfolios.

    Args:
        portfolioReturns: Output of constructFF25Portfolios().
                          Columns: [year_month, esg_quintile, sat_quintile, excess_ret].
        factors: Monthly factor returns with columns matching FF_FACTORS,
                 indexed or with a year_month column.

    Returns:
        DataFrame with one row per portfolio, columns:
        [esg_quintile, sat_quintile, alpha, t_stat, p_value,
         beta_mkt, beta_smb, beta_hml, beta_rmw, beta_cma, beta_umd,
         r_squared, n_obs].
    """
    _validateAlphaInputs(portfolioReturns, factors)

    factors_aligned = _alignFactors(factors)
    results = []

    for (k, s), group in portfolioReturns.groupby(
        [ESG_QUINTILE_COL, SAT_QUINTILE_COL]
    ):
        result = _estimateSinglePortfolioAlpha(group, factors_aligned, k, s)
        results.append(result)

    return pd.DataFrame(results).sort_values(
        [ESG_QUINTILE_COL, SAT_QUINTILE_COL]
    ).reset_index(drop=True)


def buildAlphaMatrix(alphaResults: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot alpha estimates into a 5×5 matrix (PE quintile × SAT quintile).
    Useful for producing the paper's Table 2 Panel B.
    """
    return alphaResults.pivot(
        index=ESG_QUINTILE_COL,
        columns=SAT_QUINTILE_COL,
        values="alpha",
    )


def buildTStatMatrix(alphaResults: pd.DataFrame) -> pd.DataFrame:
    """Pivot t-statistics into a 5×5 matrix."""
    return alphaResults.pivot(
        index=ESG_QUINTILE_COL,
        columns=SAT_QUINTILE_COL,
        values="t_stat",
    )


# ── private ───────────────────────────────────────────────────────────────────

def _estimateSinglePortfolioAlpha(
    portfolioData: pd.DataFrame,
    factors: pd.DataFrame,
    esg_quintile: int,
    sat_quintile: int,
) -> dict:
    """Estimate FF6 alpha for a single (PE, SAT) portfolio."""
    merged = portfolioData.merge(factors, on="year_month", how="inner")

    if len(merged) < MIN_OBS_FOR_REGRESSION:
        return _emptyAlphaResult(esg_quintile, sat_quintile)

    y = merged["excess_ret"].values
    X = sm.add_constant(merged[FF_FACTORS].values)

    try:
        model = sm.OLS(y, X).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": NEWEY_WEST_LAGS},
        )
        return {
            ESG_QUINTILE_COL: esg_quintile,
            SAT_QUINTILE_COL: sat_quintile,
            "alpha": float(model.params[0]),
            "t_stat": float(model.tvalues[0]),
            "p_value": float(model.pvalues[0]),
            **{f"beta_{f}": float(model.params[i + 1])
               for i, f in enumerate(FF_FACTORS)},
            "r_squared": float(model.rsquared),
            "n_obs": int(model.nobs),
        }
    except Exception as e:
        print(f"  Alpha estimation failed for ({esg_quintile},{sat_quintile}): {e}")
        return _emptyAlphaResult(esg_quintile, sat_quintile)


def _emptyAlphaResult(esg_quintile: int, sat_quintile: int) -> dict:
    """Return a NaN-filled result row for insufficient data cases."""
    result = {
        ESG_QUINTILE_COL: esg_quintile,
        SAT_QUINTILE_COL: sat_quintile,
        "alpha": np.nan,
        "t_stat": np.nan,
        "p_value": np.nan,
        "r_squared": np.nan,
        "n_obs": 0,
    }
    result.update({f"beta_{f}": np.nan for f in FF_FACTORS})
    return result


def _alignFactors(factors: pd.DataFrame) -> pd.DataFrame:
    """Ensure factors have a year_month column in Period dtype."""
    factors = factors.copy()
    if "year_month" not in factors.columns:
        if isinstance(factors.index, pd.DatetimeIndex):
            factors["year_month"] = factors.index.to_period("M")
        else:
            raise ValueError(
                "factors must have a year_month column or DatetimeIndex"
            )
    # year_month may already be Period (from _stepRunEstimation), a string,
    # or a datetime — handle all three without calling .dt.to_period() on
    # a PeriodArray (which raises AttributeError in pandas >= 2.0).
    ym = factors["year_month"]
    if not isinstance(ym.dtype, pd.PeriodDtype):
        factors["year_month"] = pd.PeriodIndex(ym, freq="M")
    return factors


def _validateAlphaInputs(
    portfolioReturns: pd.DataFrame,
    factors: pd.DataFrame,
):
    required_ports = {"year_month", ESG_QUINTILE_COL, SAT_QUINTILE_COL, "excess_ret"}
    missing_ports = required_ports - set(portfolioReturns.columns)
    if missing_ports:
        raise ValueError(f"portfolioReturns missing columns: {missing_ports}")

    missing_factors = set(FF_FACTORS) - set(factors.columns)
    if missing_factors:
        raise ValueError(f"factors DataFrame missing columns: {missing_factors}")