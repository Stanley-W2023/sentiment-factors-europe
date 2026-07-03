"""
fama_macbeth.py — Fama-MacBeth cross-sectional regressions for the RSF.

Estimates the monthly cross-sectional regression:

  r_{i,t+1} - rf_t = lambda_0
                     + lambda_PE  * PEQ_{i,t}
                     + lambda_SAT * SATQ_{i,t}
                     + lambda_Int * (PEQ_{i,t} × SATQ_{i,t})
                     + epsilon_{i,t+1}

where PEQ and SATQ are quintile ranks (1-5). Quintile ranks are used
rather than continuous SAT because the raw SAT series contains extreme
outliers from the exp(AR-residual) transformation (std ≈ 984 vs p99 ≈ 0.8).
Rank-based regressors are robust to these outliers, and are fully
consistent with the portfolio sort in Step 4 which also uses quintile ranks.

lambda_Int is the primary test statistic. A positive, significant
lambda_Int confirms that the interaction between PE valuation and retail
sentiment predicts returns, consistent with Hypothesis 1.

Time-series means of monthly coefficients are reported with
Newey-West standard errors (NEWEY_WEST_LAGS lags).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from features.forward_returns import (
    addForwardMonthlyReturns,
    FWD_RETURN_COL,
    FWD_RF_COL,
)
from config.constants import (
    NEWEY_WEST_LAGS,
    MIN_OBS_FOR_REGRESSION,
    PE_QUINTILE_COL,
    SAT_QUINTILE_COL,
    RETURN_COL,
    RISK_FREE_COL,
)

INTERACTION_COL = "pe_sat_interaction"

WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99


def estimateFamaMacBeth(
    stockPanel: pd.DataFrame,
    portfolioReturns: pd.DataFrame | None = None,  # kept for call-site compatibility
) -> pd.DataFrame:
    """
    Run Fama-MacBeth regressions using quintile-rank regressors.

    Args:
        stockPanel      : Monthly stock-level panel from _stepBuildFeatures.
        portfolioReturns: Unused — kept for call-site compatibility.

    Returns:
        DataFrame with one row per regressor:
        [variable, lambda_mean, t_stat, p_value, n_months].
    """
    _validateFMBInputs(stockPanel)

    # Forward return: sort on characteristics at t, predict returns at t+1.
    # addForwardMonthlyReturns enforces that t+1 is the immediately following
    # calendar month (no gap-jumping across delistings/suspensions).
    panel = addForwardMonthlyReturns(stockPanel)
    panel["excess_ret_fwd"] = panel[FWD_RETURN_COL] - panel[FWD_RF_COL]

    # Winsorise excess_ret_fwd cross-sectionally within each month to remove
    # extreme return outliers that would distort the cross-sectional regressions
    panel["excess_ret_fwd"] = (
        panel.groupby("year_month")["excess_ret_fwd"]
        .transform(_winsorize)
    )

    panel[INTERACTION_COL] = (
        pd.to_numeric(panel[PE_QUINTILE_COL], errors="coerce")
        * pd.to_numeric(panel[SAT_QUINTILE_COL], errors="coerce")
    )

    regressors = [PE_QUINTILE_COL, SAT_QUINTILE_COL, INTERACTION_COL]

    monthly_coefs = []
    for month, month_data in panel.groupby("year_month"):
        coefs = _runOneCrossSection(month_data, regressors)
        if coefs is not None:
            coefs["year_month"] = month
            monthly_coefs.append(coefs)

    if not monthly_coefs:
        raise RuntimeError("No valid cross-sections found for Fama-MacBeth")

    coef_panel = pd.DataFrame(monthly_coefs).set_index("year_month")
    return _computeTimeSeriesMeans(coef_panel, regressors + ["const"])


def computeChowTest(
    stockPanel: pd.DataFrame,
    regimeBreakDate: pd.Period,
    portfolioReturns: pd.DataFrame | None = None,  # kept for call-site compatibility
) -> pd.DataFrame:
    """
    Chow-type test on the time series of monthly lambda_Int estimates:

      lambda_Int_t = alpha + beta * Post2020_t + epsilon_t

    beta > 0 and significant supports Hypothesis 4 — the PE×SAT
    interaction strengthened in the post-2020 retail trading era.
    """
    _validateFMBInputs(stockPanel)

    panel = addForwardMonthlyReturns(stockPanel)
    panel["excess_ret_fwd"] = panel[FWD_RETURN_COL] - panel[FWD_RF_COL]
    panel["excess_ret_fwd"] = (
        panel.groupby("year_month")["excess_ret_fwd"]
        .transform(_winsorize)
    )
    panel[INTERACTION_COL] = (
        pd.to_numeric(panel[PE_QUINTILE_COL], errors="coerce")
        * pd.to_numeric(panel[SAT_QUINTILE_COL], errors="coerce")
    )

    # Monthly lambda_Int: interaction coefficient each month
    monthly_interaction = []
    for month, month_data in panel.groupby("year_month"):
        coefs = _runOneCrossSection(
            month_data,
            [PE_QUINTILE_COL, SAT_QUINTILE_COL, INTERACTION_COL],
        )
        if coefs is not None:
            monthly_interaction.append({
                "year_month":  month,
                "lambda_int":  coefs.get(INTERACTION_COL, np.nan),
            })

    if not monthly_interaction:
        raise RuntimeError("No valid cross-sections for Chow test")

    ts = pd.DataFrame(monthly_interaction).set_index("year_month")
    ts["post_2020"] = (ts.index > regimeBreakDate).astype(float)

    y = ts["lambda_int"].dropna().values
    X = sm.add_constant(
        ts.loc[ts["lambda_int"].notna(), ["post_2020"]].values
    )

    model = sm.OLS(y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": NEWEY_WEST_LAGS},
    )

    return pd.DataFrame({
        "variable":    ["const", "post_2020"],
        "coefficient": model.params,
        "t_stat":      model.tvalues,
        "p_value":     model.pvalues,
    })


# ── private ───────────────────────────────────────────────────────────────────

def _winsorize(series: pd.Series) -> pd.Series:
    """Winsorise a series at WINSOR_LOWER / WINSOR_UPPER quantiles."""
    lo = series.quantile(WINSOR_LOWER)
    hi = series.quantile(WINSOR_UPPER)
    return series.clip(lower=lo, upper=hi)


def _runOneCrossSection(
    monthData: pd.DataFrame,
    regressors: list[str],
) -> dict | None:
    clean = monthData[["excess_ret_fwd"] + regressors].dropna()
    if len(clean) < MIN_OBS_FOR_REGRESSION:
        return None
    y = clean["excess_ret_fwd"].values
    X = sm.add_constant(clean[regressors].values, has_constant="add")
    try:
        model = sm.OLS(y, X).fit()
        return dict(zip(["const"] + regressors, model.params))
    except Exception:
        return None


def _computeTimeSeriesMeans(
    coefPanel: pd.DataFrame,
    variables: list[str],
) -> pd.DataFrame:
    """Time-series mean and Newey-West SE for each coefficient series."""
    records = []
    for var in variables:
        if var not in coefPanel.columns:
            continue
        series = coefPanel[var].dropna().values
        n = len(series)
        if n < MIN_OBS_FOR_REGRESSION:
            records.append({
                "variable": var, "lambda_mean": np.nan,
                "t_stat": np.nan, "p_value": np.nan, "n_months": n,
            })
            continue
        model = sm.OLS(series, np.ones(n)).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": NEWEY_WEST_LAGS},
        )
        records.append({
            "variable":    var,
            "lambda_mean": float(model.params[0]),
            "t_stat":      float(model.tvalues[0]),
            "p_value":     float(model.pvalues[0]),
            "n_months":    n,
        })
    return pd.DataFrame(records)


def _validateFMBInputs(panel: pd.DataFrame):
    required = {
        "year_month", "ric", RETURN_COL, RISK_FREE_COL,
        PE_QUINTILE_COL, SAT_QUINTILE_COL,
    }
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Stock panel missing columns: {missing}")