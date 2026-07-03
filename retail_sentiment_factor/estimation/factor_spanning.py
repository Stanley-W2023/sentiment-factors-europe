"""
factor_spanning.py — Test whether RSF complements the Fama-French regressors.

RSF is treated as a candidate RIGHT-HAND-SIDE factor, not a prediction
target. Two questions are answered:

1. Spanning: is RSF redundant given FF5 + momentum?
     RSF_t = α + β' F_t + e_t          (F = Mkt-RF, SMB, HML, RMW, CMA, UMD)
   A significant α (Newey-West) means the FF factors do NOT span RSF —
   RSF carries priced variation they miss, so it earns a seat on the RHS.
   Barillas & Shanken (2017): for factor-model comparison, only the alpha
   from spanning regressions matters.

2. Stripping: what happens to the 25 portfolio alphas when RSF is added
   as a 7th regressor? If RSF is a genuine common factor, adding it should
   absorb sentiment-driven comovement: |alphas| shrink and fewer survive,
   while the RSF betas line up monotonically with the SAT sort.

Both use the same monthly factor matrix as the FF6 alpha step, with RSF
merged on the holding-month year_month label.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from estimation.time_series_alpha import estimateTimeSeriesAlpha
from factors.build_rsf_factor import RSF_FACTOR_COL
from config.constants import (
    FF_FACTORS,
    NEWEY_WEST_LAGS,
    MIN_OBS_FOR_REGRESSION,
    PE_QUINTILE_COL,
    SAT_QUINTILE_COL,
)

_SIG_T: float = 2.0


def runSpanningRegression(
    rsfFactor: pd.DataFrame,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Regress the RSF factor on the FF5 + momentum factors.

    Args:
        rsfFactor: [year_month, RSF] from buildRSFFactor (holding-month label).
        factors:   Monthly factor matrix with FF_FACTORS columns and a
                   year_month column (same input as estimateTimeSeriesAlpha).

    Returns:
        Tidy DataFrame [variable, coefficient, t_stat, p_value] with one row
        per regressor plus 'alpha', and appended diagnostic rows for
        r_squared and n_months. alpha is the monthly unspanned mean return.
    """
    merged = _mergeFactorPanels(rsfFactor, factors)
    if len(merged) < MIN_OBS_FOR_REGRESSION:
        raise RuntimeError(
            f"Only {len(merged)} overlapping months between RSF and FF factors"
        )

    y = merged[RSF_FACTOR_COL].values
    X = sm.add_constant(merged[FF_FACTORS].values)
    model = sm.OLS(y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": NEWEY_WEST_LAGS},
    )

    names = ["alpha"] + list(FF_FACTORS)
    records = [
        {
            "variable":    name,
            "coefficient": float(model.params[i]),
            "t_stat":      float(model.tvalues[i]),
            "p_value":     float(model.pvalues[i]),
        }
        for i, name in enumerate(names)
    ]
    records.append({"variable": "r_squared",
                    "coefficient": float(model.rsquared),
                    "t_stat": np.nan, "p_value": np.nan})
    records.append({"variable": "n_months",
                    "coefficient": float(model.nobs),
                    "t_stat": np.nan, "p_value": np.nan})
    return pd.DataFrame(records)


def computeFactorCorrelations(
    rsfFactor: pd.DataFrame,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pearson correlation of RSF with each FF factor over the common sample.
    Low correlations are the first-pass evidence that RSF is not a
    repackaged FF factor.
    """
    merged = _mergeFactorPanels(rsfFactor, factors)
    records = [
        {
            "factor": f,
            "correlation": float(merged[RSF_FACTOR_COL].corr(merged[f])),
        }
        for f in FF_FACTORS
    ]
    return pd.DataFrame(records)


def compareAlphasWithRSF(
    portfolioReturns: pd.DataFrame,
    factors: pd.DataFrame,
    rsfFactor: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Estimate the 25 portfolio alphas under FF6 and FF6 + RSF.

    Returns:
        comparison: one row per portfolio —
          [pe_quintile, sat_quintile, alpha_ff6, t_ff6, alpha_ff7, t_ff7,
           beta_RSF, t_... ] where alpha_ff7 is the FF6+RSF alpha.
        summary: one row per model —
          [model, mean_abs_alpha, n_significant, mean_r_squared].
    """
    factors_with_rsf = _mergeFactorPanels(rsfFactor, factors)

    ff6 = estimateTimeSeriesAlpha(portfolioReturns, factors, FF_FACTORS)
    ff7 = estimateTimeSeriesAlpha(
        portfolioReturns, factors_with_rsf, FF_FACTORS + [RSF_FACTOR_COL]
    )

    keys = [PE_QUINTILE_COL, SAT_QUINTILE_COL]
    comparison = (
        ff6[keys + ["alpha", "t_stat", "r_squared"]]
        .rename(columns={"alpha": "alpha_ff6", "t_stat": "t_ff6",
                         "r_squared": "r2_ff6"})
        .merge(
            ff7[keys + ["alpha", "t_stat", "r_squared", f"beta_{RSF_FACTOR_COL}"]]
            .rename(columns={"alpha": "alpha_ff7", "t_stat": "t_ff7",
                             "r_squared": "r2_ff7"}),
            on=keys,
        )
    )

    summary = pd.DataFrame([
        _summariseModel("FF6", comparison, "alpha_ff6", "t_ff6", "r2_ff6"),
        _summariseModel("FF6+RSF", comparison, "alpha_ff7", "t_ff7", "r2_ff7"),
    ])
    return comparison, summary


def printSpanningSummary(
    spanning: pd.DataFrame,
    correlations: pd.DataFrame,
    alphaSummary: pd.DataFrame,
):
    """Console summary of the three spanning outputs."""
    alpha_row = spanning[spanning["variable"] == "alpha"].iloc[0]
    r2_row = spanning[spanning["variable"] == "r_squared"].iloc[0]
    print("\n  [spanning] RSF on FF5+UMD:")
    print(f"    alpha = {alpha_row['coefficient']*100:.3f}%/month  "
          f"(t = {alpha_row['t_stat']:.2f}, p = {alpha_row['p_value']:.4f})")
    print(f"    R2    = {r2_row['coefficient']:.3f}")

    print("\n  [spanning] Correlations with FF factors:")
    for _, row in correlations.iterrows():
        print(f"    corr(RSF, {row['factor']:<7}) = {row['correlation']:+.3f}")

    print("\n  [spanning] 25-portfolio alpha comparison:")
    print(alphaSummary.to_string(index=False))


# ── private ───────────────────────────────────────────────────────────────────

def _mergeFactorPanels(
    rsfFactor: pd.DataFrame,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join RSF onto the FF factor matrix by year_month (Period[M])."""
    rsf = rsfFactor[["year_month", RSF_FACTOR_COL]].copy()
    if not isinstance(rsf["year_month"].dtype, pd.PeriodDtype):
        rsf["year_month"] = pd.PeriodIndex(rsf["year_month"], freq="M")

    ff = factors.copy()
    if not isinstance(ff["year_month"].dtype, pd.PeriodDtype):
        ff["year_month"] = pd.PeriodIndex(ff["year_month"], freq="M")

    merged = ff.merge(rsf, on="year_month", how="inner")
    return merged.dropna(subset=list(FF_FACTORS) + [RSF_FACTOR_COL])


def _summariseModel(
    label: str,
    comparison: pd.DataFrame,
    alphaCol: str,
    tCol: str,
    r2Col: str,
) -> dict:
    valid = comparison.dropna(subset=[alphaCol, tCol])
    return {
        "model":          label,
        "mean_abs_alpha": float(valid[alphaCol].abs().mean()),
        "n_significant":  int((valid[tCol].abs() > _SIG_T).sum()),
        "mean_r_squared": float(valid[r2Col].mean()),
    }
