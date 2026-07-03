"""
daily_fmb.py — Daily Fama-MacBeth cross-sectional regressions.

Exact parallel of fama_macbeth.py but operating at daily frequency.

Each trading day d provides one cross-section:
  r_{i,d+1} - rf_{d+1} = lambda_0
                        + lambda_PE  * PEQ_{i,d}
                        + lambda_SAT * SATQ_{i,d}
                        + lambda_Int * (PEQ_{i,d} × SATQ_{i,d})
                        + epsilon_{i,d+1}

With ~4,000 trading days in the sample, the time-series means are estimated
from roughly 20× more cross-sections than the monthly FMB. Each cross-section
has ~480 stocks — same as monthly — so per-cross-section precision is unchanged.
The benefit is tighter NW standard errors on the time-series means.

Newey-West lags = 22 (≈ 1 trading month) to account for the higher
autocorrelation in daily lambda series compared to monthly. Using 6 lags
(the monthly default) would underestimate standard errors because daily
cross-sectional slopes are positively autocorrelated within each month.

Three SAT variants are estimated simultaneously. Results are returned as a
single DataFrame with a 'variant' column for easy comparison.

Regime analysis: estimates are split at the Mar-2020 COVID crash (same break
as the parametric monthly analysis) to test whether daily signal strengthened.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from features.build_daily_panel import (
    SAT_RAW_Q_COL,
    SAT_5D_Q_COL,
    SAT_10D_Q_COL,
)
from portfolios.construct_daily_ff25 import DAILY_NW_LAGS, SAT_VARIANTS
from config.constants import (
    PE_QUINTILE_COL,
    MIN_OBS_FOR_REGRESSION,
)

# Regime break for daily parametric analysis (COVID crash)
_DAILY_REGIME_BREAK = pd.Timestamp("2020-03-01")

_INTERACTION_COL = "pe_sat_interaction"
_WINSOR_LOWER    = 0.005   # 0.5% — tighter than monthly for daily returns
_WINSOR_UPPER    = 0.995


def estimateDailyFMB(dailyPanel: pd.DataFrame) -> pd.DataFrame:
    """
    Run daily FMB for all three SAT variants on the full sample.

    Args:
        dailyPanel: Output of buildDailyAnalysisPanel. Must contain
                    [date, ric, excess_ret_fwd, pe_quintile,
                     sat_raw_quintile, sat_5d_quintile, sat_10d_quintile].

    Returns:
        DataFrame with columns:
          [variant, variable, lambda_mean, t_stat, p_value, n_days]
        One row per (variant, regressor).
    """
    all_results = []
    for variant, sat_q_col in SAT_VARIANTS.items():
        print(f"  [daily FMB] Estimating {variant} variant "
              f"(full sample)...")
        res = _estimateOneVariant(dailyPanel, sat_q_col, label=f"{variant} (full)")
        res["variant"] = variant
        res["regime"]  = "full"
        all_results.append(res)

    result = pd.concat(all_results, ignore_index=True)
    _printFMBSummary(result)
    return result


def estimateDailyFMBRegimes(dailyPanel: pd.DataFrame) -> pd.DataFrame:
    """
    Run daily FMB for all three SAT variants split at Mar-2020.

    Returns same schema as estimateDailyFMB with additional 'regime' column
    taking values 'full', 'pre_2020', 'post_2020'.
    """
    pre  = dailyPanel[dailyPanel["date"] <  _DAILY_REGIME_BREAK]
    post = dailyPanel[dailyPanel["date"] >= _DAILY_REGIME_BREAK]

    all_results = []
    for variant, sat_q_col in SAT_VARIANTS.items():
        for regime_label, subpanel in [
            ("full",      dailyPanel),
            ("pre_2020",  pre),
            ("post_2020", post),
        ]:
            if len(subpanel) < MIN_OBS_FOR_REGRESSION * 50:
                continue
            print(f"  [daily FMB] {variant} / {regime_label} "
                  f"({subpanel['date'].nunique():,} days)...")
            res = _estimateOneVariant(subpanel, sat_q_col,
                                      label=f"{variant}/{regime_label}")
            res["variant"] = variant
            res["regime"]  = regime_label
            all_results.append(res)

    result = pd.concat(all_results, ignore_index=True)
    return result


# ── private ───────────────────────────────────────────────────────────────────

def _estimateOneVariant(
    panel: pd.DataFrame,
    satQCol: str,
    label: str = "",
) -> pd.DataFrame:
    """Run FMB for one SAT quintile column."""
    needed = ["date", "ric", "excess_ret_fwd", PE_QUINTILE_COL, satQCol]
    sub = panel[needed].dropna().copy()
    sub = sub.rename(columns={satQCol: "sat_q"})

    sub["sat_q"]  = pd.to_numeric(sub["sat_q"],  errors="coerce")
    sub["pe_q"]   = pd.to_numeric(sub[PE_QUINTILE_COL], errors="coerce")
    sub[_INTERACTION_COL] = sub["pe_q"] * sub["sat_q"]

    regressors = ["pe_q", "sat_q", _INTERACTION_COL]

    daily_coefs = []
    for day, day_data in sub.groupby("date"):
        coefs = _runOneDayCrossSection(day_data, regressors)
        if coefs is not None:
            coefs["date"] = day
            daily_coefs.append(coefs)

    if not daily_coefs:
        raise RuntimeError(f"No valid daily cross-sections for {label}")

    coef_panel = pd.DataFrame(daily_coefs).set_index("date")
    n_days = len(coef_panel)

    records = []
    for var in regressors + ["const"]:
        if var not in coef_panel.columns:
            continue
        series = coef_panel[var].dropna().values
        n = len(series)
        if n < DAILY_NW_LAGS + 2:
            records.append({
                "variable": var, "lambda_mean": np.nan,
                "t_stat": np.nan, "p_value": np.nan, "n_days": n,
            })
            continue
        model = sm.OLS(series, np.ones(n)).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": DAILY_NW_LAGS},
        )
        records.append({
            "variable":    _rename(var),
            "lambda_mean": float(model.params[0]),
            "t_stat":      float(model.tvalues[0]),
            "p_value":     float(model.pvalues[0]),
            "n_days":      n,
        })
    return pd.DataFrame(records)


def _runOneDayCrossSection(
    dayData: pd.DataFrame,
    regressors: list[str],
) -> dict | None:
    clean = dayData[["excess_ret_fwd"] + regressors].dropna()
    if len(clean) < MIN_OBS_FOR_REGRESSION:
        return None
    y = clean["excess_ret_fwd"].values
    X = sm.add_constant(clean[regressors].values, has_constant="add")
    try:
        model = sm.OLS(y, X).fit()
        return dict(zip(["const"] + regressors, model.params))
    except Exception:
        return None


def _rename(col: str) -> str:
    """Map internal column names to human-readable variable names."""
    return {
        "pe_q":              "pe_quintile",
        "sat_q":             "sat_quintile",
        _INTERACTION_COL:    "pe_sat_interaction",
        "const":             "const",
    }.get(col, col)


def _printFMBSummary(results: pd.DataFrame):
    """Print a side-by-side comparison table for the three SAT variants."""
    print("\n  [daily FMB] Results — full sample")
    print(f"  {'Variable':<22} {'raw lam':>10} {'raw t':>8} "
          f"{'5d lam':>10} {'5d t':>8} {'10d lam':>10} {'10d t':>8}")
    print("  " + "-" * 82)

    variables = ["pe_quintile", "sat_quintile", "pe_sat_interaction", "const"]
    for var in variables:
        row_vals = {}
        for variant in ["raw", "5d", "10d"]:
            sub = results[(results["variant"] == variant) &
                          (results["variable"] == var) &
                          (results["regime"]   == "full")]
            if sub.empty:
                row_vals[variant] = (np.nan, np.nan)
            else:
                row_vals[variant] = (
                    sub["lambda_mean"].iloc[0],
                    sub["t_stat"].iloc[0],
                )

        def _fmt(lam, t):
            if np.isnan(lam):
                return "       n/a", "     n/a"
            stars = "***" if abs(t) > 3 else "**" if abs(t) > 2.5 \
                    else "*" if abs(t) > 2 else ""
            return f"{lam:10.6f}", f"{t:6.2f}{stars}"

        parts = []
        for v in ["raw", "5d", "10d"]:
            lm, ts = _fmt(*row_vals[v])
            parts += [lm, ts]

        print(f"  {var:<22} {'  '.join(parts)}")