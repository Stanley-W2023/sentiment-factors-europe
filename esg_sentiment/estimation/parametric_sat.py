"""
parametric_sat.py — Parametric framing of the ESG-conditioned SAT premium.

Core idea: the SAT premium (lambda_SAT) varies non-linearly with ESG score.
We estimate lambda_SAT *within* each ESG quintile by running FMB cross-sections
restricted to stocks in that quintile. The resulting 5 (ESG_median, lambda_SAT)
pairs are then fitted with a Weibull-based f(ESG) curve.

This answers: "Does ESG score amplify the SAT return premium, and if so, what
shape does that amplification take?" — the parametric interpretation of the
ESG×SAT interaction hypothesis.

Three-regime analysis: we split on both structural breaks from constants:
  pre_2018          — before Jan 2018 (pre-EU Action Plan)
  between_2018_2020 — Jan 2018 – Dec 2019 (institutional ESG mainstreaming)
  post_2020         — Jan 2020 onwards (retail surge + SFDR)

Comparing f(ESG) curves across all three regimes tests whether the ESG-
conditioned SAT amplification emerged with institutional flows (2018 break)
or was further amplified by retail sentiment (2020 break).

The Weibull shape captures three phenomena:
  - Low-ESG penalty (ESG < esg_star):  reduced/negative SAT amplification
    ("sustainability laggard" narrative drives disproportionate retail selling)
  - Sweet spot (ESG ~ peak_esg): maximum retail amplification of the ESG premium
  - High-ESG moderation (ESG >> peak_esg): amplification decays as the
    "already priced" perception overrides narrative enthusiasm
"""

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import curve_fit, OptimizeWarning

from config.constants import (
    NEWEY_WEST_LAGS,
    MIN_OBS_FOR_REGRESSION,
    ESG_QUINTILE_COL,
    SAT_QUINTILE_COL,
    ESG_COL,
    RETURN_COL,
    RISK_FREE_COL,
    N_QUINTILES,
    REGIME_BREAK_DATE_2018,
    REGIME_BREAK_DATE_2020,
)

# Weibull f(ESG) parameter bounds — rescaled for 0-100 ESG score range.
# [scale, peak_esg, asymmetry, esg_min, delta, esg_star]
_FESG_INIT  = [0.003,  50.0, 2.0, 5.0,  0.001, 20.0]
_FESG_LOWER = [0,       0,   1.0, 0,    0,      0   ]
_FESG_UPPER = [0.05,  100,   8.0, 30,   0.05,  50   ]

WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99


def estimateParametricSATAmplification(
    stockPanel: pd.DataFrame,
) -> dict:
    """
    Estimate the ESG-conditioned SAT premium and fit the parametric f(ESG) curve.

    Returns a dict with keys:
      "full"              : results for the full sample
      "pre_2018"          : before Jan 2018 (pre-EU Action Plan)
      "between_2018_2020" : Jan 2018 – Dec 2019 (institutional mainstreaming)
      "post_2020"         : from Jan 2020 onwards (retail surge + SFDR)

    Each sub-dict contains:
      "lambda_sat_by_esg" : DataFrame [esg_quintile, esg_median, lambda_sat,
                            t_stat, p_value, n_months]
      "fesg_fit"          : dict with keys [params, param_se, r_squared,
                            converged, esg_grid, fesg_curve]
    """
    panel = _preparePanel(stockPanel)

    break_2018 = pd.Period(str(REGIME_BREAK_DATE_2018)[:7], freq="M")
    break_2020 = pd.Period(str(REGIME_BREAK_DATE_2020)[:7], freq="M")

    results = {}
    results["full"] = _estimateOneRegime(panel, label="full sample")
    results["pre_2018"] = _estimateOneRegime(
        panel[panel["year_month"] <= break_2018], label="pre-2018"
    )
    results["between_2018_2020"] = _estimateOneRegime(
        panel[
            (panel["year_month"] > break_2018) &
            (panel["year_month"] <= break_2020)
        ],
        label="2018-2019 (institutional)",
    )
    results["post_2020"] = _estimateOneRegime(
        panel[panel["year_month"] > break_2020], label="post-2020"
    )

    _printSummary(results)
    return results


def _preparePanel(stockPanel: pd.DataFrame) -> pd.DataFrame:
    panel = stockPanel.copy().sort_values(["ric", "year_month"])
    panel["excess_ret_fwd"] = (
        panel.groupby("ric")[RETURN_COL].shift(-1)
        - panel.groupby("ric")[RISK_FREE_COL].shift(-1)
    )
    panel["excess_ret_fwd"] = (
        panel.groupby("year_month")["excess_ret_fwd"]
        .transform(_winsorize)
    )
    return panel


def _estimateOneRegime(panel: pd.DataFrame, label: str) -> dict:
    """
    For each ESG quintile, run FMB with SAT as the sole regressor.
    Returns lambda_SAT(k) for k=1..5 plus the fitted f(ESG) curve.
    """
    records = []

    for k in range(1, N_QUINTILES + 1):
        esg_sub = panel[panel[ESG_QUINTILE_COL] == k]

        esg_median = pd.to_numeric(
            esg_sub[ESG_COL], errors="coerce"
        ).median()

        monthly_lambda = []
        for month, month_data in esg_sub.groupby("year_month"):
            coef = _runSATOnlyCrossSection(month_data)
            if coef is not None:
                monthly_lambda.append(coef)

        n = len(monthly_lambda)
        if n < MIN_OBS_FOR_REGRESSION:
            records.append({
                "esg_quintile": k,
                "esg_median":   esg_median,
                "lambda_sat":   np.nan,
                "t_stat":       np.nan,
                "p_value":      np.nan,
                "n_months":     n,
            })
            continue

        series = np.array(monthly_lambda)
        model = sm.OLS(series, np.ones(n)).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": NEWEY_WEST_LAGS},
        )
        records.append({
            "esg_quintile": k,
            "esg_median":   esg_median,
            "lambda_sat":   float(model.params[0]),
            "t_stat":       float(model.tvalues[0]),
            "p_value":      float(model.pvalues[0]),
            "n_months":     n,
        })

    lam_df   = pd.DataFrame(records)
    fesg_fit = _fitFESGToLambdaSAT(lam_df)

    return {
        "lambda_sat_by_esg": lam_df,
        "fesg_fit":          fesg_fit,
        "label":             label,
    }


def _runSATOnlyCrossSection(monthData: pd.DataFrame) -> float | None:
    """
    Run one cross-section within an ESG quintile: r_fwd ~ const + SAT.
    Returns the SAT coefficient only.
    """
    clean = monthData[["excess_ret_fwd", SAT_QUINTILE_COL]].dropna()
    if len(clean) < MIN_OBS_FOR_REGRESSION:
        return None
    y = clean["excess_ret_fwd"].values
    X = sm.add_constant(
        pd.to_numeric(clean[SAT_QUINTILE_COL], errors="coerce").values,
        has_constant="add",
    )
    try:
        model = sm.OLS(y, X).fit()
        return float(model.params[1])
    except Exception:
        return None


def _weibull(esg, scale, peak_esg, k, esg_min, delta, esg_star):
    """
    Weibull f(ESG) — same functional form as rsf_spec/fpe_function.py,
    reparameterised for ESG 0-100 scale.
    """
    esg     = np.asarray(esg, dtype=float)
    shifted = np.maximum(esg - esg_min, 1e-10)
    b       = np.maximum(peak_esg - esg_min, 1e-10)
    weibull = scale * (shifted / b) ** (k - 1) * np.exp(-((shifted / b) ** k))
    penalty = delta * (esg < esg_star).astype(float)
    return weibull - penalty


def _fitFESGToLambdaSAT(lambdaDf: pd.DataFrame) -> dict:
    """
    Fit the Weibull f(ESG) to (esg_median, lambda_SAT) pairs.
    Returns fit diagnostics and a dense evaluation grid for plotting.
    """
    valid = lambdaDf.dropna(subset=["esg_median", "lambda_sat"])
    if len(valid) < 3:
        return {"converged": False, "error": "insufficient data points"}

    esg_x = valid["esg_median"].values
    y     = valid["lambda_sat"].values

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, pcov = curve_fit(
                _weibull, esg_x, y,
                p0=_FESG_INIT,
                bounds=(_FESG_LOWER, _FESG_UPPER),
                maxfev=10000,
            )

        fitted  = _weibull(esg_x, *popt)
        ss_res  = np.sum((y - fitted) ** 2)
        ss_tot  = np.sum((y - y.mean()) ** 2)
        r_sq    = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        param_names = ["scale", "peak_esg", "asymmetry", "esg_min", "delta", "esg_star"]
        params      = dict(zip(param_names, popt))
        param_se    = dict(zip(param_names, np.sqrt(np.diag(pcov))))

        # Dense grid for smooth curve plotting (ESG 0-100 range)
        esg_grid   = np.linspace(max(1, esg_x.min() - 5), min(100, esg_x.max() + 10), 200)
        fesg_curve = _weibull(esg_grid, *popt)

        return {
            "converged":   True,
            "params":      params,
            "param_se":    param_se,
            "r_squared":   float(r_sq),
            "esg_grid":    esg_grid,
            "fesg_curve":  fesg_curve,
        }

    except (RuntimeError, ValueError) as e:
        return {"converged": False, "error": str(e)}


# ── private: printing ─────────────────────────────────────────────────────────

def _printSummary(results: dict):
    for regime, res in results.items():
        lam   = res["lambda_sat_by_esg"]
        fit   = res["fesg_fit"]
        label = res.get("label", regime)
        print(f"\n  [Parametric SAT] {label}")
        print(f"    lambda_SAT by ESG quintile:")
        for _, row in lam.iterrows():
            stars = ("***" if abs(row["t_stat"]) > 3
                     else "**" if abs(row["t_stat"]) > 2.5
                     else "*"  if abs(row["t_stat"]) > 2 else "")
            print(f"      ESG Q{int(row['esg_quintile'])} "
                  f"(med ESG={row['esg_median']:.1f}): "
                  f"lambda={row['lambda_sat']:.5f}  t={row['t_stat']:.2f}{stars}")
        if fit.get("converged"):
            p = fit["params"]
            print(f"    f(ESG) fit: peak_esg={p['peak_esg']:.1f}  "
                  f"scale={p['scale']:.4f}  "
                  f"asymmetry={p['asymmetry']:.2f}  "
                  f"r2={fit['r_squared']:.3f}")
        else:
            print(f"    f(ESG) fit: did not converge -- {fit.get('error', '')}")


# ── helpers ───────────────────────────────────────────────────────────────────

def _winsorize(series: pd.Series) -> pd.Series:
    lo = series.quantile(WINSOR_LOWER)
    hi = series.quantile(WINSOR_UPPER)
    return series.clip(lower=lo, upper=hi)
