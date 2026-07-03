"""
parametric_sat.py — Parametric framing of the PE-conditioned SAT premium.

Core idea: the SAT premium (lambda_SAT) varies non-linearly with PE valuation.
We estimate lambda_SAT *within* each PE quintile by running FMB cross-sections
restricted to stocks in that quintile. The resulting 5 (PE_median, lambda_SAT)
pairs are then fitted with a Weibull-based f(PE) curve.

This answers: "Does PE amplify the SAT return premium, and if so, what shape
does that amplification take?" — the parametric interpretation of Hypothesis 1.

Regime analysis: we repeat the procedure for pre-2020 and post-2020 subsamples.
Comparing the fitted f(PE) curves across regimes tests Hypothesis 4 — whether
the PE-conditioned SAT amplification strengthened in the post-retail-era.

The Weibull shape captures three phenomena simultaneously:
  - Value trap (PE < PE_star): SAT has reduced or negative amplification
  - Sweet spot (PE ~ peak_pe): maximum SAT amplification
  - Growth moderation (PE >> peak_pe): amplification decays as bubble fear
    overrides retail sentiment
"""

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import curve_fit, OptimizeWarning

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
    PE_COL,
    RETURN_COL,
    RISK_FREE_COL,
    N_QUINTILES,
)

_FPE_INIT   = [0.003, 25.0, 2.0, 3.0, 0.001, 10.0] 
_FPE_LOWER  = [0,    0,  1.0, 0,  0,    0  ]
_FPE_UPPER  = [0.05, 150, 8.0, 20, 0.05, 50 ]

WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99


def estimateParametricSATAmplification(
    stockPanel: pd.DataFrame,
) -> dict:
    """
    Estimate the PE-conditioned SAT premium and fit the parametric f(PE) curve.

    Returns a dict with keys:
      "full"     : results for the full sample
      "pre_2020" : results for months up to and including REGIME_BREAK_DATE
      "post_2020": results for months after REGIME_BREAK_DATE

    Each sub-dict contains:
      "lambda_sat_by_pe" : DataFrame [pe_quintile, pe_median, lambda_sat,
                           t_stat, p_value, n_months]
      "fpe_fit"          : dict with keys [params, param_se, r_squared,
                           converged, pe_grid, fpe_curve]
    """
    panel = _preparePanel(stockPanel)

    regime_break = pd.Period("2020-03", freq="M")

    results = {}
    results["full"]      = _estimateOneRegime(panel, label="full sample")
    results["pre_2020"]  = _estimateOneRegime(
        panel[panel["year_month"] <= regime_break], label="pre-2020"
    )
    results["post_2020"] = _estimateOneRegime(
        panel[panel["year_month"] > regime_break], label="post-2020"
    )

    _printSummary(results)
    return results


def _preparePanel(stockPanel: pd.DataFrame) -> pd.DataFrame:
    panel = addForwardMonthlyReturns(stockPanel)
    panel["excess_ret_fwd"] = panel[FWD_RETURN_COL] - panel[FWD_RF_COL]
    panel["excess_ret_fwd"] = (
        panel.groupby("year_month")["excess_ret_fwd"]
        .transform(_winsorize)
    )
    return panel


def _estimateOneRegime(panel: pd.DataFrame, label: str) -> dict:
    """
    For each PE quintile, run FMB with SAT as the sole regressor.
    Returns lambda_SAT(k) for k=1..5 plus the fitted f(PE) curve.
    """
    records = []

    for k in range(1, N_QUINTILES + 1):
        pe_sub = panel[panel[PE_QUINTILE_COL] == k]

        pe_median = pd.to_numeric(
            pe_sub[PE_COL], errors="coerce"
        ).median()

        monthly_lambda = []
        for month, month_data in pe_sub.groupby("year_month"):
            coef = _runSATOnlyCrossSection(month_data)
            if coef is not None:
                monthly_lambda.append(coef)

        n = len(monthly_lambda)
        if n < MIN_OBS_FOR_REGRESSION:
            records.append({
                "pe_quintile": k,
                "pe_median":   pe_median,
                "lambda_sat":  np.nan,
                "t_stat":      np.nan,
                "p_value":     np.nan,
                "n_months":    n,
            })
            continue

        series = np.array(monthly_lambda)
        model = sm.OLS(series, np.ones(n)).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": NEWEY_WEST_LAGS},
        )
        records.append({
            "pe_quintile": k,
            "pe_median":   pe_median,
            "lambda_sat":  float(model.params[0]),
            "t_stat":      float(model.tvalues[0]),
            "p_value":     float(model.pvalues[0]),
            "n_months":    n,
        })

    lam_df = pd.DataFrame(records)
    fpe_fit = _fitFPEToLambdaSAT(lam_df)

    return {
        "lambda_sat_by_pe": lam_df,
        "fpe_fit":          fpe_fit,
        "label":            label,
    }


def _runSATOnlyCrossSection(monthData: pd.DataFrame) -> float | None:
    """
    Run one cross-section within a PE quintile: r_fwd ~ const + SAT.
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



def _weibull(pe, scale, peak_pe, k, pe_min, delta, pe_star):
    """Weibull f(PE) — same functional form as rsf_spec/fpe_function.py."""
    pe = np.asarray(pe, dtype=float)
    shifted = np.maximum(pe - pe_min, 1e-10)
    b = np.maximum(peak_pe - pe_min, 1e-10)
    weibull = scale * (shifted / b) ** (k - 1) * np.exp(-((shifted / b) ** k))
    penalty = delta * (pe < pe_star).astype(float)
    return weibull - penalty


def _fitFPEToLambdaSAT(lambdaDf: pd.DataFrame) -> dict:
    """
    Fit the Weibull f(PE) to (pe_median, lambda_SAT) pairs.
    Returns fit diagnostics and a dense evaluation grid for plotting.
    """
    valid = lambdaDf.dropna(subset=["pe_median", "lambda_sat"])
    if len(valid) < 3:
        return {"converged": False, "error": "insufficient data points"}

    pe_x = valid["pe_median"].values
    y    = valid["lambda_sat"].values

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, pcov = curve_fit(
                _weibull, pe_x, y,
                p0=_FPE_INIT,
                bounds=(_FPE_LOWER, _FPE_UPPER),
                maxfev=10000,
            )

        fitted   = _weibull(pe_x, *popt)
        ss_res   = np.sum((y - fitted) ** 2)
        ss_tot   = np.sum((y - y.mean()) ** 2)
        r_sq     = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        param_names = ["scale", "peak_pe", "asymmetry", "pe_min", "delta", "pe_star"]
        params      = dict(zip(param_names, popt))
        param_se    = dict(zip(param_names, np.sqrt(np.diag(pcov))))

        # Dense grid for smooth curve plotting
        pe_grid  = np.linspace(max(1, pe_x.min() - 5), pe_x.max() + 10, 200)
        fpe_curve = _weibull(pe_grid, *popt)

        return {
            "converged":  True,
            "params":     params,
            "param_se":   param_se,
            "r_squared":  float(r_sq),
            "pe_grid":    pe_grid,
            "fpe_curve":  fpe_curve,
        }

    except (RuntimeError, ValueError) as e:
        return {"converged": False, "error": str(e)}


# ── private: printing ─────────────────────────────────────────────────────────

def _printSummary(results: dict):
    for regime, res in results.items():
        lam = res["lambda_sat_by_pe"]
        fit = res["fpe_fit"]
        label = res.get("label", regime)
        print(f"\n  [Parametric SAT] {label}")
        print(f"    lambda_SAT by PE quintile:")
        for _, row in lam.iterrows():
            stars = ("***" if abs(row["t_stat"]) > 3
                     else "**" if abs(row["t_stat"]) > 2.5
                     else "*"  if abs(row["t_stat"]) > 2 else "")
            print(f"      PE Q{int(row['pe_quintile'])} "
                  f"(med PE={row['pe_median']:.1f}): "
                  f"lambda={row['lambda_sat']:.5f}  t={row['t_stat']:.2f}{stars}")
        if fit.get("converged"):
            p = fit["params"]
            print(f"    f(PE) fit: peak_pe={p['peak_pe']:.1f}  "
                  f"scale={p['scale']:.4f}  "
                  f"asymmetry={p['asymmetry']:.2f}  "
                  f"r2={fit['r_squared']:.3f}")
        else:
            print(f"    f(PE) fit: did not converge — {fit.get('error', '')}")


# ── helpers ───────────────────────────────────────────────────────────────────

def _winsorize(series: pd.Series) -> pd.Series:
    lo = series.quantile(WINSOR_LOWER)
    hi = series.quantile(WINSOR_UPPER)
    return series.clip(lower=lo, upper=hi)