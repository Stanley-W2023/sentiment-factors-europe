"""
fpe_function.py — The Weibull-based f(PE) retail sentiment intensity function.

Implements Equation (4) from the paper:

  f(PE) = A * ((PE - PE_min) / b)^(k-1) * exp(-((PE - PE_min) / b)^k)
        - delta * 1[PE < PE_star]

Properties:
  f(PE) < 0  for low-PE value stocks  (value-trap zone)
  f(PE) peak near PE ≈ 25-35          (maximum retail amplification)
  f(PE) → 0  for ultra-high PE        (bubble-fear moderation)

Parameters are estimated by NLS using the 5×5 portfolio mean returns
as targets, with initial values from FPE_*_INIT constants.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
import warnings

from config.constants import (
    FPE_SCALE_INIT,
    FPE_PEAK_PE_INIT,
    FPE_ASYMMETRY_INIT,
    FPE_PE_MIN_INIT,
    FPE_DELTA_INIT,
    FPE_PE_STAR_INIT,
    N_QUINTILES,
    ESG_QUINTILE_COL,
    SAT_QUINTILE_COL,
)


def evaluateFPE(
    pe_values: np.ndarray,
    scale: float,
    peak_pe: float,
    asymmetry: float,
    pe_min: float,
    delta: float,
    pe_star: float,
) -> np.ndarray:
    """
    Evaluate the Weibull f(PE) function at given PE values.

    Args:
        pe_values : Array of PE values.
        scale     : A — overall scale parameter.
        peak_pe   : b — location of the peak (PE at maximum sentiment sensitivity).
        asymmetry : k — shape/asymmetry parameter (k > 1 → right-skewed bell).
        pe_min    : PE_min — lower threshold below which Weibull is zero.
        delta     : depth of the value-trap penalty for PE < pe_star.
        pe_star   : PE_star — threshold below which penalty applies.

    Returns:
        Array of f(PE) values, same shape as pe_values.
    """
    pe = np.asarray(pe_values, dtype=float)
    shifted = np.maximum(pe - pe_min, 1e-10)
    b = np.maximum(peak_pe - pe_min, 1e-10)

    weibull = scale * (shifted / b) ** (asymmetry - 1) * np.exp(
        -((shifted / b) ** asymmetry)
    )
    penalty = delta * (pe < pe_star).astype(float)

    return weibull - penalty


def fitFPEFunction(
    portfolioMeanReturns: pd.DataFrame,
    peMedians: pd.Series,
    satMedians: pd.Series,
) -> dict:
    """
    Fit f(PE) parameters by NLS using portfolio mean excess returns as targets.

    The fitting strategy: for each PE quintile k, the average return across
    SAT quintiles provides a PE-level signal. We regress this on f(PE) using
    the median PE value of each quintile as the covariate.

    Args:
        portfolioMeanReturns: 5×5 DataFrame (PE quintile × SAT quintile)
                              of mean excess returns.
        peMedians: Series indexed by PE quintile (1–5) of median PE values.
        satMedians: Series indexed by SAT quintile (1–5) of median SAT values.

    Returns:
        Dict with fitted parameters and convergence diagnostics.
    """
    pe_avg_return = portfolioMeanReturns.mean(axis=1)  # Average over SAT quintiles

    pe_x = np.array([peMedians[k] for k in range(1, N_QUINTILES + 1)])
    y = np.array([pe_avg_return[k] for k in range(1, N_QUINTILES + 1)])

    initial_params = [
        FPE_SCALE_INIT,
        FPE_PEAK_PE_INIT,
        FPE_ASYMMETRY_INIT,
        FPE_PE_MIN_INIT,
        FPE_DELTA_INIT,
        FPE_PE_STAR_INIT,
    ]

    param_names = ["scale", "peak_pe", "asymmetry", "pe_min", "delta", "pe_star"]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, pcov = curve_fit(
                evaluateFPE,
                pe_x,
                y,
                p0=initial_params,
                bounds=(
                    [0,   0, 1,  0, 0,   0],  # lower bounds
                    [10, 300, 10, 30, 1, 100], # upper bounds — peak_pe raised to 300 to let data speak
                ),
                maxfev=5000,
            )

        fitted_values = evaluateFPE(pe_x, *popt)
        residuals = y - fitted_values
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        return {
            "params": dict(zip(param_names, popt)),
            "param_se": dict(zip(param_names, np.sqrt(np.diag(pcov)))),
            "r_squared": float(r_squared),
            "converged": True,
        }

    except (RuntimeError, ValueError) as e:
        return {
            "params": dict(zip(param_names, initial_params)),
            "param_se": dict(zip(param_names, [np.nan] * 6)),
            "r_squared": np.nan,
            "converged": False,
            "error": str(e),
        }


def plotFPECurve(
    fitResult: dict,
    peRange: tuple[float, float] = (3.0, 80.0),
    n_points: int = 200,
) -> pd.DataFrame:
    """
    Generate plotting data for the fitted f(PE) curve.

    Args:
        fitResult: Output of fitFPEFunction().
        peRange  : (min_pe, max_pe) for the x-axis.
        n_points : Number of evaluation points.

    Returns:
        DataFrame with columns [pe, fpe] for plotting.
    """
    if not fitResult.get("converged", False):
        raise ValueError("Cannot plot: f(PE) fit did not converge.")

    params = fitResult["params"]
    pe_grid = np.linspace(peRange[0], peRange[1], n_points)
    fpe_values = evaluateFPE(pe_grid, **params)

    return pd.DataFrame({"pe": pe_grid, "fpe": fpe_values})