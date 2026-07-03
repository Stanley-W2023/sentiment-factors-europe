"""
build_pe.py — Construct and clean the trailing PE ratio variable.

Sources TR.PE from Refinitiv (price / trailing 12-month EPS).
Applies cross-sectional winsorisation and excludes negative/missing PE.
"""

import numpy as np
import pandas as pd

from config.constants import (
    PE_WINSOR_LOWER,
    PE_WINSOR_UPPER,
    EXCLUDED_GICS_SECTORS,
    PE_COL,
)


def buildTrailingPE(rawPanel: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and winsorise the trailing PE ratio from the raw Refinitiv panel.

    Args:
        rawPanel: Monthly panel with columns [year_month, ric, pe_raw, gics_sector].

    Returns:
        Panel with additional column [pe_trailing], NaN where PE is missing
        or negative, winsorised cross-sectionally within each month.
    """
    _validatePEInputs(rawPanel)

    panel = rawPanel.copy()

    panel["pe_raw"] = pd.to_numeric(panel["pe_raw"], errors="coerce")
    if "gics_sector" in panel.columns:
        panel["gics_sector"] = pd.to_numeric(panel["gics_sector"], errors="coerce")

    panel = _excludeFinancials(panel)

    panel.loc[panel["pe_raw"] <= 0, "pe_raw"] = np.nan

    panel[PE_COL] = _winsoriseWithinMonth(panel, "pe_raw")

    return panel.drop(columns=["pe_raw"])


def buildSectorNormalisedPE(rawPanel: pd.DataFrame) -> pd.DataFrame:
    """
    Build sector-demeaned PE as an alternative sort variable.
    Used in robustness tests (Section 9 of the paper).

    Each stock's PE is expressed as a deviation from its GICS sector mean
    within each month, reducing the effect of cross-sector PE level differences.
    """
    _validatePEInputs(rawPanel)

    panel = rawPanel.copy()
    panel["pe_raw"] = pd.to_numeric(panel["pe_raw"], errors="coerce")
    if "gics_sector" in panel.columns:
        panel["gics_sector"] = pd.to_numeric(panel["gics_sector"], errors="coerce")
    panel = _excludeFinancials(panel)
    panel.loc[panel["pe_raw"] <= 0, "pe_raw"] = np.nan

    panel["pe_sector_mean"] = (
        panel.groupby(["year_month", "gics_sector"])["pe_raw"]
        .transform("mean")
    )
    panel[PE_COL] = panel["pe_raw"] - panel["pe_sector_mean"]

    return panel.drop(columns=["pe_raw", "pe_sector_mean"])


def _winsoriseWithinMonth(
    panel: pd.DataFrame,
    col: str,
) -> pd.Series:
    """
    Winsorise `col` at PE_WINSOR_LOWER and PE_WINSOR_UPPER cross-sectionally
    within each month. Applied after negative PE exclusion.
    """
    def _winsoriseOneMonth(series: pd.Series) -> pd.Series:
        lower = series.quantile(PE_WINSOR_LOWER)
        upper = series.quantile(PE_WINSOR_UPPER)
        return series.clip(lower=lower, upper=upper)

    return panel.groupby("year_month")[col].transform(_winsoriseOneMonth)


def _excludeFinancials(panel: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where gics_sector is in EXCLUDED_GICS_SECTORS."""
    if "gics_sector" not in panel.columns:
        return panel
    return panel[~panel["gics_sector"].isin(EXCLUDED_GICS_SECTORS)].copy()


def _validatePEInputs(rawPanel: pd.DataFrame):
    required = {"year_month", "ric", "pe_raw"}
    missing = required - set(rawPanel.columns)
    if missing:
        raise ValueError(f"Missing required columns in PE panel: {missing}")