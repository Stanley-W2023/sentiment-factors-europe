"""
build_esg.py — Construct and clean the Refinitiv ESG composite score variable.

Sources TR.TRESGScore from Refinitiv (0–100 composite of E, S, and G pillar
scores). Applies cross-sectional winsorisation and excludes financial firms.

Differences from the original PE-based sort:
  - No negative-value exclusion: ESG scores are bounded [0, 100] by construction.
  - Coverage is patchy pre-2013 — months below ESG_MIN_COVERAGE stocks are
    flagged rather than dropped, allowing the caller to decide.
  - No sector-normalised variant in the primary analysis (ESG scores are already
    constructed cross-sectorally by Refinitiv). A sector-adjusted robustness
    check can be added later if required.
"""

import numpy as np
import pandas as pd

from config.constants import (
    ESG_WINSOR_LOWER,
    ESG_WINSOR_UPPER,
    ESG_MIN_COVERAGE,
    EXCLUDED_GICS_SECTORS,
    ESG_COL,
)


def buildESGScore(rawPanel: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and winsorise the Refinitiv ESG composite score.

    Args:
        rawPanel: Monthly panel with columns [year_month, ric, esg_raw,
                  gics_sector]. esg_raw is TR.TRESGScore (0–100 scale).

    Returns:
        Panel with additional column [esg_score], NaN where ESG is missing.
        Winsorised cross-sectionally within each month. A coverage warning is
        printed for months below ESG_MIN_COVERAGE valid observations.
    """
    _validateESGInputs(rawPanel)

    panel = rawPanel.copy()
    panel["esg_raw"] = pd.to_numeric(panel["esg_raw"], errors="coerce")

    if "gics_sector" in panel.columns:
        panel["gics_sector"] = pd.to_numeric(panel["gics_sector"], errors="coerce")

    panel = _excludeFinancials(panel)

    # ESG scores are 0–100 — no negative exclusion needed.
    # Scores of exactly 0 are ambiguous (missing vs. genuine zero) in Refinitiv;
    # treat them as missing to avoid polluting the bottom quintile.
    panel.loc[panel["esg_raw"] == 0, "esg_raw"] = np.nan

    panel[ESG_COL] = _winsoriseWithinMonth(panel, "esg_raw")

    _checkCoverage(panel)

    return panel.drop(columns=["esg_raw"])


def _winsoriseWithinMonth(
    panel: pd.DataFrame,
    col: str,
) -> pd.Series:
    """
    Winsorise `col` at ESG_WINSOR_LOWER and ESG_WINSOR_UPPER cross-sectionally
    within each month. Applied after zero-score exclusion.
    """
    def _winsoriseOneMonth(series: pd.Series) -> pd.Series:
        lower = series.quantile(ESG_WINSOR_LOWER)
        upper = series.quantile(ESG_WINSOR_UPPER)
        return series.clip(lower=lower, upper=upper)

    return panel.groupby("year_month")[col].transform(_winsoriseOneMonth)


def _excludeFinancials(panel: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where gics_sector is in EXCLUDED_GICS_SECTORS."""
    if "gics_sector" not in panel.columns:
        return panel
    return panel[~panel["gics_sector"].isin(EXCLUDED_GICS_SECTORS)].copy()


def _checkCoverage(panel: pd.DataFrame):
    """
    Print a warning for any month with fewer than ESG_MIN_COVERAGE valid
    ESG observations. Coverage is thin pre-2013 in Refinitiv — these months
    will produce noisy quintile sorts and should be noted in the paper.
    """
    coverage = (
        panel.groupby("year_month")[ESG_COL]
        .apply(lambda s: s.notna().sum())
        .reset_index(name="n_valid")
    )
    thin = coverage[coverage["n_valid"] < ESG_MIN_COVERAGE]
    if not thin.empty:
        print(
            f"  [ESG coverage] {len(thin)} months have fewer than "
            f"{ESG_MIN_COVERAGE} valid ESG observations "
            f"(earliest: {thin['year_month'].min()}, "
            f"latest: {thin['year_month'].max()}). "
            f"Consider restricting the primary sample to post-2013."
        )


def _validateESGInputs(rawPanel: pd.DataFrame):
    required = {"year_month", "ric", "esg_raw"}
    missing = required - set(rawPanel.columns)
    if missing:
        raise ValueError(f"Missing required columns in ESG panel: {missing}")
