"""
build_rsf_factor.py — Construct RSF as a tradable Fama-French-style factor.

Purpose: reframe the Retail Sentiment Factor as a COMPLEMENTARY right-hand-
side factor alongside the FF5 + momentum regressors — something that is
stripped out of returns like SMB or HML — rather than a return-prediction
signal.

Construction mirrors the Fama-French SMB/HML methodology (2×3 double sort):

  At the end of formation month t:
    - Size split : Small / Big at the cross-sectional median of me_eur.
    - SAT split  : Low / Mid / High at the 30th and 70th percentiles of
                   sat_monthly (already cross-sectionally z-scored at t).
  Six value-weighted portfolios are formed from the intersections and held
  over month t+1 (weights = month-t market cap; strictly no look-ahead —
  forward returns come from features.forward_returns).

  RSF_{t+1} = ½ (Small/High + Big/High) − ½ (Small/Low + Big/Low)

The size neutralisation is the point of the 2×3 design: retail activity
concentrates in smaller names, so a plain high-minus-low SAT spread would
be substantially a size bet. Averaging across the size legs isolates the
sentiment tilt, keeping RSF complementary to (rather than a repackaging of)
SMB.

Output year_month is the HOLDING month (t+1), aligned with the Ken French
factor convention so RSF can be appended directly to the FF factor matrix.
"""

import numpy as np
import pandas as pd

from features.forward_returns import (
    addForwardMonthlyReturns,
    FWD_RETURN_COL,
    FWD_RF_COL,
)
from config.constants import (
    SAT_COL,
    MARKET_CAP_COL,
)

RSF_FACTOR_COL = "RSF"

_SAT_LOW_PCT: float = 0.30
_SAT_HIGH_PCT: float = 0.70
_MIN_STOCKS_PER_LEG: int = 5


def buildRSFFactor(monthlyPanel: pd.DataFrame) -> pd.DataFrame:
    """
    Build the monthly RSF factor return series.

    Args:
        monthlyPanel: Monthly stock panel with columns
                      [year_month, ric, sat_monthly, me_eur, ret_eur, rf_eur].
                      (The standard monthly_panel from Step 3.)

    Returns:
        DataFrame with columns
          [year_month, RSF, rsf_small_high, rsf_big_high,
           rsf_small_low, rsf_big_low, n_stocks]
        where year_month is the holding month (formation + 1) and RSF is
        the factor return in decimal units. Months where any of the four
        corner legs has fewer than _MIN_STOCKS_PER_LEG stocks are dropped.
    """
    _validateFactorInputs(monthlyPanel)

    panel = addForwardMonthlyReturns(monthlyPanel)
    panel = panel.dropna(
        subset=[SAT_COL, MARKET_CAP_COL, FWD_RETURN_COL]
    )
    panel = panel[panel[MARKET_CAP_COL] > 0]

    records = []
    for month, grp in panel.groupby("year_month"):
        row = _buildOneMonth(grp)
        if row is not None:
            # Label with the holding month so RSF aligns with FF factors
            row["year_month"] = month + 1
            records.append(row)

    if not records:
        raise RuntimeError("No months with enough stocks to build the RSF factor")

    factor = pd.DataFrame(records)
    cols = ["year_month", RSF_FACTOR_COL,
            "rsf_small_high", "rsf_big_high",
            "rsf_small_low", "rsf_big_low", "n_stocks"]
    return factor[cols].sort_values("year_month").reset_index(drop=True)


# ── private ───────────────────────────────────────────────────────────────────

def _buildOneMonth(grp: pd.DataFrame) -> dict | None:
    """2×3 sort for one formation month; returns None if legs are too thin."""
    size_median = grp[MARKET_CAP_COL].median()
    sat_lo = grp[SAT_COL].quantile(_SAT_LOW_PCT)
    sat_hi = grp[SAT_COL].quantile(_SAT_HIGH_PCT)

    small = grp[MARKET_CAP_COL] <= size_median
    big = ~small
    low = grp[SAT_COL] <= sat_lo
    high = grp[SAT_COL] >= sat_hi

    legs = {
        "rsf_small_high": grp[small & high],
        "rsf_big_high":   grp[big & high],
        "rsf_small_low":  grp[small & low],
        "rsf_big_low":    grp[big & low],
    }

    row = {}
    for name, leg in legs.items():
        if len(leg) < _MIN_STOCKS_PER_LEG:
            return None
        row[name] = _vwReturn(leg)

    row[RSF_FACTOR_COL] = (
        0.5 * (row["rsf_small_high"] + row["rsf_big_high"])
        - 0.5 * (row["rsf_small_low"] + row["rsf_big_low"])
    )
    row["n_stocks"] = int(sum(len(leg) for leg in legs.values()))
    return row


def _vwReturn(leg: pd.DataFrame) -> float:
    """Value-weighted holding-month return of one leg (formation-month caps)."""
    weights = leg[MARKET_CAP_COL] / leg[MARKET_CAP_COL].sum()
    return float((weights * leg[FWD_RETURN_COL]).sum())


def _validateFactorInputs(panel: pd.DataFrame):
    required = {"year_month", "ric", SAT_COL, MARKET_CAP_COL}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Monthly panel missing columns for RSF factor: {missing}")
