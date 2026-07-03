"""
build_quintiles.py — Assign ESG and SAT quintile ranks for the FF-25 sort.

Quintiles are assigned cross-sectionally within each month using all
eligible STOXX-universe stocks. No exchange-based breakpoint restriction
is applied (there is no NYSE equivalent in the European context).
"""

import pandas as pd
import numpy as np

from config.constants import (
    N_QUINTILES,
    MIN_HISTORY_MONTHS,
    ESG_COL,
    SAT_COL,
    ESG_QUINTILE_COL,
    SAT_QUINTILE_COL,
)

_ELIGIBILITY_WINDOW_MONTHS: int = max(MIN_HISTORY_MONTHS + 2, MIN_HISTORY_MONTHS)


def assignQuintiles(monthlyPanel: pd.DataFrame) -> pd.DataFrame:
    """
    Add PE and SAT quintile columns to the monthly panel.

    Quintile 1 = lowest PE / most bearish SAT.
    Quintile 5 = highest PE / most bullish SAT.

    Both sorts are independent: a stock's PE quintile is assigned without
    reference to its SAT quintile, and vice versa.

    Eligibility gate: rolling window approach. For each stock-month t, the
    stock is eligible only if it has at least MIN_HISTORY_MONTHS valid
    observations in the preceding _ELIGIBILITY_WINDOW_MONTHS months.

    This is strictly point-in-time:
      - New entrants to the STOXX 600 earn eligibility as they accumulate
        history and are excluded only during their initial ramp-up period.
      - Stocks that exit and re-enter (M&A, suspension, reconstitution) must
        re-earn eligibility from their re-entry date — not from 2009.
      - Sporadic data gaps (1-2 missing months) do not permanently exclude
        an otherwise well-observed stock.

    Args:
        monthlyPanel: Monthly panel with columns [year_month, ric, esg_score,
                      sat_monthly]. Rows with NaN in either sort variable are
                      excluded from that month's sort but do not count against
                      the stock's history.

    Returns:
        Panel with additional columns [esg_quintile, sat_quintile]. Ineligible
        stock-months are excluded so they do not inflate cross-sectional counts.
    """
    _validateQuintileInputs(monthlyPanel)

    panel = monthlyPanel.copy()

    both_nan = panel[ESG_COL].isna() & panel[SAT_COL].isna()
    dropped_nan = both_nan.sum()
    panel = panel[~both_nan]
    if dropped_nan > 0:
        print(f"  [quintiles] Dropped {dropped_nan} rows with NaN in both ESG and SAT")

    panel = panel.sort_values(["ric", "year_month"]).reset_index(drop=True)

    panel["_has_data"] = (~panel[ESG_COL].isna() | ~panel[SAT_COL].isna()).astype(int)

    panel["_month_int"] = panel["year_month"].apply(lambda p: p.year * 12 + p.ordinal % 12)

    def _rollingValidCount(group: pd.DataFrame) -> pd.Series:
        """
        For each row, count valid observations strictly before this month
        within the eligibility window.
        """
        months = group["_month_int"].values
        has_data = group["_has_data"].values
        counts = np.zeros(len(group), dtype=int)
        for i in range(len(group)):
            current = months[i]
            window_start = current - _ELIGIBILITY_WINDOW_MONTHS
            # Count prior months only (strictly before current)
            counts[i] = has_data[
                (months < current) & (months >= window_start)
            ].sum()
        return pd.Series(counts, index=group.index)

    prior_obs = (
        panel.groupby("ric", group_keys=False)
        .apply(_rollingValidCount, include_groups=False)
    )

    eligible_mask = prior_obs >= MIN_HISTORY_MONTHS
    dropped_history = (~eligible_mask).sum()
    panel = panel[eligible_mask].reset_index(drop=True)
    panel = panel.drop(columns=["_has_data", "_month_int"])

    if dropped_history > 0:
        print(
            f"  [quintiles] Excluded {dropped_history} stock-months below the "
            f"{MIN_HISTORY_MONTHS}-month history requirement "
            f"(rolling {_ELIGIBILITY_WINDOW_MONTHS}-month window); "
            f"{panel['ric'].nunique()} stocks have at least one eligible month"
        )

    panel[ESG_QUINTILE_COL] = _assignOneQuintile(panel, ESG_COL)
    panel[SAT_QUINTILE_COL] = _assignOneQuintile(panel, SAT_COL)

    return panel.dropna(subset=[ESG_QUINTILE_COL, SAT_QUINTILE_COL])


def assignPEQuintiles(monthlyPanel: pd.DataFrame) -> pd.DataFrame:
    """Assign only PE quintiles. Used in partial-sort robustness tests."""
    _validateColumn(monthlyPanel, ESG_COL)
    panel = monthlyPanel.copy()
    panel[ESG_QUINTILE_COL] = _assignOneQuintile(panel, ESG_COL)
    return panel


def assignSATQuintiles(monthlyPanel: pd.DataFrame) -> pd.DataFrame:
    """Assign only SAT quintiles. Used in partial-sort robustness tests."""
    _validateColumn(monthlyPanel, SAT_COL)
    panel = monthlyPanel.copy()
    panel[SAT_QUINTILE_COL] = _assignOneQuintile(panel, SAT_COL)
    return panel


def _assignOneQuintile(panel: pd.DataFrame, col: str) -> pd.Series:
    """
    Assign quintile ranks (1–5) cross-sectionally within each month.
    Ties are broken by averaging (method='average' in qcut).
    Months with fewer than N_QUINTILES distinct values return NaN.
    """
    def _rankOneMonth(series: pd.Series) -> pd.Series:
        if series.dropna().nunique() < N_QUINTILES:
            return pd.Series(np.nan, index=series.index)
        try:
            return pd.qcut(
                series,
                q=N_QUINTILES,
                labels=False,
                duplicates="drop",
            ) + 1  # Labels 1–5 rather than 0–4
        except ValueError:
            return pd.Series(np.nan, index=series.index)

    return panel.groupby("year_month")[col].transform(_rankOneMonth)


def _validateQuintileInputs(panel: pd.DataFrame):
    required = {"year_month", "ric", ESG_COL, SAT_COL}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Missing required columns for quintile assignment: {missing}")


def _validateColumn(panel: pd.DataFrame, col: str):
    if col not in panel.columns:
        raise ValueError(f"Column '{col}' not found in panel")