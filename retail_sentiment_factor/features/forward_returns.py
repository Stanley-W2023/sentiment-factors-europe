"""
forward_returns.py — Strictly aligned forward (t+1) returns for the monthly panel.

Single place where the sort-at-t / hold-at-t+1 convention is enforced.
All monthly modules (portfolio construction, Fama-MacBeth, parametric SAT)
use this helper so the alignment logic cannot drift between them.

The contiguity check matters: a per-stock shift(-1) on a panel with listing
gaps would silently pair month t characteristics with a return from month
t+k (the stock's next observed month). That return is still in the future,
so it is not look-ahead, but it misstates the holding horizon and pairs the
return with the wrong risk-free rate. Rows whose next observation is not
exactly one month ahead get NaN forward returns and drop out of the sort.
"""

import pandas as pd

from config.constants import RETURN_COL, RISK_FREE_COL

FWD_RETURN_COL = "ret_fwd"
FWD_RF_COL = "rf_fwd"


def addForwardMonthlyReturns(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add ret_fwd and rf_fwd columns: the stock's return and the risk-free
    rate in the calendar month immediately after year_month.

    Args:
        panel: Monthly panel with [ric, year_month, ret_eur, rf_eur].
               year_month must be Period[M].

    Returns:
        Copy of the panel with [ret_fwd, rf_fwd] columns. NaN where the
        stock has no observation in month t+1.
    """
    required = {"ric", "year_month", RETURN_COL, RISK_FREE_COL}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Panel missing columns for forward returns: {missing}")

    panel = panel.sort_values(["ric", "year_month"]).copy()
    grouped = panel.groupby("ric")

    panel[FWD_RETURN_COL] = grouped[RETURN_COL].shift(-1)
    panel[FWD_RF_COL] = grouped[RISK_FREE_COL].shift(-1)

    # Contiguity: the next row within the stock must be exactly month t+1
    month_ordinal = pd.PeriodIndex(panel["year_month"], freq="M").asi8
    next_ordinal = pd.Series(month_ordinal, index=panel.index).groupby(
        panel["ric"]
    ).shift(-1)
    contiguous = (next_ordinal - month_ordinal) == 1

    panel.loc[~contiguous, [FWD_RETURN_COL, FWD_RF_COL]] = float("nan")
    return panel
