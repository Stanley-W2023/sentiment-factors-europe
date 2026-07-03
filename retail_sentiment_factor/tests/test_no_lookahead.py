"""
test_no_lookahead.py — Regression tests against look-ahead bias.

The killer scenario: SAT is signed by month-t return direction, so any
portfolio that pairs a month-t sort with month-t returns is mechanically
profitable. These tests build panels where the sort variable is perfectly
correlated with the FORMATION-month return but future returns carry no
signal — any nonzero spread in that setup is look-ahead.

Covers: forward-return alignment, contiguity across listing gaps,
FF-25 portfolio construction, and the 2×3 RSF factor.
"""

import numpy as np
import pandas as pd
import pytest

from features.forward_returns import (
    addForwardMonthlyReturns,
    FWD_RETURN_COL,
    FWD_RF_COL,
)
from portfolios.construct_ff25 import constructFF25Portfolios
from factors.build_rsf_factor import buildRSFFactor, RSF_FACTOR_COL
from config.constants import (
    N_QUINTILES,
    RETURN_COL,
    MARKET_CAP_COL,
    RISK_FREE_COL,
    SAT_COL,
    PE_QUINTILE_COL,
    SAT_QUINTILE_COL,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

def _makeContaminationPanel(
    n_stocks: int = 250,
    n_months: int = 24,
    future_signal: float = 0.0,
    seed: int = 7,
) -> pd.DataFrame:
    """
    Panel where the month-t return is a deterministic function of the SAT
    quintile (mimicking SAT's mechanical same-month correlation), while
    the FOLLOWING month's return contains `future_signal` × sat_quintile
    plus nothing else. With future_signal=0, any measured SAT spread is
    look-ahead contamination by construction.
    """
    months = pd.period_range("2020-01", periods=n_months, freq="M")
    records = []
    for month in months:
        for stock_id in range(n_stocks):
            sat_q = (stock_id // N_QUINTILES) % N_QUINTILES + 1
            pe_q = stock_id % N_QUINTILES + 1
            # Same-month return: strongly increasing in SAT quintile
            contemporaneous_ret = 0.05 * (sat_q - 3)
            records.append({
                "year_month": month,
                "ric": f"STOCK_{stock_id:03d}",
                RETURN_COL: contemporaneous_ret,
                MARKET_CAP_COL: 1e9,
                RISK_FREE_COL: 0.0,
                PE_QUINTILE_COL: pe_q,
                SAT_QUINTILE_COL: sat_q,
                SAT_COL: float(sat_q - 3),
            })
    panel = pd.DataFrame(records)

    # Overwrite: return at month t = future_signal × sat_quintile at t-1.
    # With the deterministic layout above sat_q is constant per stock, so
    # the "next month" return is future_signal × sat_q + contemporaneous
    # part. To isolate cleanly, set the return to the contemporaneous part
    # PLUS the future component only when future_signal != 0.
    if future_signal != 0.0:
        panel[RETURN_COL] = panel[RETURN_COL] + future_signal * (
            panel[SAT_QUINTILE_COL] - 3
        )
    return panel


# ── forward return alignment ──────────────────────────────────────────────────

def test_addForwardMonthlyReturns_takes_next_month_value():
    # Arrange
    months = pd.period_range("2021-01", periods=3, freq="M")
    panel = pd.DataFrame({
        "year_month": months,
        "ric": "A",
        RETURN_COL: [0.01, 0.02, 0.03],
        RISK_FREE_COL: [0.001, 0.002, 0.003],
    })

    # Act
    result = addForwardMonthlyReturns(panel)

    # Assert — January's forward return is February's return
    assert result[FWD_RETURN_COL].iloc[0] == pytest.approx(0.02)
    assert result[FWD_RF_COL].iloc[0] == pytest.approx(0.002)
    # Last month has no forward return
    assert np.isnan(result[FWD_RETURN_COL].iloc[-1])


def test_addForwardMonthlyReturns_masks_gap_months():
    # Arrange — stock observed in Jan and Mar (Feb missing: suspension)
    panel = pd.DataFrame({
        "year_month": [pd.Period("2021-01", freq="M"),
                       pd.Period("2021-03", freq="M")],
        "ric": "A",
        RETURN_COL: [0.01, 0.99],
        RISK_FREE_COL: [0.0, 0.0],
    })

    # Act
    result = addForwardMonthlyReturns(panel)

    # Assert — January must NOT pick up March's return as its t+1 return
    assert np.isnan(result[FWD_RETURN_COL].iloc[0])


# ── portfolio construction ────────────────────────────────────────────────────

def test_ff25_contemporaneous_correlation_produces_no_spread():
    # Arrange — SAT quintile drives the SAME-month return; next-month
    # returns carry zero SAT signal (each stock repeats its deterministic
    # return, so return at t+1 equals return at t for every stock — the
    # spread across SAT quintiles in HOLDING-month returns is real only
    # if the code holds over t+1... here it is identical by symmetry).
    # The sharper test: zero out all returns after the first month.
    panel = _makeContaminationPanel(n_months=3)
    last_two = panel["year_month"] > pd.Period("2020-01", freq="M")
    panel.loc[last_two, RETURN_COL] = 0.0

    # Act — formation in Jan (contaminated returns), holding Feb/Mar (zero)
    result = constructFF25Portfolios(panel)

    # Assert — every portfolio return must be exactly zero: the Jan
    # returns that correlate with SAT may never leak into the output
    assert np.allclose(result["excess_ret"].values, 0.0, atol=1e-12)


def test_ff25_output_labelled_with_holding_month():
    # Arrange
    panel = _makeContaminationPanel(n_months=4)

    # Act
    result = constructFF25Portfolios(panel)

    # Assert — first possible holding month is formation start + 1
    assert result["year_month"].min() == pd.Period("2020-02", freq="M")
    assert result["year_month"].max() == pd.Period("2020-04", freq="M")


def test_ff25_genuine_future_signal_is_recovered():
    # Arrange — every stock's return is (0.05 + 0.01) × (sat_q − 3) in every
    # month (contemporaneous component plus genuine persistent signal), so
    # the holding-month spread between SAT Q5 and Q1 is 0.06 × 4 = 24%
    panel = _makeContaminationPanel(n_months=6, future_signal=0.01)

    # Act
    result = constructFF25Portfolios(panel)
    one_month = result[result["year_month"] == pd.Period("2020-03", freq="M")]
    q5 = one_month[
        (one_month[PE_QUINTILE_COL] == 1) & (one_month[SAT_QUINTILE_COL] == 5)
    ]["excess_ret"].iloc[0]
    q1 = one_month[
        (one_month[PE_QUINTILE_COL] == 1) & (one_month[SAT_QUINTILE_COL] == 1)
    ]["excess_ret"].iloc[0]

    # Assert
    assert q5 - q1 == pytest.approx(0.24, abs=1e-10)


# ── RSF factor ────────────────────────────────────────────────────────────────

def _makeFactorPanel(
    n_stocks: int = 200,
    n_months: int = 12,
    future_signal: float = 0.0,
    seed: int = 11,
) -> pd.DataFrame:
    """Panel for the 2×3 factor: continuous SAT, varying caps."""
    rng = np.random.default_rng(seed)
    months = pd.period_range("2020-01", periods=n_months, freq="M")
    caps = rng.uniform(1e8, 1e11, n_stocks)
    sat = rng.normal(0, 1, n_stocks)

    records = []
    for month in months:
        for stock_id in range(n_stocks):
            # Same-month return mechanically tied to SAT sign
            ret = 0.03 * np.sign(sat[stock_id]) + future_signal * sat[stock_id]
            records.append({
                "year_month": month,
                "ric": f"STOCK_{stock_id:03d}",
                RETURN_COL: ret,
                MARKET_CAP_COL: caps[stock_id],
                RISK_FREE_COL: 0.0,
                SAT_COL: sat[stock_id],
            })
    return pd.DataFrame(records)


def test_rsf_factor_zero_when_future_returns_are_zero():
    # Arrange — SAT drives month-t returns, but all returns after month 1
    # are zero: a clean factor must be exactly zero in every holding month
    panel = _makeFactorPanel(n_months=3)
    later = panel["year_month"] > pd.Period("2020-01", freq="M")
    panel.loc[later, RETURN_COL] = 0.0

    # Act
    factor = buildRSFFactor(panel)

    # Assert
    assert np.allclose(factor[RSF_FACTOR_COL].values, 0.0, atol=1e-12)


def test_rsf_factor_positive_when_high_sat_outperforms_next_month():
    # Arrange — SAT genuinely predicts next-month returns
    panel = _makeFactorPanel(n_months=6, future_signal=0.02)

    # Act
    factor = buildRSFFactor(panel)

    # Assert — high-SAT legs beat low-SAT legs in every holding month
    assert (factor[RSF_FACTOR_COL] > 0).all()


def test_rsf_factor_labelled_with_holding_month():
    # Arrange
    panel = _makeFactorPanel(n_months=4)

    # Act
    factor = buildRSFFactor(panel)

    # Assert
    assert factor["year_month"].min() == pd.Period("2020-02", freq="M")
    assert factor["year_month"].max() == pd.Period("2020-04", freq="M")
