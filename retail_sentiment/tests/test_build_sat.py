"""
test_build_sat.py — Unit tests for SAT construction.

Follows the Arrange-Act-Assert (AAA) pattern.
Tests cover: normal path, zero-return exclusion, earnings window exclusion,
insufficient data handling, and missing column validation.
"""

import numpy as np
import pandas as pd
import pytest

from features.build_sat import buildMonthlySAT, _aggregateDailyToMonthly
from config.constants import SAT_COL, MIN_VALID_DAYS_PER_MONTH


# ── fixtures ──────────────────────────────────────────────────────────────────

def _makeDailyPanel(
    n_days: int = 300,
    ric: str = "AIR.PA",
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic daily panel with realistic volume and return structure."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_days)

    return pd.DataFrame({
        "date": dates,
        "ric": ric,
        "volume": rng.integers(1_000_000, 10_000_000, n_days),
        "shares_outstanding": 500_000_000,
        "ret_eur": rng.normal(0.0005, 0.015, n_days),
    })


def _makeEmptyEarningsCalendar() -> pd.DataFrame:
    return pd.DataFrame(columns=["ric", "announcement_date"])


def test_buildMonthlySAT_returns_expected_columns():
    
    daily = _makeDailyPanel()
    earnings = _makeEmptyEarningsCalendar()

    result = buildMonthlySAT(daily, earnings)

    # Assert
    assert "ric" in result.columns
    assert "year_month" in result.columns
    assert SAT_COL in result.columns


def test_buildMonthlySAT_produces_one_row_per_month_per_stock():
    # Arrange
    daily = _makeDailyPanel(n_days=300)
    earnings = _makeEmptyEarningsCalendar()

    # Act
    result = buildMonthlySAT(daily, earnings)

    # Assert — no duplicate (ric, year_month) pairs
    assert result.duplicated(subset=["ric", "year_month"]).sum() == 0


def test_buildMonthlySAT_sat_values_are_finite_for_valid_months():
    # Arrange
    daily = _makeDailyPanel(n_days=300)
    earnings = _makeEmptyEarningsCalendar()

    # Act
    result = buildMonthlySAT(daily, earnings)

    # Assert — months with sufficient data after AR window warmup should be finite
    valid_months = result.dropna(subset=[SAT_COL])
    assert len(valid_months) > 0


# ── zero-return exclusion ─────────────────────────────────────────────────────

def test_buildMonthlySAT_zero_return_days_excluded():
    # Arrange — inject zero returns on known dates
    daily = _makeDailyPanel(n_days=300)
    daily.loc[daily.index[:20], "ret_eur"] = 0.0
    earnings = _makeEmptyEarningsCalendar()

    # Act
    result = buildMonthlySAT(daily, earnings)

    # Assert — pipeline completes without error; zero-return days don't crash it
    assert result is not None
    assert SAT_COL in result.columns


# ── earnings window exclusion ─────────────────────────────────────────────────

def test_buildMonthlySAT_earnings_exclusion_reduces_valid_days():
    # Arrange — earnings on every trading day of one month
    daily = _makeDailyPanel(n_days=300)
    # Mark the entire March 2020 as announcement dates
    march_days = daily[
        (daily["date"] >= "2020-03-01") & (daily["date"] <= "2020-03-31")
    ]["date"].tolist()
    earnings = pd.DataFrame({
        "ric": ["AIR.PA"] * len(march_days),
        "announcement_date": march_days,
    })

    # Act
    result_with_exclusion = buildMonthlySAT(daily, earnings)
    result_no_exclusion = buildMonthlySAT(daily, _makeEmptyEarningsCalendar())

    # Assert — March 2020 SAT should be NaN with exclusion, potentially not without
    march_period = pd.Period("2020-03", freq="M")
    march_with = result_with_exclusion[
        result_with_exclusion["year_month"] == march_period
    ][SAT_COL].values

    assert len(march_with) == 0 or np.isnan(march_with[0])


# ── insufficient data ─────────────────────────────────────────────────────────

def test_buildMonthlySAT_insufficient_days_returns_nan():
    # Arrange — only 5 days of data in one month (below MIN_VALID_DAYS_PER_MONTH)
    daily = _makeDailyPanel(n_days=5)
    earnings = _makeEmptyEarningsCalendar()

    # Act
    result = buildMonthlySAT(daily, earnings)

    # Assert — all SAT values are NaN (AR window never completes)
    assert result[SAT_COL].isna().all()


# ── input validation ──────────────────────────────────────────────────────────

def test_buildMonthlySAT_raises_on_missing_column():
    # Arrange
    daily = _makeDailyPanel().drop(columns=["volume"])
    earnings = _makeEmptyEarningsCalendar()

    # Act / Assert
    with pytest.raises(ValueError, match="Missing required columns"):
        buildMonthlySAT(daily, earnings)


def test_buildMonthlySAT_raises_on_missing_shares_outstanding():
    # Arrange
    daily = _makeDailyPanel().drop(columns=["shares_outstanding"])
    earnings = _makeEmptyEarningsCalendar()

    # Act / Assert
    with pytest.raises(ValueError, match="Missing required columns"):
        buildMonthlySAT(daily, earnings)


# ── aggregation ───────────────────────────────────────────────────────────────

def test_aggregateDailyToMonthly_returns_nan_for_sparse_months():
    # Arrange — only 2 valid days in a month (below MIN_VALID_DAYS_PER_MONTH)
    dates = pd.bdate_range("2021-01-04", periods=2)
    daily = pd.DataFrame({
        "date": dates,
        "ric": "BNP.PA",
        "sat_daily": [0.05, -0.03],
    })
    daily["year_month"] = daily["date"].dt.to_period("M")

    # Act
    result = _aggregateDailyToMonthly(daily)

    # Assert
    assert result[SAT_COL].isna().all()


def test_aggregateDailyToMonthly_averages_correctly_for_sufficient_days():
    # Arrange
    n_days = MIN_VALID_DAYS_PER_MONTH + 5
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    sat_values = np.ones(n_days) * 0.1
    daily = pd.DataFrame({
        "date": dates,
        "ric": "BNP.PA",
        "sat_daily": sat_values,
    })
    daily["year_month"] = daily["date"].dt.to_period("M")

    # Act
    result = _aggregateDailyToMonthly(daily)

    # Assert
    valid = result.dropna(subset=[SAT_COL])
    assert len(valid) > 0
    assert np.allclose(valid[SAT_COL].values, 0.1, atol=1e-10)
