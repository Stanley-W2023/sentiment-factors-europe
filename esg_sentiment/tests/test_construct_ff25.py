"""
test_construct_ff25.py — Unit tests for FF-25 portfolio construction.

Covers: value weighting, spread computation, long-short portfolio,
input validation, and edge cases (single stock, zero market cap).
"""

import numpy as np
import pandas as pd
import pytest

from portfolios.construct_ff25 import (
    constructFF25Portfolios,
    computeSpreads,
    buildRSFLongShort,
)
from config.constants import (
    N_QUINTILES,
    ESG_QUINTILE_COL,
    SAT_QUINTILE_COL,
    RETURN_COL,
    MARKET_CAP_COL,
    RISK_FREE_COL,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

def _makeMonthlyPanel(
    n_stocks: int = 100,
    n_months: int = 24,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic monthly panel with all required columns."""
    rng = np.random.default_rng(seed)
    months = pd.period_range("2020-01", periods=n_months, freq="M")

    records = []
    for month in months:
        for stock_id in range(n_stocks):
            records.append({
                "year_month": month,
                "ric": f"STOCK_{stock_id:03d}",
                RETURN_COL: rng.normal(0.005, 0.06),
                MARKET_CAP_COL: rng.uniform(1e8, 1e11),
                RISK_FREE_COL: 0.0003,
                ESG_QUINTILE_COL: (stock_id % N_QUINTILES) + 1,
                SAT_QUINTILE_COL: ((stock_id // N_QUINTILES) % N_QUINTILES) + 1,
            })

    return pd.DataFrame(records)


# ── portfolio construction ────────────────────────────────────────────────────

def test_constructFF25Portfolios_returns_25_portfolios_per_month():
    # Arrange
    panel = _makeMonthlyPanel(n_stocks=250)

    # Act
    result = constructFF25Portfolios(panel)

    # Assert
    n_portfolios_per_month = (
        result.groupby("year_month")[[ESG_QUINTILE_COL, SAT_QUINTILE_COL]]
        .nunique()
        .prod(axis=1)
    )
    assert (n_portfolios_per_month == N_QUINTILES ** 2).all()


def test_constructFF25Portfolios_excess_returns_are_finite():
    # Arrange
    panel = _makeMonthlyPanel(n_stocks=250)

    # Act
    result = constructFF25Portfolios(panel)

    # Assert
    assert result["excess_ret"].isna().sum() == 0
    assert np.isfinite(result["excess_ret"].values).all()


def test_constructFF25Portfolios_value_weights_sum_to_one():
    # Arrange — use a small known panel where we can check weights manually
    month = pd.Period("2021-01", freq="M")
    panel = pd.DataFrame({
        "year_month": [month, month],
        "ric": ["A", "B"],
        RETURN_COL: [0.01, 0.02],
        MARKET_CAP_COL: [100.0, 300.0],  # Weights: 0.25 and 0.75
        RISK_FREE_COL: [0.0, 0.0],
        ESG_QUINTILE_COL: [1, 1],
        SAT_QUINTILE_COL: [1, 1],
    })

    # Act
    result = constructFF25Portfolios(panel)

    # Assert — expected VW return = 0.25*0.01 + 0.75*0.02 = 0.0175
    expected_ret = 0.0175
    assert abs(result["excess_ret"].iloc[0] - expected_ret) < 1e-10


# ── spread computation ────────────────────────────────────────────────────────

def test_computeSpreads_returns_expected_stats():
    # Arrange
    panel = _makeMonthlyPanel(n_stocks=250, n_months=36)
    portfolios = constructFF25Portfolios(panel)

    # Act
    spreads = computeSpreads(portfolios)

    # Assert
    stat_names = spreads["stat"].tolist()
    assert "delta_cross" in stat_names
    assert any("delta_esg" in s for s in stat_names)
    assert any("delta_sat" in s for s in stat_names)


def test_computeSpreads_delta_cross_is_scalar():
    # Arrange
    panel = _makeMonthlyPanel(n_stocks=250, n_months=36)
    portfolios = constructFF25Portfolios(panel)

    # Act
    spreads = computeSpreads(portfolios)
    delta_cross = spreads[spreads["stat"] == "delta_cross"]["value"].values

    # Assert
    assert len(delta_cross) == 1
    assert np.isfinite(delta_cross[0])


# ── long-short portfolio ──────────────────────────────────────────────────────

def test_buildRSFLongShort_returns_one_row_per_month():
    # Arrange
    panel = _makeMonthlyPanel(n_stocks=250, n_months=24)
    portfolios = constructFF25Portfolios(panel)
    n_months = panel["year_month"].nunique()

    # Act
    rsf = buildRSFLongShort(portfolios)

    # Assert
    assert len(rsf) == n_months
    assert "rsf_return" in rsf.columns


# ── input validation ──────────────────────────────────────────────────────────

def test_constructFF25Portfolios_raises_on_missing_column():
    # Arrange
    panel = _makeMonthlyPanel().drop(columns=[MARKET_CAP_COL])

    # Act / Assert
    with pytest.raises(ValueError, match="Missing required columns"):
        constructFF25Portfolios(panel)


def test_constructFF25Portfolios_handles_zero_market_cap_gracefully():
    # Arrange — one portfolio with all-zero market caps
    panel = _makeMonthlyPanel(n_stocks=50)
    panel.loc[panel[ESG_QUINTILE_COL] == 1, MARKET_CAP_COL] = 0.0

    # Act
    result = constructFF25Portfolios(panel)

    # Assert — PE quintile 1 portfolios return NaN, not raise
    pe1 = result[result[ESG_QUINTILE_COL] == 1]
    assert pe1["excess_ret"].isna().all()
