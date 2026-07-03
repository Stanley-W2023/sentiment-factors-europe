"""
main.py — RSF Research Pipeline Orchestrator.

Runs the full pipeline in five steps. Each step is independently
togglable via the flags below. Safe to rerun: Refinitiv pulls are
incremental, feature builds are idempotent.

Steps:
  1. PULL_DATA        — Incremental Refinitiv pull to DuckDB
  2. FETCH_FACTORS    — Download Ken French European FF5+Mom factors
  3. BUILD_FEATURES   — Compute SAT, PE, and quintile assignments
  4. BUILD_PORTFOLIOS — Construct FF-25 double-sort portfolios
  5. RUN_ESTIMATION   — FF6 alphas, Fama-MacBeth, Chow test

Requires Python 3.12 and a valid Refinitiv Eikon session for Step 1.
"""

import sys
import pandas as pd
import os
        
from config.constants import (
    DB_PATH,
    STOXX_INDEX_RIC,
    STOXX50_INDEX_RIC,
    SAMPLE_START,
    SAMPLE_END,
    REGIME_BREAK_DATE,
    TABLE_STOXX,
    TABLE_BENCHMARK,
    TABLE_GICS,
    RETURN_COL,
    MARKET_CAP_COL,
    RISK_FREE_COL,
    PE_COL,
    SAT_COL,
    MAX_PRICE_FILL_DAYS,
    FF_FACTORS,
)
from data.db import Database
from data.fetch_refinitiv import RefinitivClient
from data.fetch_french_factors import (
    fetchEuropeanFactors,
    TABLE_FF_EUROPE,
)
from features.build_sat import buildMonthlySAT
from features.build_pe import buildTrailingPE
from features.build_quintiles import assignQuintiles
from portfolios.construct_ff25 import (
    constructFF25Portfolios,
    constructFF25PortfoliosEW,
    filterSmallMidCap,
    computeSpreads,
    computeRollingDeltaCross,
    buildRSFLongShort,
)
from estimation.time_series_alpha import estimateTimeSeriesAlpha
from estimation.fama_macbeth import estimateFamaMacBeth, computeChowTest
from estimation.parametric_sat import estimateParametricSATAmplification
from features.build_daily_panel import buildDailyAnalysisPanel
from portfolios.construct_daily_ff25 import (
    constructAllDailyPortfolios,
    computeDailySpreads,
    buildDailyRSFLongShort,
    computeRollingDailyDeltaCross,
    printDailySpreadSummary,
)
from estimation.daily_fmb import estimateDailyFMB, estimateDailyFMBRegimes
from factors.build_rsf_factor import buildRSFFactor, RSF_FACTOR_COL
from estimation.factor_spanning import (
    runSpanningRegression,
    computeFactorCorrelations,
    compareAlphasWithRSF,
    printSpanningSummary,
)
from plot import generateAllPlots

IMPORT_STATIC    = False
PULL_DATA        = False
FETCH_FACTORS    = False
BUILD_FEATURES   = False
BUILD_PORTFOLIOS = False
RUN_ESTIMATION   = False
RUN_SPANNING     = False  # Step 8: RSF as a complementary FF-style factor
GENERATE_PLOTS   = False
BUILD_DAILY      = False

PULL_CONFIG = {
    "indices": [STOXX_INDEX_RIC, STOXX50_INDEX_RIC],
    "start":   SAMPLE_START,
    "end":     SAMPLE_END,
}


def main():
    print("RSF Research Pipeline")
    print(f"Sample: {SAMPLE_START} to {SAMPLE_END}")
    print(f"Regime break: {REGIME_BREAK_DATE}")
    print("-" * 60)

    with Database(DB_PATH) as db:
        if IMPORT_STATIC:
            _pullstatic(db)
            
        if PULL_DATA:
            _stepPullData(db)

        if FETCH_FACTORS:
            _stepFetchFactors(db)

        if BUILD_FEATURES:
            monthly_panel, daily_panel = _stepBuildFeatures(db)
        else:
            monthly_panel  = _loadMonthlyPanel(db)
            daily_panel    = None   # rebuilt on demand inside Step 7

        if BUILD_PORTFOLIOS:
            portfolio_returns = _stepBuildPortfolios(db, monthly_panel)
        else:
            portfolio_returns = db.readTable("portfolio_returns")
            portfolio_returns["year_month"] = pd.PeriodIndex(portfolio_returns["year_month"], freq="M")

        if RUN_ESTIMATION:
            alphas, fmb_results, parametric_results = _stepRunEstimation(
                db, portfolio_returns, monthly_panel
            )
        else:
            alphas             = db.readTable("ff6_alphas")
            fmb_results        = db.readTable("fmb_results")
            parametric_results = None

        if GENERATE_PLOTS:
            _stepGeneratePlots(
                db, monthly_panel, portfolio_returns,
                alphas, fmb_results, parametric_results,
            )

        if RUN_SPANNING:
            _stepRunFactorSpanning(db, monthly_panel, portfolio_returns)

        if BUILD_DAILY:
            _stepBuildDailyAnalysis(db, monthly_panel, daily_panel)

    print("\nPipeline complete.")


def _stepRunFactorSpanning(
    db: Database,
    monthlyPanel: pd.DataFrame,
    portfolioReturns: pd.DataFrame,
):
    """
    Step 8: Build RSF as an FF-style 2×3 (size × SAT) factor and test it
    as a complementary right-hand-side regressor:
      - spanning regression of RSF on FF5+UMD (is RSF redundant?)
      - correlations with the FF factors
      - 25-portfolio alphas under FF6 vs FF6+RSF (does RSF strip
        sentiment comovement out of the cross-section?)
    """
    print("\n[Step 8] RSF factor construction and FF spanning tests...")

    factors = _loadMonthlyFactors(db)

    print("  Building 2x3 size/SAT RSF factor...")
    rsf_factor = buildRSFFactor(monthlyPanel)
    db.writeTable("rsf_factor", rsf_factor)
    print(f"  RSF factor: {len(rsf_factor)} months, "
          f"mean {rsf_factor[RSF_FACTOR_COL].mean()*100:.3f}%/month, "
          f"ann. Sharpe {rsf_factor[RSF_FACTOR_COL].mean() / rsf_factor[RSF_FACTOR_COL].std() * (12 ** 0.5):.2f}")

    print("  Running spanning regression (RSF on FF5+UMD)...")
    spanning = runSpanningRegression(rsf_factor, factors)
    db.writeTable("rsf_spanning", spanning)

    correlations = computeFactorCorrelations(rsf_factor, factors)
    db.writeTable("rsf_correlations", correlations)

    print("  Comparing FF6 vs FF6+RSF alphas on the 25 portfolios...")
    comparison, summary = compareAlphasWithRSF(
        portfolioReturns, factors, rsf_factor
    )
    db.writeTable("alpha_comparison_rsf", comparison)
    db.writeTable("alpha_comparison_summary", summary)

    printSpanningSummary(spanning, correlations, summary)
    print("Step 8 complete.")


def _loadMonthlyFactors(db: Database) -> pd.DataFrame:
    """
    Load the daily Ken French European factors from DuckDB and aggregate
    to monthly. Factor returns are summed within the month (standard
    approximation for daily long-short percent returns); RF is compounded.
    """
    factors = db.readTable(TABLE_FF_EUROPE)

    date_col = "Date" if "Date" in factors.columns else "date"
    factors[date_col] = pd.to_datetime(factors[date_col])
    factors["year_month"] = factors[date_col].dt.to_period("M")
    for col in factors.columns:
        if col not in (date_col, "year_month"):
            factors[col] = pd.to_numeric(factors[col], errors="coerce")

    rf_col = "RF" if "RF" in factors.columns else "rf"
    factors = (
        factors.groupby("year_month")
        [FF_FACTORS + [rf_col]]
        .sum()
        .reset_index()
        .rename(columns={rf_col: RISK_FREE_COL})
    )
    for col in FF_FACTORS:
        factors[col] = factors[col] / 100.0
    # Ensure year_month is Period dtype — groupby may return it as object
    if not isinstance(factors["year_month"].dtype, pd.PeriodDtype):
        factors["year_month"] = pd.PeriodIndex(factors["year_month"], freq="M")
    return factors

def _stepBuildDailyAnalysis(
    db: Database,
    monthlyPanel: pd.DataFrame,
    dailyPanelRaw: pd.DataFrame | None = None,
):
    """
    Step 7: Build daily FF-25 portfolios and daily FMB for all three SAT variants.

    This step is compute-intensive (~15-30 min for the full 2009-2025 panel)
    because it runs the AR(2) residual computation per stock per day and then
    loops over ~4,000 daily cross-sections for FMB. Run once and save results.

    Args:
        dailyPanelRaw : The flat long-format daily panel from _reshapeToLongFormat.
                        If None (e.g. BUILD_FEATURES=False), it is rebuilt from the
                        raw STOXX table in DuckDB.

    Microstructure caveat: bid-ask bounce and single-day data noise are not
    corrected. The rolling SAT variants (5d, 10d) are more robust than raw.
    See limitations section in the paper.
    """
    print("\n[Step 7] Building daily analysis...")

    # Use the already-flattened daily panel from Step 3 if available;
    # otherwise rebuild it from the raw DB table (slow, ~2 min read).
    if dailyPanelRaw is not None:
        print("  Using daily panel from Step 3 (already in memory).")
        daily_raw = dailyPanelRaw
    else:
        print("  Rebuilding daily panel from DB (BUILD_FEATURES=False)...")
        raw_stoxx = db.readTable(TABLE_STOXX)
        daily_raw = _reshapeToLongFormat(raw_stoxx)

    # Monthly RF is needed to approximate daily RF
    monthly_panel_with_rf = monthlyPanel[
        ["year_month", RISK_FREE_COL]
    ].drop_duplicates("year_month")

    daily_panel = buildDailyAnalysisPanel(daily_raw, monthly_panel_with_rf)

    print("\n  Building daily portfolios (VW, 3 SAT variants)...")
    daily_portfolios = constructAllDailyPortfolios(daily_panel)

    print("\n  Computing daily spread statistics...")
    daily_spreads = computeDailySpreads(daily_portfolios)
    printDailySpreadSummary(daily_spreads)

    print("\n  Computing rolling 60-day delta_Cross...")
    daily_rolling_dc = computeRollingDailyDeltaCross(daily_portfolios)

    print("\n  Running daily Fama-MacBeth (full sample)...")
    daily_fmb = estimateDailyFMB(daily_panel)

    print("\n  Running daily Fama-MacBeth (pre/post-2020 regime split)...")
    daily_fmb_regimes = estimateDailyFMBRegimes(daily_panel)

    # Persist results to DuckDB
    db.writeTable("daily_spreads",     daily_spreads)
    db.writeTable("daily_rolling_dc",  daily_rolling_dc)
    db.writeTable("daily_fmb",         daily_fmb)
    db.writeTable("daily_fmb_regimes", daily_fmb_regimes)

    # Generate daily plots if GENERATE_PLOTS is also True
    if GENERATE_PLOTS:
        # Load monthly results for full plot suite
        portfolio_returns = db.readTable("portfolio_returns")
        portfolio_returns["year_month"] = pd.PeriodIndex(
            portfolio_returns["year_month"], freq="M"
        )
        alphas      = db.readTable("ff6_alphas")
        fmb_results = db.readTable("fmb_results")
        rsf_ls      = db.readTable("rsf_longshort")
        rsf_ls["year_month"] = pd.PeriodIndex(rsf_ls["year_month"], freq="M")
        spreads_full = computeSpreads(portfolio_returns)
        portfolio_returns_smc = db.readTable("portfolio_returns_smc")
        portfolio_returns_smc["year_month"] = pd.PeriodIndex(
            portfolio_returns_smc["year_month"], freq="M"
        )
        spreads_smc = computeSpreads(portfolio_returns_smc)
        rolling_dc  = computeRollingDeltaCross(portfolio_returns)
        portfolio_returns_ew = db.readTable("portfolio_returns_ew")
        portfolio_returns_ew["year_month"] = pd.PeriodIndex(
            portfolio_returns_ew["year_month"], freq="M"
        )
        plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
        generateAllPlots(
            portfolioReturns    = portfolio_returns,
            portfolioReturnsEW  = portfolio_returns_ew,
            spreads             = spreads_full,
            spreadsSmallMid     = spreads_smc,
            rollingDeltaCross   = rolling_dc,
            rsfLongShort        = rsf_ls,
            alphaResults        = alphas,
            fmbResults          = fmb_results,
            parametricResults   = None,
            dailySpreads        = daily_spreads,
            dailyFMBResults     = daily_fmb,
            dailyRollingDC      = daily_rolling_dc,
            outDir              = plots_dir,
        )

    print("Step 7 complete.")


def _pullstatic(db: Database):
    print("\n[Step 0] Pulling Refinitiv static...")
    client = RefinitivClient()
    if not client.connect():
        raise RuntimeError(
            "Could not connect to Refinitiv. "
        )
    client.pullGICSSnapshot(db)
    print("Step 0 complete.")

def _stepPullData(db: Database):
    print("\n[Step 1] Pulling Refinitiv data...")
    client = RefinitivClient()
    if not client.connect():
        raise RuntimeError(
            "Could not connect to Refinitiv. "
        )
    client.pullIndicesToDb(PULL_CONFIG, db)
    print("Step 1 complete.")

def _stepFetchFactors(db: Database):
    print("\n[Step 2] Fetching Ken French European factors...")
    fetchEuropeanFactors(db)
    print("Step 2 complete.")


def _stepBuildFeatures(db: Database) -> pd.DataFrame:
    print("\n[Step 3] Building SAT, PE, and quintile features...")

    raw_stoxx = db.readTable(TABLE_STOXX)

    daily_panel = _reshapeToLongFormat(raw_stoxx)

    earnings_calendar = pd.DataFrame(columns=["ric", "announcement_date"])

    print("  Computing monthly SAT...")
    monthly_sat = buildMonthlySAT(daily_panel, earnings_calendar)

    print("  Cleaning trailing PE...")
    pe_panel = _buildMonthlyPEPanel(daily_panel, db=db)
    pe_clean = buildTrailingPE(pe_panel)

    print("  Merging and assigning quintiles...")
 
    pe_clean = pe_clean.drop(columns=["me_eur"], errors="ignore")
    merged = monthly_sat.merge(pe_clean, on=["year_month", "ric"], how="inner")
    merged = _addReturnAndMarketCap(merged, daily_panel)
    merged = _addRiskFreeRate(merged, db)
    panel_with_quintiles = assignQuintiles(merged)

    db.writeTable("monthly_panel", panel_with_quintiles)
    print(f"  Monthly panel: {len(panel_with_quintiles)} rows, "
          f"{panel_with_quintiles['ric'].nunique()} stocks")
    print("Step 3 complete.")

    return panel_with_quintiles, daily_panel


def _stepBuildPortfolios(
    db: Database,
    monthlyPanel: pd.DataFrame,
) -> pd.DataFrame:
    print("\n[Step 4] Constructing FF-25 portfolios...")

    # Primary: value-weighted
    portfolio_returns = constructFF25Portfolios(monthlyPanel)
    db.writeTable("portfolio_returns", portfolio_returns)

    spreads = computeSpreads(portfolio_returns)
    print("\n  Summary spread statistics (VW, with NW t-stats):")
    print(spreads.to_string(index=False))

    rsf_ls = buildRSFLongShort(portfolio_returns)
    db.writeTable("rsf_longshort", rsf_ls)
    print(f"\n  RSF long-short mean monthly return: "
          f"{rsf_ls['rsf_return'].mean():.4f}")

    # Robustness: equal-weighted
    print("\n  Computing equal-weighted portfolios (robustness)...")
    portfolio_returns_ew = constructFF25PortfoliosEW(monthlyPanel)
    db.writeTable("portfolio_returns_ew", portfolio_returns_ew)
    spreads_ew = computeSpreads(portfolio_returns_ew)
    delta_cross_ew = spreads_ew[spreads_ew["stat"] == "delta_cross"]["mean"].iloc[0]
    print(f"  EW delta_cross: {delta_cross_ew:.4f}")

    # Robustness: small/mid-cap subsample
    print("\n  Computing small/mid-cap subsample (robustness)...")
    panel_smc = filterSmallMidCap(monthlyPanel)
    portfolio_returns_smc = constructFF25Portfolios(panel_smc)
    db.writeTable("portfolio_returns_smc", portfolio_returns_smc)
    spreads_smc = computeSpreads(portfolio_returns_smc)
    delta_cross_smc = spreads_smc[spreads_smc["stat"] == "delta_cross"]["mean"].iloc[0]
    print(f"  Small/mid-cap delta_cross: {delta_cross_smc:.4f}")

    print("Step 4 complete.")
    return portfolio_returns


def _stepRunEstimation(
    db: Database,
    portfolioReturns: pd.DataFrame,
    monthlyPanel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n[Step 5] Running estimation...")

    factors = _loadMonthlyFactors(db)

    print("  Estimating FF6 alphas for all 25 portfolios...")
    alphas = estimateTimeSeriesAlpha(portfolioReturns, factors)
    db.writeTable("ff6_alphas", alphas)
    print(f"  Alpha range: [{alphas['alpha'].min():.4f}, {alphas['alpha'].max():.4f}]")

    print("  Running Fama-MacBeth regressions...")
    fmb_results = estimateFamaMacBeth(monthlyPanel, portfolioReturns)
    db.writeTable("fmb_results", fmb_results)
    print("\n  Fama-MacBeth results:")
    print(fmb_results.to_string(index=False))

    print("\n  Running Chow test (pre vs post-2020)...")
    regime_break = pd.Period(str(REGIME_BREAK_DATE)[:7], freq="M")
    chow_results = computeChowTest(monthlyPanel, regime_break, portfolioReturns)
    db.writeTable("chow_results", chow_results)
    print("\n  Chow test results:")
    print(chow_results.to_string(index=False))

    print("\n  Running parametric SAT amplification by PE regime...")
    parametric_results = estimateParametricSATAmplification(monthlyPanel)

    print("Step 5 complete.")
    return alphas, fmb_results, parametric_results


def _stepGeneratePlots(
    db: Database,
    monthlyPanel: pd.DataFrame,
    portfolioReturns: pd.DataFrame,
    alphas: pd.DataFrame,
    fmbResults: pd.DataFrame,
    parametricResults: dict | None = None,
):
    print("\n[Step 6] Generating plots...")
    plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")

    portfolio_returns_ew  = db.readTable("portfolio_returns_ew")
    portfolio_returns_ew["year_month"] = pd.PeriodIndex(
        portfolio_returns_ew["year_month"], freq="M"
    )
    portfolio_returns_smc = db.readTable("portfolio_returns_smc")
    portfolio_returns_smc["year_month"] = pd.PeriodIndex(
        portfolio_returns_smc["year_month"], freq="M"
    )
    rsf_ls = db.readTable("rsf_longshort")
    rsf_ls["year_month"] = pd.PeriodIndex(rsf_ls["year_month"], freq="M")

    spreads_full    = computeSpreads(portfolioReturns)
    spreads_smc     = computeSpreads(portfolio_returns_smc)
    rolling_dc      = computeRollingDeltaCross(portfolioReturns)

    generateAllPlots(
        portfolioReturns    = portfolioReturns,
        portfolioReturnsEW  = portfolio_returns_ew,
        spreads             = spreads_full,
        spreadsSmallMid     = spreads_smc,
        rollingDeltaCross   = rolling_dc,
        rsfLongShort        = rsf_ls,
        alphaResults        = alphas,
        fmbResults          = fmbResults,
        parametricResults   = parametricResults,
        outDir              = plots_dir,
    )
    print("Step 6 complete.")

def _reshapeToLongFormat(rawStoxx: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape the MultiIndex STOXX table from wide (Date × RIC||Field)
    to long (date, ric, volume, shares_outstanding, ret_eur, pe_raw, me_eur, gics_sector).

    TR.TotalReturn is used directly as ret_eur — it is already adjusted for
    splits, dividends, and currency redenominations by Refinitiv, so no
    pct_change(), no price cleaning, and no outlier screening are needed.

    Volume, shares_outstanding, and me_eur are forward-filled up to
    MAX_PRICE_FILL_DAYS to cover sporadic Refinitiv reporting gaps.
    Rows where ret_eur is NaN after filling (pre/post-listing) are dropped.
    """
    if not isinstance(rawStoxx.columns, pd.MultiIndex):
        raise ValueError(
            "Expected MultiIndex columns from STOXX table. "
            "Check db.readTable() is restoring the MultiIndex correctly."
        )

    rics = rawStoxx.columns.get_level_values(0).unique().tolist()
    rics = [r for r in rics if r not in ("Date", "")]

    parts = []
    for ric in rics:
        try:
            ric_df = rawStoxx[ric].copy()
            ric_df["ric"] = ric
            ric_df["date"] = rawStoxx[("Date", "")]
            ric_df = ric_df.rename(columns={
           
                "Total Return":                    "ret_eur",
                "Volume":                          "volume",
                "Outstanding Shares":              "shares_outstanding",
                "SharesOut":                       "shares_outstanding",
                "P/E (Daily Time Series Ratio)":   "pe_raw",
                "PE":                              "pe_raw",
                "Company Market Cap":              "me_eur",
                "CompanyMarketCap":                "me_eur",
                "GICS Sector Code":                "gics_sector",
                "GICSSectorCode":                  "gics_sector",
            })
            parts.append(ric_df)
        except KeyError:
            continue

    long_df = pd.concat(parts, ignore_index=True)
    long_df["date"] = pd.to_datetime(long_df["date"])
    long_df = long_df.sort_values(["ric", "date"]).reset_index(drop=True)

    long_df = long_df[long_df["date"] >= pd.Timestamp(SAMPLE_START)].copy()

    for col in ("ret_eur", "volume", "shares_outstanding", "pe_raw", "me_eur", "gics_sector"):
        if col in long_df.columns:
            long_df[col] = pd.to_numeric(long_df[col], errors="coerce")

    if "ret_eur" in long_df.columns:
        long_df["ret_eur"] = long_df["ret_eur"] / 100.0

 
    for col in ("shares_outstanding", "me_eur", "volume"):
        if col in long_df.columns:
            long_df[col] = (
                long_df.groupby("ric")[col]
                .transform(lambda s: s.ffill(limit=MAX_PRICE_FILL_DAYS))
            )

    clean = long_df.dropna(subset=["ret_eur", "volume", "shares_outstanding"])
    dropped_rows = len(long_df) - len(clean)
    if dropped_rows > 0:
        affected = long_df[
            long_df[["ret_eur", "volume", "shares_outstanding"]].isna().any(axis=1)
        ]["ric"].nunique()
        print(
            f"  [reshape] Dropped {dropped_rows:,} rows with no return data "
            f"across {affected} stocks (pre/post-listing)"
        )

    return clean


def _buildMonthlyPEPanel(dailyPanel: pd.DataFrame, db: "Database | None" = None) -> pd.DataFrame:
    """Extract monthly PE and GICS sector code from the pre-reshaped daily panel.
    
    If gics_sector is not present in the daily panel (because TR.GICSSectorCode
    was not in the time-series pull), falls back to the static GICS snapshot
    table pulled by pullGICSSnapshot() via rd.get_data.
    """
    long_df = dailyPanel.copy()
    long_df["year_month"] = pd.to_datetime(long_df["date"]).dt.to_period("M")

    agg_cols = {"pe_raw": "last", "me_eur": "last"}
    if "gics_sector" in long_df.columns:
        agg_cols["gics_sector"] = "last"

    monthly = (
        long_df.sort_values("date")
        .groupby(["ric", "year_month"])
        .agg(**{k: (k, v) for k, v in agg_cols.items()})
        .reset_index()
    )

    # If gics_sector not in daily panel, join from the static GICS snapshot table.
    # rd.get_data returns columns named "Instrument" and "GICS Sector Code" —
    # we normalise both names defensively.
    if "gics_sector" not in monthly.columns and db is not None:
        if db.hasTable(TABLE_GICS):
            try:
                gics_raw = db.readTable(db.resolveTableName(TABLE_GICS))
                # Normalise column names — handle both rd.get_data output and
                # any already-renamed versions
                gics_raw = gics_raw.rename(columns={
                    "Instrument":       "ric",
                    "GICS Sector Code": "gics_sector",
                    "GICSSectorCode":   "gics_sector",
                    "gics_sector_code": "gics_sector",
                })
                if "ric" not in gics_raw.columns or "gics_sector" not in gics_raw.columns:
                    raise ValueError(f"Unexpected GICS table columns: {list(gics_raw.columns)}")
                gics = gics_raw[["ric", "gics_sector"]].copy()
                gics["gics_sector"] = pd.to_numeric(gics["gics_sector"], errors="coerce")
                monthly = monthly.merge(gics, on="ric", how="left")
                n_mapped = monthly["gics_sector"].notna().sum()
                n_financials = (monthly["gics_sector"] == 40).sum()
                print(f"  [PE panel] Joined GICS from snapshot: {n_mapped:,} stock-months mapped, "
                      f"{n_financials:,} financials (sector 40) will be excluded")
            except Exception as e:
                print(f"  [PE panel] GICS join failed: {e}")

    if "gics_sector" not in monthly.columns:
        monthly["gics_sector"] = -1
        print(
            "  [PE panel] WARNING: gics_sector not found — "
            "financials will NOT be excluded. Run Step 0 to pull GICS snapshot."
        )

    return monthly


def _addReturnAndMarketCap(
    panel: pd.DataFrame,
    dailyPanel: pd.DataFrame,
) -> pd.DataFrame:
    """Merge monthly compounded returns and month-end market cap into the feature panel."""
    long_df = dailyPanel.copy()
    long_df["year_month"] = pd.to_datetime(long_df["date"]).dt.to_period("M")

    def _compoundReturns(s: pd.Series) -> float:
        valid = s.dropna()
        if valid.empty:
            return float("nan")
        return float((1 + valid).prod() - 1)

    monthly = (
        long_df.sort_values("date")
        .groupby(["ric", "year_month"])
        .agg(
            ret_eur=(RETURN_COL, _compoundReturns),
            me_eur=("me_eur", "last"),
        )
        .reset_index()
    )
    return panel.merge(monthly, on=["ric", "year_month"], how="left")


def _addRiskFreeRate(
    panel: pd.DataFrame,
    db: Database,
) -> pd.DataFrame:
    """
    Add the monthly EUR risk-free rate from the Ken French factors table.
    The FF data carries a daily RF (in percent); we compound to monthly
    and express as a decimal before merging onto the stock panel.
    """
    from data.fetch_french_factors import TABLE_FF_EUROPE

    factors = db.readTable(TABLE_FF_EUROPE)

    date_col = "Date" if "Date" in factors.columns else "date"
    factors[date_col] = pd.to_datetime(factors[date_col])
    factors["year_month"] = factors[date_col].dt.to_period("M")

    rf_col = "RF" if "RF" in factors.columns else "rf"
    monthly_rf = (
        factors.groupby("year_month")[rf_col]
        .apply(lambda s: (1 + s / 100).prod() - 1)
        .reset_index()
        .rename(columns={rf_col: RISK_FREE_COL})
    )

    return panel.merge(monthly_rf, on="year_month", how="left")


def _loadMonthlyPanel(db: Database) -> pd.DataFrame:
    """Load the pre-built monthly panel from DuckDB."""
    df = db.readTable("monthly_panel")
    df["year_month"] = pd.PeriodIndex(df["year_month"], freq="M")
    return df


if __name__ == "__main__":
    if sys.version_info[:2] < (3, 12):
        raise RuntimeError("RSF pipeline requires Python 3.12+")
    main()