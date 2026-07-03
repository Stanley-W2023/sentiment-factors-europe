"""
debug_main.py — Data quality diagnostics for the RSF pipeline.

Loads each stage from DuckDB and systematically checks for problems
that would corrupt downstream results. Run this after Step 3 completes
to understand what is in the data before trusting any estimation output.

Usage:
    python debug_main.py
"""

import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config.constants import (
    DB_PATH,
    TABLE_STOXX,
    TABLE_BENCHMARK,
    TABLE_GICS,
    SAMPLE_START,
    SAMPLE_END,
    REGIME_BREAK_DATE,
    ESG_COL,
    SAT_COL,
    ESG_QUINTILE_COL,
    SAT_QUINTILE_COL,
    RETURN_COL,
    MARKET_CAP_COL,
    RISK_FREE_COL,
    FF_FACTORS,
    N_QUINTILES,
    MAX_PRICE_FILL_DAYS,
    MIN_VALID_DAYS_PER_MONTH,
    EXCLUDED_GICS_SECTORS,
)
from data.db import Database

_SEP  = "=" * 70
_SEP2 = "-" * 70

def _header(title: str):
    print(f"\n{_SEP}")
    print(f"  {title}")
    print(_SEP)

def _ok(msg: str):   print(f"  [OK]   {msg}")
def _warn(msg: str): print(f"  [WARN] {msg}")
def _fail(msg: str): print(f"  [FAIL] {msg}")


# ── 1. RAW STOXX TABLE ────────────────────────────────────────────────────────

def checkRawStoxx(db: Database):
    _header("1. RAW STOXX TABLE")

    try:
        raw = db.readTable(TABLE_STOXX)
    except Exception as e:
        _fail(f"Could not read {TABLE_STOXX}: {e}")
        return None

    _ok(f"Table loaded: {raw.shape[0]:,} rows × {raw.shape[1]:,} columns")

    # Date range
    dates = pd.to_datetime(raw[("Date", "")], errors="coerce")
    _ok(f"Date range: {dates.min().date()} → {dates.max().date()}")
    if dates.min().date() > SAMPLE_START:
        _warn(f"Data starts after SAMPLE_START ({SAMPLE_START}) — missing early history")
    if dates.max().date() < REGIME_BREAK_DATE:
        _fail(f"Data ends before REGIME_BREAK_DATE ({REGIME_BREAK_DATE}) — regime test impossible")

    # RIC count
    rics = [c for c in raw.columns.get_level_values(0).unique() if c not in ("Date", "")]
    _ok(f"RICs in table: {len(rics)}")
    if len(rics) < 500:
        _warn(f"Only {len(rics)} RICs — expected ~600 for STOXX 600")

    # Fields per RIC
    sample_fields = list(raw[rics[0]].columns) if rics else []
    _ok(f"Fields per RIC: {sample_fields}")
    for expected in ["Total Return", "Volume", "Outstanding Shares",
                     "ESG Score", "Company Market Cap"]:
        if expected not in sample_fields:
            _fail(f"Missing expected field: '{expected}' — check STOXX_FIELDS in constants.py")
    if "GICS Sector Code" not in sample_fields:
        _warn("GICS Sector Code not present — financials cannot be excluded from PE sort")

    # ESG coverage
    pe_coverage = sum(
        raw[ric]["ESG Score"].notna().any()
        for ric in rics
        if "ESG Score" in raw[ric].columns
    )
    pe_pct = pe_coverage / len(rics) * 100
    msg = f"ESG data present for {pe_coverage}/{len(rics)} RICs ({pe_pct:.1f}%)"
    (_ok if pe_pct > 90 else _warn)(msg)

    # Volume zeros
    zero_vol = sum(
        (pd.to_numeric(raw[ric]["Volume"], errors="coerce") == 0).sum()
        for ric in rics
        if "Volume" in raw[ric].columns
    )
    if zero_vol > 0:
        _warn(f"{zero_vol:,} zero-volume observations across all RICs")
    else:
        _ok("No zero-volume observations")

    return raw


# ── 2. DAILY PANEL (post-reshape) ─────────────────────────────────────────────


def checkDailyPanel(raw):
    _header("2. DAILY PANEL (post _reshapeToLongFormat)")

    if raw is None:
        _fail("Skipped — raw STOXX not loaded")
        return None

    # Import here to avoid circular issues
    sys.path.insert(0, ".")
    from main import _reshapeToLongFormat

    try:
        daily = _reshapeToLongFormat(raw)
    except Exception as e:
        _fail(f"_reshapeToLongFormat failed: {e}")
        return None

    _ok(f"Daily panel: {len(daily):,} rows, {daily['ric'].nunique()} stocks")

    # Date coverage per stock
    obs_per_stock = daily.groupby("ric")["date"].count()
    _ok(f"Obs per stock — min: {obs_per_stock.min()}, "
        f"median: {obs_per_stock.median():.0f}, "
        f"max: {obs_per_stock.max()}")
    thin_stocks = (obs_per_stock < 252).sum()
    if thin_stocks > 0:
        _warn(f"{thin_stocks} stocks with < 252 daily observations (< 1 year)")

    # Return distribution
    rets = daily[RETURN_COL].dropna()
    _ok(f"Daily returns — mean: {rets.mean():.4f}, "
        f"std: {rets.std():.4f}, "
        f"min: {rets.min():.4f}, "
        f"max: {rets.max():.4f}")
    extreme = (rets.abs() > 0.5).sum()
    if extreme > 0:
        _warn(f"{extreme:,} daily returns with |ret| > 50% — possible data errors")

    # Turnover sanity
    daily["turnover"] = pd.to_numeric(daily["volume"], errors="coerce") / \
                        pd.to_numeric(daily["shares_outstanding"], errors="coerce")
    zero_turnover = (daily["turnover"] == 0).sum()
    nan_turnover  = daily["turnover"].isna().sum()
    if zero_turnover > 0:
        _warn(f"{zero_turnover:,} zero-turnover days (volume=0 after fill)")
    if nan_turnover > 0:
        _warn(f"{nan_turnover:,} NaN-turnover days")
    else:
        _ok("Turnover defined for all rows")

    # Forward-fill extent

    # ── Top extreme return observations (sanity check post-repull) ───────────
    extreme = (daily[RETURN_COL].dropna().abs() > 0.5).sum()
    if extreme > 0:
        _warn(f"{extreme:,} daily returns |ret| > 50% — investigate after repull")
        print(f"\n  Top 10 extreme daily return observations:")
        worst = (
            daily[["date", "ric", RETURN_COL]]
            .assign(abs_ret=daily[RETURN_COL].abs())
            .nlargest(10, "abs_ret")
            .drop(columns="abs_ret")
        )
        print(worst.to_string(index=False))
    else:
        _ok("No daily returns |ret| > 50% — return data looks clean")

    return daily




# ── 3. MONTHLY SAT ────────────────────────────────────────────────────────────

def checkMonthlySAT(db: Database):
    _header("3. MONTHLY SAT")

    try:
        panel = db.readTable("monthly_panel")
        panel["year_month"] = pd.PeriodIndex(panel["year_month"], freq="M")
    except Exception as e:
        _fail(f"Could not read monthly_panel: {e}")
        return None

    sat = panel[["ric", "year_month", SAT_COL]].copy()
    sat[SAT_COL] = pd.to_numeric(sat[SAT_COL], errors="coerce")

    total    = len(sat)
    nan_sat  = sat[SAT_COL].isna().sum()
    nan_pct  = nan_sat / total * 100
    msg = f"SAT: {nan_sat:,}/{total:,} NaN ({nan_pct:.1f}%)"
    (_ok if nan_pct < 10 else _warn)(msg)

    # SAT distribution
    s = sat[SAT_COL].dropna()
    _ok(f"SAT distribution — mean: {s.mean():.4f}, std: {s.std():.4f}, "
        f"p1: {s.quantile(0.01):.4f}, p99: {s.quantile(0.99):.4f}")

    extreme_sat = (s.abs() > 10).sum()
    if extreme_sat > 0:
        _warn(f"{extreme_sat:,} |SAT| > 10 — possible AR fit instability")

    # Coverage by year
    print(f"\n  SAT coverage by year (non-NaN stock-months):")
    sat["year"] = sat["year_month"].dt.year
    by_year = sat.groupby("year")[SAT_COL].agg(
        stocks=lambda x: x.notna().sum(),
        nan_pct=lambda x: x.isna().mean() * 100
    )
    print(by_year.to_string())

    return panel


# ── 4. MONTHLY ESG ─────────────────────────────────────────────────────────────

def checkMonthlyESG(panel: pd.DataFrame):
    _header("4. MONTHLY PE")

    if panel is None:
        _fail("Skipped — monthly_panel not loaded")
        return

    pe = pd.to_numeric(panel[ESG_COL], errors="coerce")
    total   = len(pe)
    nan_pe  = pe.isna().sum()
    nan_pct = nan_pe / total * 100
    msg = f"PE: {nan_pe:,}/{total:,} NaN ({nan_pct:.1f}%)"
    (_ok if nan_pct < 20 else _warn)(msg)

    # ESG distribution
    p = pe.dropna()
    _ok(f"PE distribution — mean: {p.mean():.1f}, median: {p.median():.1f}, "
        f"p1: {p.quantile(0.01):.1f}, p99: {p.quantile(0.99):.1f}")

    # Negative PE (should be zero after buildESGScore)
    neg_pe = (pe < 0).sum()
    if neg_pe > 0:
        _fail(f"{neg_pe:,} negative ESG values — buildESGScore exclusion not working")
    else:
        _ok("No negative ESG values")

    # Extreme PE
    extreme_pe = (pe > 100).sum()
    if extreme_pe > 0:
        _warn(f"{extreme_pe:,} ESG > 200 — possible data quality issue — review raw ESG values")

    # Coverage by stock
    pe_by_stock = panel.groupby("ric")[ESG_COL].apply(
        lambda x: pd.to_numeric(x, errors="coerce").notna().mean() * 100
    )
    zero_coverage = (pe_by_stock == 0).sum()
    low_coverage  = (pe_by_stock < 50).sum()
    if zero_coverage > 0:
        _fail(f"{zero_coverage} stocks with ZERO ESG observations — TR.TRESGScore not pulling for these")
    if low_coverage > 0:
        _warn(f"{low_coverage} stocks with < 50% ESG coverage")
    _ok(f"Median ESG coverage per stock: {pe_by_stock.median():.1f}%")

    # GICS sector exclusion
    if "gics_sector" in panel.columns:
        gics = pd.to_numeric(panel["gics_sector"], errors="coerce")
        fin_rows = gics.isin(EXCLUDED_GICS_SECTORS).sum()
        if fin_rows > 0:
            _fail(f"{fin_rows:,} financial sector rows still in panel — exclusion not applied")
        else:
            _ok("Financials correctly excluded from panel")
    else:
        _warn("gics_sector column not in panel — cannot verify financial exclusion")


# ── 5. QUINTILE DISTRIBUTION ──────────────────────────────────────────────────

def checkQuintiles(panel: pd.DataFrame):
    _header("5. QUINTILE DISTRIBUTION")

    if panel is None:
        _fail("Skipped — monthly_panel not loaded")
        return

    for col, name in [(ESG_QUINTILE_COL, "ESG"), (SAT_QUINTILE_COL, "SAT")]:
        q = pd.to_numeric(panel[col], errors="coerce")
        nan_pct = q.isna().mean() * 100
        dist = q.value_counts().sort_index()
        total = q.notna().sum()
        expected = total / N_QUINTILES

        print(f"\n  {name} quintile distribution (expected ~{expected:.0f} per bin):")
        for qval, count in dist.items():
            pct  = count / total * 100
            flag = " ← IMBALANCED" if abs(count - expected) / expected > 0.2 else ""
            print(f"    Q{int(qval)}: {count:6,} ({pct:.1f}%){flag}")
        if nan_pct > 0:
            _warn(f"{name} quintile NaN: {nan_pct:.1f}% of rows unassigned")

    # Joint distribution — check all 25 cells are populated
    print(f"\n  Stock-month counts per portfolio cell (PE row × SAT col):")
    pe  = pd.to_numeric(panel[ESG_QUINTILE_COL],  errors="coerce")
    sat = pd.to_numeric(panel[SAT_QUINTILE_COL], errors="coerce")
    joint = pd.crosstab(pe, sat)
    print(joint.to_string())

    empty_cells = (joint == 0).sum().sum()
    thin_cells  = (joint < 24).sum().sum()  # < 24 = fewer than 2 years of monthly obs
    if empty_cells > 0:
        _fail(f"{empty_cells} portfolio cells with ZERO observations")
    if thin_cells > 0:
        _warn(f"{thin_cells} portfolio cells with < 24 stock-month observations")
    else:
        _ok("All 25 portfolio cells have sufficient observations")


# ── 6. RETURN AND MARKET CAP ──────────────────────────────────────────────────

def checkReturnsAndMarketCap(panel: pd.DataFrame):
    _header("6. RETURNS AND MARKET CAP")

    if panel is None:
        _fail("Skipped — monthly_panel not loaded")
        return

    # Monthly returns
    rets = pd.to_numeric(panel[RETURN_COL], errors="coerce")
    nan_ret = rets.isna().mean() * 100
    msg = f"Monthly returns: {nan_ret:.1f}% NaN"
    (_ok if nan_ret < 5 else _warn)(msg)

    r = rets.dropna()
    _ok(f"Return distribution — mean: {r.mean():.4f}, std: {r.std():.4f}, "
        f"min: {r.min():.4f}, max: {r.max():.4f}")

    extreme = (r.abs() > 0.5).sum()
    if extreme > 0:
        _warn(f"{extreme:,} monthly returns |ret| > 50%")

    # Market cap
    me = pd.to_numeric(panel[MARKET_CAP_COL], errors="coerce")
    nan_me = me.isna().mean() * 100
    msg = f"Market cap: {nan_me:.1f}% NaN"
    (_ok if nan_me < 5 else _warn)(msg)

    m = me.dropna()
    _ok(f"Market cap (EUR) — median: {m.median():,.0f}, "
        f"min: {m.min():,.0f}, max: {m.max():,.0f}")

    zero_me = (me <= 0).sum()
    if zero_me > 0:
        _fail(f"{zero_me:,} zero/negative market cap rows — value-weighting will break")

    # Risk-free rate
    rf = pd.to_numeric(panel[RISK_FREE_COL], errors="coerce")
    _ok(f"Risk-free rate — mean: {rf.mean():.6f}, "
        f"min: {rf.min():.6f}, max: {rf.max():.6f}")
    if (rf == 0.0003).all():
        _fail("Risk-free rate is the hardcoded placeholder (0.0003) — "
              "FF factors RF not merged correctly")
    if (rf < 0).sum() > 0:
        _warn(f"{(rf < 0).sum():,} negative risk-free rate months (plausible post-2012 EURIBOR)")


# ── 7. PORTFOLIO CONSTRUCTION ─────────────────────────────────────────────────

def checkPortfolios(db: Database):
    _header("7. PORTFOLIO CONSTRUCTION")

    try:
        port = db.readTable("portfolio_returns")
        port["year_month"] = pd.PeriodIndex(port["year_month"], freq="M")
    except Exception as e:
        _fail(f"Could not read portfolio_returns: {e}")
        return

    _ok(f"Portfolio returns: {len(port):,} rows")

    # Coverage — all 25 cells across all months?
    months = port["year_month"].nunique()
    expected_rows = months * N_QUINTILES * N_QUINTILES
    _ok(f"Months: {months}, expected rows: {expected_rows:,}, actual: {len(port):,}")
    if len(port) < expected_rows * 0.9:
        _warn(f"Missing {expected_rows - len(port):,} portfolio-month observations "
              f"({(expected_rows - len(port))/expected_rows*100:.1f}% of cells empty)")

    # Portfolio size distribution
    if "n_stocks" in port.columns:
        n = pd.to_numeric(port["n_stocks"], errors="coerce")
        print(f"\n  Stocks per portfolio-month:")
        print(f"    mean: {n.mean():.1f}, median: {n.median():.0f}, "
              f"min: {n.min():.0f}, max: {n.max():.0f}")
        thin = (n < 5).sum()
        if thin > 0:
            _warn(f"{thin:,} portfolio-months with < 5 stocks — spreads unreliable")

        # Per-cell average stock count
        print(f"\n  Mean stock count per portfolio cell (PE row × SAT col):")
        pivot = port.pivot_table(
            values="n_stocks", index=ESG_QUINTILE_COL,
            columns=SAT_QUINTILE_COL, aggfunc="mean"
        )
        print(pivot.round(1).to_string())

    # Excess return distribution
    er = pd.to_numeric(port["excess_ret"], errors="coerce").dropna()
    _ok(f"Excess returns — mean: {er.mean():.4f}, std: {er.std():.4f}, "
        f"min: {er.min():.4f}, max: {er.max():.4f}")

    extreme = (er.abs() > 0.5).sum()
    if extreme > 0:
        _fail(f"{extreme:,} portfolio excess returns |ret| > 50% — "
              f"almost certainly a data error upstream")

    # 5×5 mean return matrix
    print(f"\n  Mean monthly excess return per cell (PE row × SAT col):")
    matrix = port.pivot_table(
        values="excess_ret", index=ESG_QUINTILE_COL,
        columns=SAT_QUINTILE_COL, aggfunc="mean"
    )
    print(matrix.round(4).to_string())

    # Regime split
    regime_break = pd.Period(str(REGIME_BREAK_DATE)[:7], freq="M")
    pre  = port[port["year_month"] <= regime_break]
    post = port[port["year_month"] >  regime_break]
    _ok(f"Pre-2020 months: {pre['year_month'].nunique()}, "
        f"Post-2020 months: {post['year_month'].nunique()}")
    if post["year_month"].nunique() < 24:
        _warn("Fewer than 24 post-2020 months — Chow test will have low power")


# ── GICS DIAGNOSTIC ──────────────────────────────────────────────────────────

def checkGICS(db: Database):
    _header("GICS SECTOR DIAGNOSTIC")

    # 1. Raw table presence — case-insensitive
    all_tables = db.getTableNames()
    print(f"  All tables in db: {all_tables}")

    gics_match = [t for t in all_tables if t.lower() == TABLE_GICS.lower()]
    if not gics_match:
        _fail(f"GICS table not found. TABLE_GICS='{TABLE_GICS}', tables={all_tables}")
        return
    actual_name = gics_match[0]
    _ok(f"GICS table found as '{actual_name}' (TABLE_GICS='{TABLE_GICS}')")

    # 2. Raw contents
    try:
        gics_raw = db.readTable(actual_name)
    except Exception as e:
        _fail(f"Could not read GICS table: {e}")
        return

    _ok(f"GICS table: {len(gics_raw):,} rows × {len(gics_raw.columns)} columns")
    _ok(f"Columns: {list(gics_raw.columns)}")
    print(f"\n  First 5 rows:")
    print(gics_raw.head().to_string(index=False))

    # 3. Check rename will work
    rename_map = {
        "Instrument":       "ric",
        "GICS Sector Code": "gics_sector",
        "GICSSectorCode":   "gics_sector",
        "gics_sector_code": "gics_sector",
    }
    renamed = gics_raw.rename(columns=rename_map)
    if "ric" not in renamed.columns:
        _fail(f"Cannot find 'ric' column after rename — raw columns: {list(gics_raw.columns)}")
        return
    if "gics_sector" not in renamed.columns:
        _fail(f"Cannot find 'gics_sector' column after rename — raw columns: {list(gics_raw.columns)}")
        return
    _ok("Column rename to (ric, gics_sector) will succeed")

    # 4. Sector code distribution
    gics = renamed[["ric", "gics_sector"]].copy()
    gics["gics_sector"] = pd.to_numeric(gics["gics_sector"], errors="coerce")
    n_null = gics["gics_sector"].isna().sum()
    if n_null > 0:
        _warn(f"{n_null} RICs have null GICS sector code")

    print(f"\n  Sector code distribution:")
    dist = gics["gics_sector"].value_counts().sort_index()
    for code, count in dist.items():
        marker = " ← WILL BE EXCLUDED" if code in EXCLUDED_GICS_SECTORS else ""
        print(f"    Sector {int(code):>4}: {count:>4} RICs{marker}")

    n_financials = gics["gics_sector"].isin(EXCLUDED_GICS_SECTORS).sum()
    _ok(f"{n_financials} financials (sector 40) will be excluded from PE sort")

    # 5. Simulate the join against monthly_panel
    try:
        panel = db.readTable("monthly_panel")
        panel_rics = set(panel["ric"].unique())
        gics_rics  = set(gics["ric"].dropna().unique())
        matched    = panel_rics & gics_rics
        unmatched  = panel_rics - gics_rics
        _ok(f"Join simulation: {len(matched)}/{len(panel_rics)} panel RICs matched in GICS table")
        if unmatched:
            _warn(f"{len(unmatched)} panel RICs have no GICS code: {sorted(unmatched)[:10]}{'...' if len(unmatched) > 10 else ''}")
        if "gics_sector" in panel.columns:
            _ok("gics_sector already present in monthly_panel — join already ran correctly")
        else:
            _warn("gics_sector NOT in monthly_panel — rebuild features with Step 3 to apply join")
    except Exception as e:
        _warn(f"Could not simulate join against monthly_panel: {e}")

    # 6. Trace hasTable() result
    has = db.hasTable(TABLE_GICS)
    _ok(f"db.hasTable('{TABLE_GICS}') = {has}")
    resolved = db.resolveTableName(TABLE_GICS)
    _ok(f"db.resolveTableName('{TABLE_GICS}') = '{resolved}'")


# ── 8. FACTORS ────────────────────────────────────────────────────────────────

def checkFactors(db: Database):
    _header("8. FF FACTORS")

    try:
        factors = db.readTable("ff_factors_europe")
    except Exception as e:
        _fail(f"Could not read ff_factors_europe: {e}")
        return

    _ok(f"Factor table: {len(factors):,} rows")
    _ok(f"Columns: {list(factors.columns)}")

    # Check all required factors present
    missing = [f for f in FF_FACTORS if f not in factors.columns]
    if missing:
        _fail(f"Missing factor columns: {missing}")
    else:
        _ok(f"All FF6 factors present: {FF_FACTORS}")

    # Factor distributions
    print(f"\n  Daily factor summary:")
    for col in FF_FACTORS:
        if col in factors.columns:
            s = pd.to_numeric(factors[col], errors="coerce").dropna()
            print(f"    {col:8s}: mean={s.mean():.4f}, std={s.std():.4f}, "
                  f"min={s.min():.4f}, max={s.max():.4f}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print(_SEP)
    print("  RSF PIPELINE — DATA QUALITY DIAGNOSTICS")
    print(f"  Sample: {SAMPLE_START} → {SAMPLE_END}")
    print(f"  Regime break: {REGIME_BREAK_DATE}")
    print(_SEP)

    with Database(DB_PATH) as db:
        tables = db.getTableNames()
        print(f"\n  Tables in database: {tables}")

        checkGICS(db)
        raw   = checkRawStoxx(db)
        daily = checkDailyPanel(raw)
        panel = checkMonthlySAT(db)

        checkMonthlyESG(panel)
        checkQuintiles(panel)
        checkReturnsAndMarketCap(panel)
        checkPortfolios(db)
        checkFactors(db)

    print(f"\n{_SEP}")
    print("  DIAGNOSTICS COMPLETE")
    print(_SEP)


if __name__ == "__main__":
    if sys.version_info[:2] < (3, 12):
        raise RuntimeError("Requires Python 3.12+")
    main()