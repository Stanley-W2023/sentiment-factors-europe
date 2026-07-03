"""
constants.py — Single source of truth for all named constants in the ESF pipeline.

No magic numbers anywhere in analysis code. Every parameter that could
reasonably change between runs lives here.

ESF (ESG Sentiment Factor) adaptation of the RSF pipeline:
  - PE replaced by Refinitiv ESG composite score (TR.TRESGScore)
  - Two regime breaks: Dec-2017 (EU Action Plan) and Dec-2019 (retail surge)
  - Weibull f(ESG) bounds rescaled for 0-100 score range
"""

from datetime import date
from pathlib import Path

STOXX_INDEX_RIC: str = "0#.STOXX"
STOXX50_INDEX_RIC: str = ".STOXX50E"
EXCLUDED_GICS_SECTORS: list[int] = [40]
MIN_HISTORY_MONTHS: int = 6
CURRENCY: str = "EUR"

STOXX_FIELDS: list[str] = [
    "TR.TotalReturn",
    "TR.Volume",
    "TR.SharesOutstanding",
    "TR.CompanyMarketCap",
    "TR.TRESGScore",    # ESG composite score (0-100), added for ESF
]

STATIC_FIELDS: list[str] = [ 
    "TR.GICSSectorCode"
]

BENCHMARK_FIELDS: list[str] = [
    "TR.TotalReturnIndex",  
    "TR.Volume"
]

STOXX_BATCH_DAYS: int = 50
BENCHMARK_BATCH_DAYS: int = 100

DB_PATH: str = str(Path(__file__).parent.parent / "data" / "esf.duckdb")
TABLE_STOXX: str = "STOXX"
TABLE_BENCHMARK: str = "STOXX50E"
TABLE_GICS: str = "GICS"
SAMPLE_START: date = date(2009, 6, 1)
SAMPLE_END: date = date(2025, 12, 31)

# Two predetermined regime breaks:
#   2018 — EU Sustainable Finance Action Plan (institutional ESG mainstreaming)
#   2020 — Retail participation surge + SFDR implementation
# Using Dec year-end for both so the indicator flips at Jan 1 of the named year.
# Both are theory-driven and fixed before any subsample results were examined.
REGIME_BREAK_DATE: date = date(2019, 12, 31)       # legacy alias — kept for daily modules
REGIME_BREAK_DATE_2018: date = date(2017, 12, 31)  # Post-2018 indicator starts Jan 2018
REGIME_BREAK_DATE_2020: date = date(2019, 12, 31)  # Post-2020 indicator starts Jan 2020

N_QUINTILES: int = 5
RETURN_COL: str = "ret_eur"
MARKET_CAP_COL: str = "me_eur"
RISK_FREE_COL: str = "rf_eur"
SAT_COL: str = "sat_monthly"

# ── ESG sort variable (replaces PE in the double sort) ────────────────────────
ESG_COL: str = "esg_score"          # cleaned Refinitiv composite score
ESG_QUINTILE_COL: str = "esg_quintile"
ESG_WINSOR_LOWER: float = 0.01
ESG_WINSOR_UPPER: float = 0.99
ESG_MIN_COVERAGE: int = 30          # minimum stocks with ESG data per month

SAT_QUINTILE_COL: str = "sat_quintile"

# Maximum consecutive trading days a price gap is forward-filled before the
# row is dropped entirely. 10 days ≈ two calendar weeks, covering public
# holidays and sporadic Refinitiv outages. Gaps longer than this are treated
# as delistings or extended suspensions and excluded from the daily panel.
# A stock must then accumulate enough NaN months to fall below the
# MIN_HISTORY_MONTHS threshold in the rolling eligibility window before
# being excluded from the monthly sort entirely.
MAX_PRICE_FILL_DAYS: int = 10

TURNOVER_AR_ORDER: int = 2                      # AR(2) for ATV residuals
TURNOVER_ROLLING_WINDOW_DAYS: int = 252         # One trading year
MIN_VALID_DAYS_PER_MONTH: int = 10              # Minimum daily obs for monthly SAT
EARNINGS_EXCLUSION_WINDOW_DAYS: int = 3         # Days around announcement to drop
MIN_TURNOVER_CLIP: float = 1e-10                # Floor for log(turnover)


FF_FACTORS: list[str] = ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "UMD"]
NEWEY_WEST_LAGS: int = 6
MIN_OBS_FOR_REGRESSION: int = 24

CHOW_CONTROLS: list[str] = ["vix_europe", "mkt_return", "eurocoin"]

# ── Weibull f(ESG) initial parameters and bounds ─────────────────────────────
# ESG scores run 0-100 (vs PE which ran 5-150), so peak and scale are rescaled.
# peak_esg ~ 50 (moderate ESG is where retail amplification concentrates)
# pe_min → esg_min ~ 5 (very low scores are value-trap equivalent)
FPE_SCALE_INIT: float = 1.0
FPE_PEAK_PE_INIT: float = 50.0     # peak_esg initialised at midpoint of 0-100
FPE_ASYMMETRY_INIT: float = 2.5
FPE_PE_MIN_INIT: float = 5.0       # esg_min: floor of meaningful ESG scores
FPE_DELTA_INIT: float = 0.1
FPE_PE_STAR_INIT: float = 20.0     # esg_star: below this, value-trap penalty applies

T_STAT_THRESHOLD_HARVEY: float = 3.0            # Harvey et al. (2016)
T_STAT_THRESHOLD_BONFERRONI: float = 2.5        # Bonferroni for 3 primary tests