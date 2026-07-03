"""
constants.py — Single source of truth for all named constants in the RSF pipeline.

No magic numbers anywhere in analysis code. Every parameter that could
reasonably change between runs lives here.
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
    "TR.PE",
    "TR.SharesOutstanding",
    "TR.CompanyMarketCap"
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

DB_PATH: str = str(Path(__file__).parent.parent / "data" / "rsf.duckdb")
TABLE_STOXX: str = "STOXX"
TABLE_BENCHMARK: str = "STOXX50E"
TABLE_GICS: str = "GICS"
SAMPLE_START: date = date(2009, 6, 1)
SAMPLE_END: date = date(2025, 12, 31)
REGIME_BREAK_DATE: date = date(2019, 12, 31)   # Pre vs post-2020 split

N_QUINTILES: int = 5
RETURN_COL: str = "ret_eur"
MARKET_CAP_COL: str = "me_eur"
RISK_FREE_COL: str = "rf_eur"
SAT_COL: str = "sat_monthly"
PE_COL: str = "pe_trailing"
PE_QUINTILE_COL: str = "pe_quintile"
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

PE_WINSOR_LOWER: float = 0.01
PE_WINSOR_UPPER: float = 0.99

FF_FACTORS: list[str] = ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "UMD"]
NEWEY_WEST_LAGS: int = 6
MIN_OBS_FOR_REGRESSION: int = 24

CHOW_CONTROLS: list[str] = ["vix_europe", "mkt_return", "eurocoin"]

FPE_SCALE_INIT: float = 1.0
FPE_PEAK_PE_INIT: float = 30.0
FPE_ASYMMETRY_INIT: float = 2.5
FPE_PE_MIN_INIT: float = 5.0
FPE_DELTA_INIT: float = 0.1
FPE_PE_STAR_INIT: float = 15.0

T_STAT_THRESHOLD_HARVEY: float = 3.0            # Harvey et al. (2016)
T_STAT_THRESHOLD_BONFERRONI: float = 2.5        # Bonferroni for 3 primary tests