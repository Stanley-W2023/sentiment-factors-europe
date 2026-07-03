"""
reset_db.py — Drop all or specific tables to start fresh.

Usage:
    python reset_db.py                  # Drop STOXX raw table only (forces re-pull with new fields)
    python reset_db.py --all            # Drop every table in the database
    python reset_db.py --derived        # Drop derived tables only (keep raw STOXX data)
    python reset_db.py --purge          # Delete pre-sample rows from raw tables
"""

import sys
import os
from data.db import Database
from config.constants import DB_PATH, TABLE_STOXX, TABLE_BENCHMARK, SAMPLE_START

RAW_TABLES = [TABLE_STOXX, TABLE_BENCHMARK]

DERIVED_TABLES = [
    "monthly_panel",
    "portfolio_returns",
    "esf_longshort",
    "ff6_alphas",
    "fmb_results",
    "chow_results",
    "daily_spreads",
    "daily_rolling_dc",
    "daily_fmb",
    "daily_fmb_regimes",
]


def resetTables(table_names: list[str]):
    with Database(DB_PATH) as db:
        db.dropTables(table_names)
    print(f"Done. Rerun main.py to pull fresh data.")


def resetAllTables():
    with Database(DB_PATH) as db:
        db.dropAllTables()
    print("All tables dropped. Rerun main.py to pull fresh data.")


def purgePreSampleRows():
    """
    Delete rows before SAMPLE_START from the raw Refinitiv tables.
    Refinitiv's 0#.STOXX constituent expansion ignores the start date
    parameter and returns data from each stock's earliest available
    date — this cleans up the resulting pre-sample rows in-place
    without requiring a full repull.
    """
    cutoff = str(SAMPLE_START)
    with Database(DB_PATH) as db:
        for table in RAW_TABLES:
            db.deleteRowsBefore(table, cutoff)
    print(f"Done. Pre-sample rows (before {cutoff}) removed from {RAW_TABLES}.")


if __name__ == "__main__":
    if "--all" in sys.argv:
        resetAllTables()
    elif "--derived" in sys.argv:
        resetTables(DERIVED_TABLES)
    elif "--purge" in sys.argv:
        purgePreSampleRows()
    else:
        # Default: drop STOXX raw table so the next PULL_DATA=True run
        # re-fetches all fields including TR.TRESGScore
        print("Dropping raw STOXX table to force full re-pull with TR.TRESGScore...")
        resetTables(RAW_TABLES)
