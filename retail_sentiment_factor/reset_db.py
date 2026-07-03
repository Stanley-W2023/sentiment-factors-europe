"""
reset_db.py — Drop all or specific tables to start fresh.

Usage:
    python reset_db.py                  # Drop STOXX and STOXX50E only
    python reset_db.py --all            # Drop every table in the database
"""

import sys
import os
from data.db import Database
from config.constants import TABLE_STOXX, TABLE_BENCHMARK, SAMPLE_START

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "rsf.duckdb")

RAW_TABLES = [TABLE_STOXX, TABLE_BENCHMARK]

DERIVED_TABLES = [
    "monthly_panel",
    "portfolio_returns",
    "rsf_longshort",
    "ff6_alphas",
    "fmb_results",
    "chow_results",
]

## DEFAULT_TABLES = RAW_TABLES + DERIVED_TABLES


def resetTables(table_names: list[str], db_path: str = DB_PATH):
    with Database(db_path) as db:
        db.dropTables(table_names)
    print(f"Done. Rerun main.py to pull fresh data.")


def resetAllTables(db_path: str = DB_PATH):
    with Database(db_path) as db:
        db.dropAllTables()
    print("All tables dropped. Rerun main.py to pull fresh data.")


def purgePreSampleRows(db_path: str = DB_PATH):
    """
    Delete rows before SAMPLE_START from the raw Refinitiv tables.
    Refinitiv's 0#.STOXX constituent expansion ignores the start date
    parameter and returns data from each stock's earliest available
    date — this cleans up the resulting pre-sample rows in-place
    without requiring a full repull.

    Usage:
        python reset_db.py --purge
    """
    cutoff = str(SAMPLE_START)
    with Database(db_path) as db:
        for table in RAW_TABLES:
            db.deleteRowsBefore(table, cutoff)
    print(f"Done. Pre-sample rows (before {cutoff}) removed from {RAW_TABLES}.")

if __name__ == "__main__":
    if "--all" in sys.argv:
        resetAllTables()
    elif "--purge" in sys.argv:
        purgePreSampleRows()
    else:
        resetTables(DEFAULT_TABLES)