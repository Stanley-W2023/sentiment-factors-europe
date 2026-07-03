"""
fetch_french_factors.py — Download European Fama-French 5-factor + momentum returns.

Source: Ken French Data Library (Europe datasets).
Factors are read from CSV, parsed, and stored in DuckDB.
Requires local CSV files or accessible paths — no Refinitiv dependency.
"""
from datetime import date
from pathlib import Path

import pandas as pd

from data.db import Database
from config.constants import SAMPLE_START, SAMPLE_END

_EUROPE_URL = Path(r"Retail_Sentiment_Factor/data/Europe_5_Factors_Daily.csv")
_MOM_EUROPE_URL = Path(r"Retail_Sentiment_Factor/data/Europe_MOM_Factor_Daily.csv")

TABLE_FF_EUROPE = "ff_factors_europe"


def _read_ff_daily_csv(path: Path) -> pd.DataFrame:
    """
    Read a Fama-French 'daily' CSV:
      - Parse first column as a daily Datetime column named 'date'
      - Strip/standardize column names
      - Coerce numeric columns
      - Drop entirely empty columns
    """
    df = pd.read_csv(path, skipinitialspace=True)
    first = df.columns[0]
    df = df.rename(columns={first: "date"})
    df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(), errors="coerce", format=None)
    df = df[df["date"].notna()]
    df.columns = [c.strip().replace(" ", "_").replace("-", "_") for c in df.columns]

    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(axis=1, how="all")
    return df


def _filterSamplePeriod(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows within SAMPLE_START to SAMPLE_END (inclusive). Expects a DatetimeIndex."""
    start = pd.Timestamp(SAMPLE_START)
    end = pd.Timestamp(SAMPLE_END)
    return df[(df.index >= start) & (df.index <= end)]


def fetchEuropeanFactors(db: Database):
    """
    Download and store European FF5 + momentum factor returns (daily).
    Stores one table: ff_factors_europe, filtered to SAMPLE_START–SAMPLE_END.
    """
    print("\nFetching European FF5 factors...")

    europe_ff5 = _read_ff_daily_csv(_EUROPE_URL)
    europe_mom = _read_ff_daily_csv(_MOM_EUROPE_URL)

    mom_col = "WML" 

    europe = europe_ff5.merge(europe_mom[["date", mom_col]].rename(columns={mom_col: "UMD"}),
                              on="date", how="left")

    europe = europe.set_index("date").sort_index()
    europe = _filterSamplePeriod(europe)


    db.writeTable(TABLE_FF_EUROPE, europe.reset_index())
    print(f"  Stored {TABLE_FF_EUROPE}: {len(europe):,} rows")