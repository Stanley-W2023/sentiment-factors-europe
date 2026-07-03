"""
fetch_refinitiv.py — Incremental Refinitiv pull for STOXX constituents.

Adapted from pull_master_data.py. Gap-safe design preserved:
  - On failure, the pull stops immediately rather than skipping batches.
  - db.lastDate() always reflects a contiguous block, so reruns retry
    the exact failed batch with no gaps introduced.
  - Each batch is upserted immediately after fetch — no data lost on crash.
"""

import refinitiv.data as rd
import pandas as pd
from datetime import date, timedelta

from data.db import Database
from config.constants import (
    STOXX_FIELDS,
    STATIC_FIELDS,
    BENCHMARK_FIELDS,
    STOXX_BATCH_DAYS,
    BENCHMARK_BATCH_DAYS,
    STOXX_INDEX_RIC,
    STOXX50_INDEX_RIC,
    TABLE_STOXX,
    TABLE_BENCHMARK,
    TABLE_GICS,
)

pd.set_option("future.no_silent_downcasting", True)


class RefinitivClient:
    """
    Fetches STOXX constituent data from Refinitiv with incremental DuckDB upsert.

    First run  : full pull from config['start'] to config['end'].
    Subsequent : resumes from db.lastDate() — only missing days are fetched.
    """

    def __init__(self):
        self._session = None

    def connect(self) -> bool:
        """Open a Refinitiv session. Returns True on success."""
        try:
            self._session = rd.open_session()
            print("Refinitiv connection: OK")
            return True
        except Exception as e:
            print(f"Refinitiv connection failed: {e}")
            return False

    def pullIndicesToDb(self, config: dict, db: Database):
        """
        Incrementally pull and upsert STOXX and benchmark data.

        Args:
            config: dict with keys 'indices' (list of RICs), 'start', 'end'.
            db    : open Database instance.
        """
        self._validateConfig(config)

        indices      = config["indices"]
        config_start = config["start"]
        config_end   = config["end"]

        for ric in indices:
            table_name = self._ricToTableName(ric)
            is_benchmark = ric == STOXX50_INDEX_RIC
            fields = BENCHMARK_FIELDS if is_benchmark else STOXX_FIELDS
            batch_days = BENCHMARK_BATCH_DAYS if is_benchmark else STOXX_BATCH_DAYS

            self._pullOneIndex(
                ric, table_name, fields, batch_days,
                config_start, config_end, db
            )


    def _pullOneIndex(
        self,
        ric: str,
        table_name: str,
        fields: list[str],
        batch_days: int,
        config_start: date,
        config_end: date,
        db: Database,
    ):
        """Pull one RIC incrementally, stopping on any batch failure."""
        last_stored = db.lastDate(table_name)

        if last_stored is None:
            pull_start = config_start
            print(f"\n[{table_name}] First run — pulling {pull_start} to {config_end}")
        else:
            pull_start = (last_stored + timedelta(days=1)).date()
            if pull_start > config_end:
                print(f"\n[{table_name}] Already up to date (last: {last_stored.date()})")
                return
            print(
                f"\n[{table_name}] Resuming from {pull_start} "
                f"(last committed: {last_stored.date()})"
            )

        current = pull_start
        batches_committed = 0

        while current <= config_end:
            chunk_end = min(current + timedelta(days=batch_days - 1), config_end)

            try:
                history = rd.get_history(
                    universe=[ric],
                    fields=fields,
                    start=current,
                    end=chunk_end,
                    interval="daily",
                )
                df = history.reset_index().replace("", pd.NA)

                # Refinitiv ignores the start parameter for constituent-list
                # RICs (e.g. 0#.STOXX) and returns data from each stock's
                # earliest available date. Enforce the window client-side.
                df["Date"] = pd.to_datetime(df["Date"])
                before = len(df)
                df = df[
                    (df["Date"] >= pd.Timestamp(current)) &
                    (df["Date"] <= pd.Timestamp(chunk_end))
                ]
                clipped = before - len(df)
                if clipped > 0:
                    print(f"  [clip] Dropped {clipped} out-of-window rows "                          f"(Refinitiv returned dates outside {current}–{chunk_end})")

                # Coerce Total Return column to Float64 explicitly —
                # Refinitiv sometimes returns it as object dtype which causes
                # DuckDB type coercion warnings on upsert.
                for col in df.columns:
                    if "Return" in str(col):
                        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Float64")

                if df.empty:
                    print(f"  No data for {current} to {chunk_end} — skipping")
                    current = chunk_end + timedelta(days=1)
                    continue

                db.upsertTable(table_name, df, on="Date")
                batches_committed += 1
                print(f"  Fetched {current} to {chunk_end} ({len(df)} rows)")

            except KeyboardInterrupt:
                raise

            except Exception as e:
                print(
                    f"\n  [{table_name}] Batch {current}–{chunk_end} failed: {e}"
                    f"\n  Stopping. Rerun to resume from {current}."
                )
                return

            current = chunk_end + timedelta(days=1)

        print(f"  [{table_name}] {batches_committed} batches committed.")

    def pullGICSSnapshot(self, db: Database):
        """
        Pull a one-time snapshot of GICS sector codes for all STOXX constituents.
        Stored in a simple wide table (Date × RIC columns) in DuckDB under
        TABLE_GICS. Called once — sector codes are effectively static.
        Skipped if the table already exists.
        """
        if TABLE_GICS in db.getTableNames():
            print(f"  [{TABLE_GICS}] Already exists — skipping GICS snapshot")
            return

        print(f"  [{TABLE_GICS}] Pulling GICS sector snapshot...")
        try:
            history = rd.get_data(
                universe=[STOXX_INDEX_RIC],
                fields=STATIC_FIELDS
            )
            db.writeTable(TABLE_GICS, history)
            print(f"  [{TABLE_GICS}] Stored GICS codes for {len(history.columns) - 1} RICs")
        except Exception as e:
            print(f"  [{TABLE_GICS}] GICS snapshot failed: {e} — financials will not be excluded")

    @staticmethod
    def _ricToTableName(ric: str) -> str:
        """Map RIC to table name"""
        if ric == STOXX_INDEX_RIC:
            return TABLE_STOXX
        if ric == STOXX50_INDEX_RIC:
            return TABLE_BENCHMARK
        return ric.split(".")[-1] if "." in ric else ric

    @staticmethod
    def _validateConfig(config: dict):
        required = {"indices", "start", "end"}
        missing = required - set(config.keys())
        if missing:
            raise ValueError(f"Config missing required keys: {missing}")
        if config["start"] > config["end"]:
            raise ValueError("config['start'] must be <= config['end']")