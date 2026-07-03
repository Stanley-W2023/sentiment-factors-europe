"""
db.py — DuckDB-backed storage for the RSF research pipeline
  - MultiIndex columns are flattened to "RIC||Field" on write, restored on read.
  - Integer columns are upcast to INT64 before writing to prevent overflow
    on large Volume values from Refinitiv.
"""

import os
from typing import Optional, Iterable, List

import pandas as pd
import duckdb

_MI_SEP = "||"


class Database:
    """DuckDB-backed storage for all RSF pipeline data."""

    def __init__(self, db_path: str = "data/rsf.duckdb"):
        self.db_path = db_path
        self.con: Optional[duckdb.DuckDBPyConnection] = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def open(self):
        os.makedirs(
            os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".",
            exist_ok=True,
        )
        self.con = duckdb.connect(self.db_path)

    def close(self):
        if self.con:
            self.con.close()
            self.con = None

    def writeTable(self, table_name: str, df: pd.DataFrame):
        """Full overwrite — use for first run or fully-recomputed tables."""
        self._checkOpen()
        df = self._prepare(df)
        self.con.execute(f"DROP TABLE IF EXISTS {self._q(table_name)}")
        self.con.execute(
            f"CREATE TABLE {self._q(table_name)} AS SELECT * FROM df"
        )

    def upsertTable(self, table_name: str, df: pd.DataFrame, on: str = "Date"):
        """
        Insert only rows whose `on` key does not already exist in the table.
        Existing rows are never overwritten. Safe to call repeatedly.
        Widens any INT32 columns to BIGINT before inserting to prevent overflow.
        """
        self._checkOpen()
        df = self._prepare(df)

        if table_name not in self.getTableNames():
            self.writeTable(table_name, df)
            print(f"  [{table_name}] Created new table ({len(df)} rows)")
            return

        existing = self.con.execute(
            f"SELECT DISTINCT {self._q(on)} FROM {self._q(table_name)}"
        ).df()[on]
        existing_set = set(pd.to_datetime(existing))

        df[on] = pd.to_datetime(df[on])
        new_rows = df[~df[on].isin(existing_set)]

        if new_rows.empty:
            print(f"  [{table_name}] Already up to date")
            return

        self._widenIntColumns(table_name)

        self.con.execute(
            f"INSERT INTO {self._q(table_name)} SELECT * FROM new_rows"
        )
        print(f"  [{table_name}] Appended {len(new_rows)} new rows")

    def readTable(self, table_name: str) -> pd.DataFrame:
        """Read full table sorted by Date. Restores MultiIndex columns if present."""
        self._checkOpen()
        if table_name not in self.getTableNames():
            raise FileNotFoundError(
                f"No table '{table_name}' in {self.db_path}"
            )
        # Not all tables have a Date column (e.g. monthly_panel uses year_month).
        # Fall back to unordered read if Date column is absent.
        cols = [r[0] for r in self.con.execute(
            f"SELECT column_name FROM information_schema.columns "
            f"WHERE table_name = '{table_name}'"
        ).fetchall()]
        if "Date" in cols:
            df = self.con.execute(
                f'SELECT * FROM {self._q(table_name)} ORDER BY "Date"'
            ).df()
        else:
            df = self.con.execute(
                f'SELECT * FROM {self._q(table_name)}'
            ).df()
        return self._restoreMultiindex(df)

    def readTableSince(
        self, table_name: str, since: pd.Timestamp
    ) -> pd.DataFrame:
        """Read only rows where Date >= since."""
        self._checkOpen()
        df = self.con.execute(
            f'SELECT * FROM {self._q(table_name)} WHERE "Date" >= ? ORDER BY "Date"',
            [since],
        ).df()
        return self._restoreMultiindex(df)

    def query(self, sql: str) -> pd.DataFrame:
        """Execute arbitrary SQL and return result as DataFrame."""
        self._checkOpen()
        return self.con.execute(sql).df()

    def lastDate(self, table_name: str) -> Optional[pd.Timestamp]:
        """
        Return the most recent Date in a table, or None if empty/missing.
        Used by RefinitivClient to determine the incremental pull start date.
        """
        self._checkOpen()
        if table_name not in self.getTableNames():
            return None
        result = self.con.execute(
            f'SELECT MAX("Date") AS last FROM {self._q(table_name)}'
        ).df()
        val = result["last"].iloc[0]
        return pd.Timestamp(val) if val is not None and not pd.isna(val) else None

    def getTableNames(self) -> List[str]:
        self._checkOpen()
        result = self.con.execute("SHOW TABLES").df()
        return result["name"].tolist() if not result.empty else []

    def hasTable(self, table_name: str) -> bool:
        """Case-insensitive existence check — DuckDB lowercases all table names."""
        return table_name.lower() in [t.lower() for t in self.getTableNames()]

    def resolveTableName(self, table_name: str) -> str:
        """Return the actual stored name matching the given name (case-insensitive)."""
        lookup = {t.lower(): t for t in self.getTableNames()}
        return lookup.get(table_name.lower(), table_name)

    def dropAllTables(self):
        """Drop every table in the database. Useful for full reset."""
        for name in self.getTableNames():
            self.con.execute(f"DROP TABLE IF EXISTS {self._q(name)}")

    def deleteRowsBefore(self, table_name: str, cutoff_date: str):
        """
        Delete all rows from table_name where Date < cutoff_date.
        Used to purge pre-sample rows that Refinitiv returns despite the
        start parameter being set (known bug with 0#.STOXX constituent expansion).
        Silently skips if table does not exist.
        """
        self._checkOpen()
        if table_name not in self.getTableNames():
            return
        result = self.con.execute(
            f'DELETE FROM {self._q(table_name)} WHERE "Date" < ?',
            [pd.Timestamp(cutoff_date)],
        )
        n = result.rowcount if hasattr(result, "rowcount") else "?"
        if isinstance(n, int) and n > 0:
            print(f"  [db] Deleted {n:,} pre-sample rows from {table_name} (before {cutoff_date})")

    def dropTables(self, table_names: List[str]):
        """Drop specific tables by name. Silently skips missing tables."""
        for name in table_names:
            if name in self.getTableNames():
                self.con.execute(f"DROP TABLE IF EXISTS {self._q(name)}")
                print(f"Dropped {name}")
            else:
                print(f"{name} not found — nothing to drop")

    def exportTables(
        self,
        out_dir: str = "data/exports",
        tables: Optional[Iterable[str]] = None,
    ):
        """Export tables to CSV for inspection or backup."""
        os.makedirs(out_dir, exist_ok=True)
        for name in tables or self.getTableNames():
            try:
                df = self.readTable(name)
                df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
                print(f"Exported {name} ({len(df)} rows)")
            except Exception as e:
                print(f"Failed to export {name}: {e}")

    # ── internals ─────────────────────────────────────────────────────────────

    def _checkOpen(self):
        if self.con is None:
            raise RuntimeError(
                "Database is not open. Use as a context manager: "
                "`with Database(...) as db:`"
            )

    def _widenIntColumns(self, table_name: str):
        """
        Widen any INT32 columns in an existing table to BIGINT.
        Called before INSERT to prevent overflow on large Volume values.
        """
        schema = self.con.execute(
            f"SELECT column_name, data_type FROM information_schema.columns "
            f"WHERE table_name = '{table_name}'"
        ).df()
        for _, row in schema.iterrows():
            if row["data_type"].upper() in ("INTEGER", "INT", "INT32", "INT4"):
                self.con.execute(
                    f"ALTER TABLE {self._q(table_name)} "
                    f"ALTER COLUMN {self._q(row['column_name'])} TYPE BIGINT"
                )

    @staticmethod
    def _q(name: str) -> str:
        """Double-quote an identifier for safe use in SQL."""
        return '"' + str(name).replace('"', "").replace(";", "") + '"'

    @staticmethod
    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise a DataFrame before writing to DuckDB:
          1. Flatten MultiIndex columns to "RIC||Field" strings.
          2. Upcast integer columns to INT64.
          3. Normalise Date column to midnight UTC.
        """
        df = df.copy()

        if isinstance(df.columns, pd.MultiIndex):
            new_cols = []
            for col in df.columns:
                parts = [str(c).strip() for c in col if str(c).strip()]
                new_cols.append(
                    _MI_SEP.join(parts) if len(parts) > 1
                    else (parts[0] if parts else "unnamed")
                )
            df.columns = new_cols

        for col in df.select_dtypes(include="integer").columns:
            df[col] = df[col].astype("Int64")

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

        # DuckDB does not recognise pandas Period dtype — convert to string.
        # Restored to Period on read via readTable callers that need it.
        for col in df.columns:
            if pd.api.types.is_period_dtype(df[col]):
                df[col] = df[col].astype(str)

        return df

    @staticmethod
    def _restoreMultiindex(df: pd.DataFrame) -> pd.DataFrame:
        """Reverse of _prepare's MultiIndex flattening."""
        if not any(_MI_SEP in str(col) for col in df.columns):
            return df

        tuples = []
        for col in df.columns:
            if _MI_SEP in str(col):
                ric, field = str(col).split(_MI_SEP, 1)
                tuples.append((ric, field))
            else:
                tuples.append((str(col), ""))

        df.columns = pd.MultiIndex.from_tuples(tuples)
        return df