from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


class SqlRunner:
    """Default SQL runner using an in-memory SQLite database.

    Tables are populated from the provided DataFrames, keyed by dataset name.
    """

    def run(self, sql: str, tables: dict[str, pd.DataFrame]) -> int:
        """Execute *sql* against the provided tables and return the row count.

        Returns:
            int: Number of rows returned by the query.
                 0  => pass (no violations found)
                 ≥1 => fail (violations found)
        """
        rows_df = self.run_with_rows(sql, tables)
        return len(rows_df)

    def run_with_rows(self, sql: str, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Execute *sql* and return query result rows as a DataFrame."""
        conn = sqlite3.connect(":memory:")
        try:
            for name, df in tables.items():
                df.to_sql(name, conn, if_exists="replace", index=False)
            return pd.read_sql_query(sql, conn)
        finally:
            conn.close()


def load_sql_file(path: str | Path) -> str:
    """Read and return the contents of a SQL file."""
    return Path(path).read_text(encoding="utf-8")


def resolve_sql_ref(ref: dict[str, Any], base_path: Path | None = None) -> tuple[str, str]:
    """Resolve a single ``sql_ref`` item to ``(sql_text, label)``.

    Args:
        ref: One of ``{"file": "<path>"}`` or ``{"inline_sql": "<sql>"}``.
        base_path: Optional base directory used to resolve relative file paths.

    Returns:
        Tuple of *(sql_text, label)* where *label* is ``"file:<path>"`` for
        file-based refs or ``"inline_sql"`` for inline refs.

    Raises:
        ValueError: If *ref* does not contain a recognised key.
    """
    if "file" in ref:
        file_path = ref["file"]
        full_path = (base_path / file_path) if base_path is not None else Path(file_path)
        return load_sql_file(full_path), f"file:{file_path}"
    if "inline_sql" in ref:
        return ref["inline_sql"], "inline_sql"
    raise ValueError(f"sql_ref item must contain 'file' or 'inline_sql', got: {list(ref.keys())}")
