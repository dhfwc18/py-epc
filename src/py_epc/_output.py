"""Write a frame of certificates to one output file."""

import contextlib
import os
import sqlite3
from enum import StrEnum
from pathlib import Path
from typing import Final

import polars as pl

# Table of an SQLite output holding the certificates.
SQLITE_TABLE: Final = "certificates"

# Rows inserted into an SQLite output at a time.
_SQLITE_BATCH_ROWS: Final = 50_000

# Separator between the items of a list, stored as text in SQLite.
_SQLITE_LIST_SEPARATOR: Final = "|"


class OutputType(StrEnum):
    """Type of file a frame of certificates is written to."""

    PARQUET = "parquet"
    """One Parquet file, keeping every column's type."""

    SQLITE = "sqlite"
    """
    One SQLite database with a table of certificates. Categories, dates
    and times are stored as text, and lists as text with their items
    separated by ``"|"``.
    """

    @property
    def suffix(self) -> str:
        """Return the file suffix, such as ``".parquet"``."""
        return f".{self.value}"


def write_output(
    frame: pl.LazyFrame,
    path: str | os.PathLike[str],
    output_type: OutputType,
    *,
    overwrite: bool = False,
) -> Path:
    """
    Write a frame of certificates to one file of `output_type`.

    The frame is streamed, so it need not fit in memory. The file is written
    beside `path` as ``.<name>.part`` and renamed to `path` only once it is
    complete, so a failed write never leaves part of a file. An SQLite output
    holds one table, ``certificates``, with a column of type ``INTEGER``,
    ``REAL`` or ``TEXT`` for each column of the frame.

    Parameters
    ----------
    frame : polars.LazyFrame
        Certificates, such as `scan` returns.
    path : str | os.PathLike[str]
        File to write. Its folder is created if missing.
    output_type : OutputType
        Type of file to write.
    overwrite : bool, default False
        Whether to replace an existing file.

    Returns
    -------
    Path
        The file written.

    Raises
    ------
    FileExistsError
        If the file exists and `overwrite` is false.
    """
    target = Path(path).expanduser()
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists; pass overwrite=True to replace it")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.part")
    try:
        match output_type:
            case OutputType.PARQUET:
                frame.sink_parquet(partial, compression="zstd")
            case OutputType.SQLITE:
                _write_sqlite(frame, partial)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def _write_sqlite(frame: pl.LazyFrame, path: Path) -> None:
    """Write `frame` to a new SQLite database at `path`, in batches."""
    schema = frame.collect_schema()
    names = ", ".join(_quote(name) for name in schema)
    declared = ", ".join(
        f"{_quote(name)} {_sqlite_type(dtype)}" for name, dtype in schema.items()
    )
    placeholders = ", ".join("?" * len(schema))
    stored = frame.select(_as_sqlite(name, dtype) for name, dtype in schema.items())
    with contextlib.closing(sqlite3.connect(path)) as connection:
        # The file is a partial one, deleted on failure, so no journal is kept.
        connection.execute("PRAGMA journal_mode = OFF")
        connection.execute("PRAGMA synchronous = OFF")
        connection.execute(f"CREATE TABLE {_quote(SQLITE_TABLE)} ({declared})")
        insert = f"INSERT INTO {_quote(SQLITE_TABLE)} ({names}) VALUES ({placeholders})"
        for batch in stored.collect_batches(chunk_size=_SQLITE_BATCH_ROWS):
            connection.executemany(insert, batch.iter_rows())
        connection.commit()


def _sqlite_type(dtype: pl.DataType) -> str:
    """Return the SQLite type a column of `dtype` is stored as."""
    if dtype == pl.Boolean or dtype.is_integer():
        return "INTEGER"
    if dtype.is_float():
        return "REAL"
    return "TEXT"


def _as_sqlite(name: str, dtype: pl.DataType) -> pl.Expr:
    """Return a column as SQLite can store it: a number or text."""
    column = pl.col(name)
    if isinstance(dtype, pl.List):
        return column.cast(pl.List(pl.String)).list.join(_SQLITE_LIST_SEPARATOR)
    if dtype == pl.Date:
        return column.dt.to_string("%Y-%m-%d")
    if isinstance(dtype, pl.Datetime):
        return column.dt.to_string("%Y-%m-%d %H:%M:%S")
    if dtype == pl.Boolean or dtype.is_numeric():
        return column
    return column.cast(pl.String)


def _quote(name: str) -> str:
    """Return a table or column name quoted for SQLite."""
    return '"' + name.replace('"', '""') + '"'
