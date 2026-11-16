"""Tests for writing a frame of certificates to one output file."""

import contextlib
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from py_epc._output import SQLITE_TABLE, OutputType, write_output

# Certificates of several types, as a dataset holds them.
_FRAME = pl.LazyFrame(
    {
        "certificate_number": ["a", "b"],
        "lodgement_date": [date(2024, 1, 2), date(2025, 3, 4)],
        "tenure": pl.Series(
            ["owner_occupied", None], dtype=pl.Enum(["owner_occupied"])
        ),
        "lzc_energy_sources": [["solar_pv"], []],
    }
)


class TestOutputType:
    """`OutputType` names each type of output and its file suffix."""

    def test_parquet_has_parquet_suffix(self) -> None:
        assert OutputType.PARQUET.suffix == ".parquet"

    def test_sqlite_has_sqlite_suffix(self) -> None:
        assert OutputType.SQLITE.suffix == ".sqlite"


class TestWriteOutput:
    """`write_output` writes a frame to one file."""

    def test_parquet_keeps_every_value_and_type(self, tmp_path: Path) -> None:
        path = write_output(_FRAME, tmp_path / "out.parquet", OutputType.PARQUET)
        assert pl.read_parquet(path).equals(_FRAME.collect())

    def test_missing_folder_is_created(self, tmp_path: Path) -> None:
        path = tmp_path / "outputs" / "gb" / "out.parquet"
        assert write_output(_FRAME, path, OutputType.PARQUET).is_file()

    def test_existing_file_raises_file_exists_error(self, tmp_path: Path) -> None:
        path = write_output(_FRAME, tmp_path / "out.parquet", OutputType.PARQUET)
        with pytest.raises(FileExistsError, match="overwrite=True"):
            write_output(_FRAME, path, OutputType.PARQUET)

    def test_existing_file_with_overwrite_is_replaced(self, tmp_path: Path) -> None:
        path = write_output(_FRAME, tmp_path / "out.parquet", OutputType.PARQUET)
        write_output(_FRAME.head(1), path, OutputType.PARQUET, overwrite=True)
        assert pl.read_parquet(path).height == 1

    def test_failed_write_leaves_no_file(self, tmp_path: Path) -> None:
        failing = _FRAME.select(pl.col("certificate_number").str.to_integer())
        with pytest.raises(pl.exceptions.ComputeError):
            write_output(failing, tmp_path / "out.parquet", OutputType.PARQUET)
        assert list(tmp_path.iterdir()) == []


def read_sqlite(path: Path, query: str) -> list[tuple[Any, ...]]:
    """Return the rows a query of an SQLite output gives."""
    with contextlib.closing(sqlite3.connect(path)) as connection:
        return connection.execute(query).fetchall()


class TestWriteOutputSqlite:
    """`write_output` writes a frame to a table of an SQLite database."""

    def test_columns_are_declared_by_type(self, tmp_path: Path) -> None:
        frame = _FRAME.with_columns(
            is_latest=pl.lit(True), total_floor_area=pl.lit(52.5), rooms=pl.lit(3)
        )
        path = write_output(frame, tmp_path / "out.sqlite", OutputType.SQLITE)
        columns = read_sqlite(path, f"PRAGMA table_info({SQLITE_TABLE})")
        assert [(column[1], column[2]) for column in columns] == [
            ("certificate_number", "TEXT"),
            ("lodgement_date", "TEXT"),
            ("tenure", "TEXT"),
            ("lzc_energy_sources", "TEXT"),
            ("is_latest", "INTEGER"),
            ("total_floor_area", "REAL"),
            ("rooms", "INTEGER"),
        ]

    def test_values_are_stored_as_numbers_and_text(self, tmp_path: Path) -> None:
        frame = _FRAME.with_columns(
            is_latest=pl.Series([True, False]),
            lodgement_datetime=pl.Series(
                [datetime(2024, 1, 2, 9, 30), None], dtype=pl.Datetime
            ),
        )
        path = write_output(frame, tmp_path / "out.sqlite", OutputType.SQLITE)
        assert read_sqlite(path, f"SELECT * FROM {SQLITE_TABLE}") == [
            ("a", "2024-01-02", "owner_occupied", "solar_pv", 1, "2024-01-02 09:30:00"),
            ("b", "2025-03-04", None, "", 0, None),
        ]

    def test_rows_in_several_batches_are_all_written(self, tmp_path: Path) -> None:
        frame = pl.LazyFrame({"row": range(120_000)})
        path = write_output(frame, tmp_path / "out.sqlite", OutputType.SQLITE)
        assert read_sqlite(path, f"SELECT COUNT(*), SUM(row) FROM {SQLITE_TABLE}") == [
            (120_000, sum(range(120_000)))
        ]
