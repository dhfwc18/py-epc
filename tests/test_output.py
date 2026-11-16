"""Tests for writing a frame of certificates to one output file."""

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from py_epc._output import OutputType, write_output

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
