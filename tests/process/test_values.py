"""Tests for parsing raw values into each column's type."""

import json
from datetime import date, datetime
from typing import Any

import polars as pl
import pytest

from py_epc._process._register import Column, Register
from py_epc._process._values import clean, is_invalid, parse

# A register with Scotland's missing, true and false values.
_REGISTER = Register.model_validate_json(
    json.dumps(
        {
            "version": 1,
            "source_key": "epc.scotland.domestic",
            "member_pattern": r"(\d{4})\.csv",
            "missing_values": ["", "N/A", "Data removed"],
            "true_values": ["Y", "True", "true"],
            "false_values": ["N", "False", "false"],
            "columns": [
                {"source": "A", "name": "a", "type": "flag", "description": "d"}
            ],
        }
    )
)


def column(type: str, **fields: Any) -> Column:
    """Return a column of a type, read from JSON as shipped ones are."""
    spec = {"source": "A", "name": "a", "type": type, "description": "d"}
    return Column.model_validate_json(json.dumps({**spec, **fields}))


def parsed(values: list[str | None], column: Column) -> list[Any]:
    """Return `values` parsed as the column."""
    frame = pl.DataFrame({"raw": values}, schema={"raw": pl.String})
    return frame.select(parse(pl.col("raw"), column, _REGISTER)).to_series().to_list()


def invalid(values: list[str | None], column: Column) -> list[bool]:
    """Return which of `values` are invalid for the column."""
    frame = pl.DataFrame({"raw": values}, schema={"raw": pl.String})
    return (
        frame.select(is_invalid(pl.col("raw"), column, _REGISTER)).to_series().to_list()
    )


class TestClean:
    """`clean` trims values and turns missing values into null."""

    def test_surrounding_spaces_are_removed(self) -> None:
        frame = pl.DataFrame({"raw": ["owner-occupied ", " a b "]})
        result = frame.select(clean(pl.col("raw"), _REGISTER)).to_series()
        assert result.to_list() == ["owner-occupied", "a b"]

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param("", id="empty"),
            pytest.param("   ", id="spaces"),
            pytest.param("N/A", id="n-a"),
            pytest.param("N/A ", id="n-a-with-space"),
            pytest.param("Data removed", id="data-removed"),
            pytest.param(None, id="null"),
        ],
    )
    def test_missing_value_becomes_null(self, value: str | None) -> None:
        frame = pl.DataFrame({"raw": [value]}, schema={"raw": pl.String})
        result = frame.select(clean(pl.col("raw"), _REGISTER)).to_series()
        assert result.to_list() == [None]


class TestParse:
    """`parse` turns text values into the column's type."""

    def test_text_value_is_cleaned_only(self) -> None:
        assert parsed(["Detached ", "N/A"], column("text")) == ["Detached", None]

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param("345.00", 345.0, id="decimal"),
            pytest.param(".30", 0.3, id="no-leading-digit"),
            pytest.param("11.", 11.0, id="no-trailing-digit"),
            pytest.param("-0.5", -0.5, id="negative"),
            pytest.param("1.5e3", 1500.0, id="exponent"),
            pytest.param(" 7 ", 7.0, id="spaces"),
        ],
    )
    def test_number_value_returns_float(self, value: str, expected: float) -> None:
        assert parsed([value], column("number")) == [expected]

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param("12", 12, id="plain"),
            pytest.param("100.0", 100, id="zero-fraction"),
            pytest.param("11.", 11, id="no-trailing-digit"),
            pytest.param("-1", -1, id="negative"),
            pytest.param("00", 0, id="zero-padded"),
            pytest.param("906700000001", 906700000001, id="uprn"),
        ],
    )
    def test_whole_number_returns_integer(self, value: str, expected: int) -> None:
        assert parsed([value], column("integer")) == [expected]

    def test_date_value_tries_each_format_in_order(self) -> None:
        dates = column("date", formats=["%Y-%m-%d", "%d/%m/%Y"])
        assert parsed(["2025-03-26", "27/03/2025"], dates) == [
            date(2025, 3, 26),
            date(2025, 3, 27),
        ]

    def test_datetime_value_returns_datetime(self) -> None:
        times = column("datetime", formats=["%d/%m/%Y %H:%M:%S"])
        assert parsed(["26/03/2025 14:37:12"], times) == [
            datetime(2025, 3, 26, 14, 37, 12)
        ]

    def test_flag_value_returns_boolean(self) -> None:
        assert parsed(["Y", "true", "False", "N", ""], column("flag")) == [
            True,
            True,
            False,
            False,
            None,
        ]


class TestIsInvalid:
    """`is_invalid` marks values that are present but do not parse."""

    @pytest.mark.parametrize(
        ("type", "fields", "value"),
        [
            pytest.param("number", {}, "abc", id="number-word"),
            pytest.param("number", {}, "inf", id="number-inf"),
            pytest.param("number", {}, "nan", id="number-nan"),
            pytest.param("number", {}, "1,000", id="number-comma"),
            pytest.param("integer", {}, "11.5", id="integer-fraction"),
            pytest.param("date", {"formats": ["%Y-%m-%d"]}, "26/03/2025", id="date"),
            pytest.param(
                "datetime",
                {"formats": ["%d/%m/%Y %H:%M:%S"]},
                "2025-03-26",
                id="datetime",
            ),
            pytest.param("flag", {}, "maybe", id="flag"),
        ],
    )
    def test_unparsable_value_is_invalid_and_becomes_null(
        self, type: str, fields: dict[str, Any], value: str
    ) -> None:
        spec = column(type, **fields)
        assert invalid([value], spec) == [True]
        assert parsed([value], spec) == [None]

    @pytest.mark.parametrize("value", ["", "N/A", None])
    def test_missing_value_is_not_invalid(self, value: str | None) -> None:
        assert invalid([value], column("number")) == [False]

    def test_text_value_is_never_invalid(self) -> None:
        assert invalid(["anything"], column("text")) == [False]
