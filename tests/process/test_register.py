"""Tests for register definitions."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from py_epc._process._register import Register, load_register

# Header of each shipped register's files, as published.
_HEADERS = Path(__file__).parent / "headers"

# Keys of the sources that have a shipped register definition.
_SHIPPED_KEYS = ["epc.scotland.domestic", "epc.england-and-wales.domestic"]

# Fields of a minimal valid register, with one text and one date column.
_REGISTER: dict[str, Any] = {
    "version": 1,
    "source_key": "epc.scotland.domestic",
    "member_pattern": r"(\d{4})\.csv",
    "missing_values": [""],
    "columns": [
        {"source": "A", "name": "a", "type": "text", "description": "d"},
        {
            "source": "B",
            "name": "b",
            "type": "date",
            "description": "d",
            "formats": ["%Y-%m-%d"],
            "aliases": ["OLD_B"],
        },
    ],
}


def parse(fields: dict[str, Any]) -> Register:
    """Return the register for `fields`, read from JSON as shipped ones are."""
    return Register.model_validate_json(json.dumps(fields))


def register_with(**changes: Any) -> dict[str, Any]:
    """Return the minimal register's fields, with some replaced."""
    return {**_REGISTER, **changes}


def column(**fields: Any) -> dict[str, Any]:
    """Return the fields of a text column, with some replaced."""
    return {"source": "C", "name": "c", "type": "text", "description": "d", **fields}


class TestRegister:
    """`Register` checks a definition when it is made."""

    def test_minimal_definition_is_valid(self) -> None:
        register = parse(_REGISTER)
        assert [column.name for column in register.columns] == ["a", "b"]

    def test_member_pattern_compiles_to_regex(self) -> None:
        register = parse(_REGISTER)
        assert register.member_regex.fullmatch("2025.csv") is not None

    @pytest.mark.parametrize(
        ("changes", "match"),
        [
            pytest.param(
                {"columns": [*_REGISTER["columns"], column(name="a")]},
                "column names must be unique",
                id="repeated-name",
            ),
            pytest.param(
                {"columns": [*_REGISTER["columns"], column(source="A")]},
                "header names must be unique",
                id="repeated-source",
            ),
            pytest.param(
                {"columns": [*_REGISTER["columns"], column(aliases=["OLD_B"])]},
                "header names must be unique",
                id="alias-repeats-another-column",
            ),
            pytest.param(
                {"columns": [column(type="date")]},
                "needs formats",
                id="date-without-formats",
            ),
            pytest.param(
                {"columns": [column(formats=["%Y"])]},
                "takes no formats",
                id="text-with-formats",
            ),
            pytest.param(
                {"columns": [column(type="flag")]},
                "true_values and false_values",
                id="flag-without-values",
            ),
            pytest.param(
                {"columns": [column(type="decimal")]},
                "Input should be",
                id="unknown-type",
            ),
            pytest.param(
                {"columns": [column(name="Not Snake")]},
                "should match pattern",
                id="name-not-snake-case",
            ),
            pytest.param(
                {"member_pattern": r"\d{4}\.csv"},
                "one group",
                id="pattern-without-group",
            ),
            pytest.param(
                {"member_pattern": "("},
                "not a valid pattern",
                id="pattern-not-valid",
            ),
            pytest.param(
                {"year_column": "a"},
                "not a date or datetime column",
                id="year-column-not-a-date",
            ),
            pytest.param(
                {"year_column": "missing"},
                "not a date or datetime column",
                id="year-column-unknown",
            ),
            pytest.param(
                {"label_rows": -1},
                "greater than or equal to 0",
                id="negative-label-rows",
            ),
            pytest.param(
                {"columns": [column(width=3)]},
                "Extra inputs are not permitted",
                id="unknown-field",
            ),
        ],
    )
    def test_invalid_definition_raises_validation_error(
        self, changes: dict[str, Any], match: str
    ) -> None:
        with pytest.raises(ValidationError, match=match):
            parse(register_with(**changes))


class TestRegisterMatchHeader:
    """`Register.match_header` matches a file's header to the columns."""

    def test_header_in_any_order_returns_column_per_name(self) -> None:
        register = parse(_REGISTER)
        matched = register.match_header(["B", "A"])
        assert [(name, column.name) for name, column in matched.items()] == [
            ("B", "b"),
            ("A", "a"),
        ]

    def test_old_name_matches_by_alias(self) -> None:
        register = parse(_REGISTER)
        assert register.match_header(["A", "OLD_B"])["OLD_B"].name == "b"

    @pytest.mark.parametrize(
        ("header", "match"),
        [
            pytest.param(["A", "B", "NEW"], r"unknown columns \['NEW'\]", id="unknown"),
            pytest.param(["A"], r"missing columns \['B'\]", id="missing"),
            pytest.param(
                ["A", "B", "OLD_B"],
                r"columns given more than once \['B'\]",
                id="name-and-alias",
            ),
        ],
    )
    def test_mismatched_header_raises_value_error(
        self, header: list[str], match: str
    ) -> None:
        register = parse(_REGISTER)
        with pytest.raises(ValueError, match=match):
            register.match_header(header)

    def test_every_problem_is_named_in_one_error(self) -> None:
        register = parse(_REGISTER)
        with pytest.raises(ValueError, match=r"unknown columns.*; missing columns"):
            register.match_header(["A", "NEW"])


class TestLoadRegister:
    """`load_register` returns a shipped register definition."""

    @pytest.mark.parametrize("key", _SHIPPED_KEYS)
    def test_shipped_definition_matches_published_header(self, key: str) -> None:
        header = (
            (_HEADERS / f"{key.replace('.', '-')}.csv")
            .read_text(encoding="utf-8")
            .strip()
            .split(",")
        )
        assert list(load_register(key).match_header(header)) == header

    @pytest.mark.parametrize("key", _SHIPPED_KEYS)
    def test_shipped_definition_names_its_source(self, key: str) -> None:
        assert load_register(key).source_key == key

    def test_unknown_key_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="No register definition"):
            load_register("epc.scotland.display")
