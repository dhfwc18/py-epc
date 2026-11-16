"""Tests for parsing the floor a dwelling is on."""

import polars as pl
import pytest

from py_epc._dataset._floor_level import (
    ParsedFloorLevel,
    floor_level_parser,
    parse_floor_level,
)
from py_epc._dataset._phrases import parse_column


class TestParseFloorLevel:
    """`parse_floor_level` reads the floor's number and position."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            pytest.param("1st", ParsedFloorLevel("described", 1), id="first"),
            pytest.param("2nd", ParsedFloorLevel("described", 2), id="second"),
            pytest.param("3rd", ParsedFloorLevel("described", 3), id="third"),
            pytest.param("14th", ParsedFloorLevel("described", 14), id="fourteenth"),
            pytest.param(
                "21st or above", ParsedFloorLevel("described", 21), id="or-above"
            ),
            pytest.param(
                "Ground", ParsedFloorLevel("described", 0, "ground"), id="ground"
            ),
            pytest.param(
                "ground floor",
                ParsedFloorLevel("described", 0, "ground"),
                id="ground-floor",
            ),
            pytest.param(
                "Basement",
                ParsedFloorLevel("described", -1, "basement"),
                id="basement",
            ),
            pytest.param(
                "mid floor",
                ParsedFloorLevel("described", None, "mid"),
                id="mid-floor",
            ),
            pytest.param(
                "TOP FLOOR",
                ParsedFloorLevel("described", None, "top"),
                id="any-case",
            ),
            pytest.param(None, ParsedFloorLevel("not_a_description"), id="none"),
            pytest.param("mezzanine", ParsedFloorLevel("unknown"), id="unknown"),
        ],
    )
    def test_phrase_returns_parsed_floor_level(
        self, text: str | None, expected: ParsedFloorLevel
    ) -> None:
        assert parse_floor_level(text) == expected


class TestFloorLevelParser:
    """`floor_level_parser` gives the floor through `parse_column`."""

    def test_floor_is_added_for_each_row(self) -> None:
        frame = pl.LazyFrame({"floor_level": ["2nd", "top floor", "Basement", None]})
        result = parse_column(frame, floor_level_parser()).collect()
        assert result.schema == pl.Schema(
            {
                "floor_level": pl.String(),
                "floor_level_number": pl.Int64(),
                "floor_level_position": pl.Enum(["basement", "ground", "mid", "top"]),
            }
        )
        assert result.drop("floor_level").rows() == [
            (2, None),
            (None, "top"),
            (-1, "basement"),
            (None, None),
        ]

    def test_unknown_floor_raises_value_error_listing_it(self) -> None:
        frame = pl.LazyFrame({"floor_level": ["mezzanine"]})
        with pytest.raises(ValueError, match=r"1 unknown phrases.*'mezzanine'"):
            parse_column(frame, floor_level_parser())
