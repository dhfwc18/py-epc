"""Tests for parsing the phrases of a dwelling's air tightness."""

import polars as pl
import pytest

from py_epc._dataset._air_tightness import (
    ParsedAirTightness,
    air_tightness_parser,
    parse_air_tightness,
)
from py_epc._dataset._phrases import parse_column


class TestParseAirTightness:
    """`parse_air_tightness` reads an air permeability and its basis."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            pytest.param(
                "Air permeability 5.4 m3/h.m2 (as tested)",
                ParsedAirTightness("described", 5.4, "tested"),
                id="tested",
            ),
            pytest.param(
                "Air permeability 15.0 m3/h.m2 (assumed)",
                ParsedAirTightness("described", 15.0, "assumed"),
                id="assumed",
            ),
            pytest.param(
                "Air permeability 10 m3/h.m2 (assessed average)",
                ParsedAirTightness("described", 10.0, "assessed_average"),
                id="assessed-average",
            ),
            pytest.param(
                "Air permeability [AP50] = 4.9 m3/h.m2 (as tested)",
                ParsedAirTightness("described", 4.9, "tested", "ap50"),
                id="with-test",
            ),
            pytest.param(
                "Air permeability [AP4] = 2.1 m3/h.m2 ()",
                ParsedAirTightness("described", 2.1, None, "ap4"),
                id="empty-basis",
            ),
            pytest.param(
                "Air permeability 7.0 m?/h.m? (assumed)",
                ParsedAirTightness("described", 7.0, "assumed"),
                id="lost-superscripts",
            ),
            pytest.param(
                "Athreiddedd aer 6.0 m3/h.m2 (rhagdybiaeth)",
                ParsedAirTightness("described", 6.0, "assumed"),
                id="welsh",
            ),
            pytest.param(
                "AIR PERMEABILITY 5.4 M3/H.M2 (AS TESTED)",
                ParsedAirTightness("described", 5.4, "tested"),
                id="any-case",
            ),
            pytest.param(
                "(not tested)",
                ParsedAirTightness("described", basis="not_tested"),
                id="not-tested",
            ),
            pytest.param(
                "(heb ei brofi)",
                ParsedAirTightness("described", basis="not_tested"),
                id="not-tested-welsh",
            ),
            pytest.param(
                "Air permeability -3.0 m3/h.m2 (as tested)",
                ParsedAirTightness("invalid", -3.0, "tested"),
                id="negative",
            ),
            pytest.param(
                "Air permeability 0.0 m3/h.m2 (as tested)",
                ParsedAirTightness("invalid", 0.0, "tested"),
                id="zero",
            ),
            pytest.param(None, ParsedAirTightness("not_a_description"), id="none"),
            pytest.param(
                "Air permeability 5 m3/h.m2 (guessed)",
                ParsedAirTightness("unknown"),
                id="unknown-basis",
            ),
            pytest.param(
                "Air permeability [AP10] = 5 m3/h.m2 (as tested)",
                ParsedAirTightness("unknown"),
                id="unknown-test",
            ),
            pytest.param("Draughty", ParsedAirTightness("unknown"), id="unknown"),
        ],
    )
    def test_phrase_returns_parsed_air_tightness(
        self, text: str | None, expected: ParsedAirTightness
    ) -> None:
        assert parse_air_tightness(text) == expected


class TestAirTightnessParser:
    """`air_tightness_parser` gives its columns through `parse_column`."""

    def test_columns_are_added_for_each_row(self) -> None:
        frame = pl.LazyFrame(
            {
                "air_tightness_description": [
                    "Air permeability [AP50] = 4.9 m3/h.m2 (as tested)",
                    "(not tested)",
                    "Air permeability -3.0 m3/h.m2 (as tested)",
                    None,
                ]
            }
        )
        result = parse_column(frame, air_tightness_parser()).collect()
        assert result.schema == pl.Schema(
            {
                "air_tightness_description": pl.String(),
                "air_tightness_m3_per_h_m2": pl.Float64(),
                "air_tightness_basis": pl.Enum(
                    ["assumed", "tested", "assessed_average", "not_tested"]
                ),
                "air_tightness_test": pl.Enum(["ap50", "ap4"]),
            }
        )
        assert result.drop("air_tightness_description").rows() == [
            (4.9, "tested", "ap50"),
            (None, "not_tested", None),
            (None, None, None),
            (None, None, None),
        ]

    def test_unknown_phrase_raises_value_error_listing_it(self) -> None:
        frame = pl.LazyFrame({"air_tightness_description": ["Draughty"]})
        with pytest.raises(ValueError, match=r"1 unknown phrases.*'Draughty'"):
            parse_column(frame, air_tightness_parser())
