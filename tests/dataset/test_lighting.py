"""Tests for parsing the phrases of a dwelling's lighting."""

import polars as pl
import pytest

from py_epc._dataset._lighting import ParsedLighting, lighting_parser, parse_lighting
from py_epc._dataset._phrases import parse_column


class TestParseLighting:
    """`parse_lighting` reads the share or efficiency of the lighting."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            pytest.param(
                "Low energy lighting in 50% of fixed outlets",
                ParsedLighting("described", 50.0),
                id="share",
            ),
            pytest.param(
                "Low energy lighting in 33% fixed outlets",
                ParsedLighting("described", 33.0),
                id="share-without-of",
            ),
            pytest.param(
                "Low energy lighting in 12.5% of fixed outlets",
                ParsedLighting("described", 12.5),
                id="share-with-decimals",
            ),
            pytest.param(
                "Goleuadau ynni-isel mewn 40% o'r mannau gosod",
                ParsedLighting("described", 40.0),
                id="share-welsh",
            ),
            pytest.param(
                "Low energy lighting in all fixed outlets",
                ParsedLighting("described", 100.0),
                id="all",
            ),
            pytest.param(
                "No low energy lighting", ParsedLighting("described", 0.0), id="none"
            ),
            pytest.param(
                "Good lighting efficiency",
                ParsedLighting("described", efficiency="good"),
                id="efficiency",
            ),
            pytest.param(
                "Excelent lighting efficiency",
                ParsedLighting("described", efficiency="excellent"),
                id="efficiency-typo",
            ),
            pytest.param(
                "Description: No low energy lighting",
                ParsedLighting("described", 0.0),
                id="description-prefix",
            ),
            pytest.param(
                "Low energy lighting in 775% of fixed outlets",
                ParsedLighting("invalid", 775.0),
                id="share-above-100",
            ),
            pytest.param(
                "SAP05:Lighting", ParsedLighting("not_a_description"), id="placeholder"
            ),
            pytest.param("???", ParsedLighting("not_a_description"), id="marks-only"),
            pytest.param(None, ParsedLighting("not_a_description"), id="none-given"),
            pytest.param("Candles", ParsedLighting("unknown"), id="unknown"),
        ],
    )
    def test_phrase_returns_parsed_lighting(
        self, text: str | None, expected: ParsedLighting
    ) -> None:
        assert parse_lighting(text) == expected


class TestLightingParser:
    """`lighting_parser` gives the lighting columns through `parse_column`."""

    def test_columns_are_added_for_each_row(self) -> None:
        frame = pl.LazyFrame(
            {
                "lighting_description": [
                    "Low energy lighting in 50% of fixed outlets",
                    "Good lighting efficiency",
                    "Low energy lighting in 775% of fixed outlets",
                    None,
                ]
            }
        )
        result = parse_column(frame, lighting_parser()).collect()
        assert result.schema == pl.Schema(
            {
                "lighting_description": pl.String(),
                "lighting_low_energy_percent": pl.Float64(),
                "lighting_efficiency": pl.Enum(["excellent", "good", "below_average"]),
            }
        )
        assert result.drop("lighting_description").rows() == [
            (50.0, None),
            (None, "good"),
            (None, None),
            (None, None),
        ]

    def test_unknown_phrase_raises_value_error_listing_it(self) -> None:
        frame = pl.LazyFrame({"lighting_description": ["Candles"]})
        with pytest.raises(ValueError, match=r"1 unknown phrases.*'Candles'"):
            parse_column(frame, lighting_parser())
