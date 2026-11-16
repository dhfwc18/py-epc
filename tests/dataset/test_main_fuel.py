"""Tests for parsing the main fuel and its community flag."""

import json
from typing import Any

import polars as pl
import pytest

from py_epc._dataset._categories import PhraseVocabulary
from py_epc._dataset._main_fuel import (
    ParsedMainFuel,
    main_fuel_parser,
    parse_main_fuel,
)
from py_epc._dataset._phrases import parse_column

# A small main fuel vocabulary, laid out as the shipped one is.
_MAIN_FUEL_VOCABULARY: dict[str, Any] = {
    "version": 1,
    "format": "phrase",
    "column": "main_fuel",
    "output_column": "main_fuel",
    "not_a_description": [
        "To be used only when there is no heating/hot-water system or data is"
        " from a community network"
    ],
    "categories": {
        "mains_gas": {
            "label": "Mains gas",
            "phrases": [
                "mains gas (community)",
                "mains gas (not community)",
                "Gas: mains gas",
            ],
        },
        "lpg": {"label": "LPG", "phrases": ["LPG (not community)"]},
        "biomass": {
            "label": "Biomass",
            "phrases": ["Community heating schemes: heat from boilers - biomass"],
        },
    },
}


def main_fuel_vocabulary() -> PhraseVocabulary:
    """Return the small main fuel vocabulary."""
    return PhraseVocabulary.model_validate_json(json.dumps(_MAIN_FUEL_VOCABULARY))


class TestParseMainFuel:
    """`parse_main_fuel` gives the fuel and whether it feeds a community."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            pytest.param(
                "mains gas (community)",
                ParsedMainFuel("described", "mains_gas", community=True),
                id="community",
            ),
            pytest.param(
                "mains gas (not community)",
                ParsedMainFuel("described", "mains_gas", community=False),
                id="not-community",
            ),
            pytest.param(
                "LPG (Not Community) ",
                ParsedMainFuel("described", "lpg", community=False),
                id="any-case",
            ),
            pytest.param(
                "Community heating schemes: heat from boilers - biomass",
                ParsedMainFuel("described", "biomass", community=True),
                id="community-scheme-prefix",
            ),
            pytest.param(
                "Gas: mains gas",
                ParsedMainFuel("described", "mains_gas"),
                id="not-stated",
            ),
            pytest.param(
                "To be used only when there is no heating/hot-water system or data"
                " is from a community network",
                ParsedMainFuel("not_a_description"),
                id="community-in-placeholder",
            ),
            pytest.param(None, ParsedMainFuel("not_a_description"), id="none"),
            pytest.param("peat", ParsedMainFuel("unknown"), id="unknown"),
        ],
    )
    def test_phrase_returns_parsed_main_fuel(
        self, text: str | None, expected: ParsedMainFuel
    ) -> None:
        assert parse_main_fuel(text, main_fuel_vocabulary()) == expected


class TestMainFuelParser:
    """`main_fuel_parser` replaces the fuel and adds its community flag."""

    def test_fuel_is_replaced_in_place_and_flag_added(self) -> None:
        frame = pl.LazyFrame(
            {
                "main_fuel": [
                    "mains gas (community)",
                    "Gas: mains gas",
                    "To be used only when there is no heating/hot-water system or"
                    " data is from a community network",
                    None,
                ],
                "uprn": [1, 2, 3, 4],
            }
        )
        result = parse_column(frame, main_fuel_parser(main_fuel_vocabulary()))
        assert result.collect_schema() == pl.Schema(
            {
                "main_fuel": pl.Enum(["mains_gas", "lpg", "biomass"]),
                "uprn": pl.Int64(),
                "main_fuel_community": pl.Boolean(),
            }
        )
        assert result.collect().rows() == [
            ("mains_gas", 1, True),
            ("mains_gas", 2, None),
            (None, 3, None),
            (None, 4, None),
        ]

    def test_unknown_phrase_raises_value_error_listing_it(self) -> None:
        frame = pl.LazyFrame({"main_fuel": ["peat"]})
        with pytest.raises(ValueError, match=r"1 unknown phrases.*'peat'"):
            parse_column(frame, main_fuel_parser(main_fuel_vocabulary()))
