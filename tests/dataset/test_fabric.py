"""Tests for parsing the phrases of a building's fabric."""

import json
from typing import Any

import polars as pl
import pytest
from pydantic import ValidationError

from py_epc._dataset._fabric import (
    FabricVocabulary,
    ParsedFabric,
    fabric_parser,
    parse_fabric,
)
from py_epc._dataset._phrases import parse_column
from py_epc._dataset._vocabulary import load_vocabulary

# A small walls vocabulary, laid out as the shipped one is.
_WALLS_VOCABULARY: dict[str, Any] = {
    "version": 1,
    "format": "fabric",
    "column": "walls_description",
    "prefix": "walls",
    "misplaced": ["Pitched"],
    "not_a_description": ["SAP05:Walls", "Error"],
    "construction": {
        "cavity": {"label": "Cavity wall", "phrases": ["Cavity wall", "Waliau ceudod"]},
        "solid_brick": {"label": "Solid brick", "phrases": ["Solid brick"]},
    },
    "insulation": {
        "none": {
            "label": "No insulation",
            "phrases": [
                "as built, no insulation",
                "fel y'u hadeiladwyd, dim inswleiddio",
            ],
        },
        "filled_cavity": {"label": "Filled cavity", "phrases": ["filled cavity"]},
    },
}

# A small roof vocabulary, whose insulation may give a thickness.
_ROOF_VOCABULARY: dict[str, Any] = {
    "version": 1,
    "format": "fabric",
    "column": "roof_description",
    "prefix": "roof",
    "insulation_has_thickness": True,
    "construction": {"pitched": {"label": "Pitched", "phrases": ["Pitched"]}},
    "insulation": {
        "loft": {"label": "Loft", "phrases": ["loft insulation"]},
        "none": {"label": "None", "phrases": ["no insulation"]},
    },
}


def walls_vocabulary(**changes: Any) -> FabricVocabulary:
    """Return the small walls vocabulary, with fields changed."""
    return FabricVocabulary.model_validate_json(
        json.dumps({**_WALLS_VOCABULARY, **changes})
    )


def roof_vocabulary() -> FabricVocabulary:
    """Return the small roof vocabulary."""
    return FabricVocabulary.model_validate_json(json.dumps(_ROOF_VOCABULARY))


def described(
    construction: str | None, insulation: str | None, *, assumed: bool = False
) -> ParsedFabric:
    """Return a described phrase of a construction and its insulation."""
    return ParsedFabric(
        "described", construction=construction, insulation=insulation, assumed=assumed
    )


class TestFabricVocabulary:
    """`FabricVocabulary` checks a vocabulary when it is made."""

    @pytest.mark.parametrize("element", ["walls", "roof", "floor"])
    def test_shipped_vocabulary_is_valid(self, element: str) -> None:
        vocabulary = load_vocabulary(element, FabricVocabulary)
        assert (vocabulary.column, vocabulary.prefix) == (
            f"{element}_description",
            element,
        )

    def test_phrase_in_two_categories_raises_validation_error(self) -> None:
        construction = {
            "cavity": {"label": "a", "phrases": ["Cavity wall"]},
            "other": {"label": "b", "phrases": ["cavity WALL"]},
        }
        with pytest.raises(ValidationError, match="is in 'cavity' and 'other'"):
            walls_vocabulary(construction=construction)

    def test_key_not_in_snake_case_raises_validation_error(self) -> None:
        construction = {"Cavity": {"label": "a", "phrases": ["Cavity wall"]}}
        with pytest.raises(ValidationError, match="not snake case"):
            walls_vocabulary(construction=construction)


class TestParseFabric:
    """`parse_fabric` says what a phrase tells of an element."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            pytest.param(
                "Cavity wall, filled cavity",
                described("cavity", "filled_cavity"),
                id="construction-and-insulation",
            ),
            pytest.param(
                "Solid brick, as built, no insulation (assumed)",
                described("solid_brick", "none", assumed=True),
                id="assumed",
            ),
            pytest.param(
                "cavity WALL, Filled Cavity",
                described("cavity", "filled_cavity"),
                id="any-case",
            ),
            pytest.param(
                "Waliau ceudod, fel y'u hadeiladwyd, dim inswleiddio (rhagdybiaeth)",
                described("cavity", "none", assumed=True),
                id="welsh",
            ),
            pytest.param(
                "Cavity wall", described("cavity", None), id="construction-only"
            ),
            pytest.param(
                ", filled cavity",
                described(None, "filled_cavity"),
                id="empty-construction",
            ),
            pytest.param(
                "Cavity wall????????????filled cavity",
                described("cavity", "filled_cavity"),
                id="lost-separator",
            ),
            pytest.param(
                "Description: Cavity wall, filled cavity",
                described("cavity", "filled_cavity"),
                id="description-prefix",
            ),
            pytest.param(
                "Cavity wall, Error",
                described("cavity", None),
                id="insulation-not-a-description",
            ),
            pytest.param(
                "Average thermal transmittance 0.25 W/m2K",
                ParsedFabric("described", u_value=0.25),
                id="u-value",
            ),
            pytest.param(
                "Trawsyriannedd thermol cyfartalog = 0.21 W/m2K",
                ParsedFabric("described", u_value=0.21),
                id="u-value-welsh-equals",
            ),
            pytest.param(
                "Average thermal transmittance -0.25 W/m2K",
                ParsedFabric("invalid", u_value=-0.25),
                id="u-value-negative",
            ),
            pytest.param(
                "Average thermal transmittance 0.00 W/m2K",
                ParsedFabric("invalid", u_value=0.0),
                id="u-value-zero",
            ),
            pytest.param(
                "Average thermal transmittance NaN W/m2K",
                ParsedFabric("not_a_description"),
                id="u-value-nan",
            ),
            pytest.param(
                "Pitched, 100 mm loft insulation",
                ParsedFabric("misplaced"),
                id="misplaced",
            ),
            pytest.param(
                "SAP05:Walls", ParsedFabric("not_a_description"), id="placeholder"
            ),
            pytest.param(
                "SAP05:Walls, filled cavity",
                ParsedFabric("not_a_description"),
                id="placeholder-first-part",
            ),
            pytest.param(
                "(assumed)", ParsedFabric("not_a_description"), id="assumed-only"
            ),
            pytest.param(None, ParsedFabric("not_a_description"), id="none"),
        ],
    )
    def test_phrase_returns_parsed_fabric(
        self, text: str | None, expected: ParsedFabric
    ) -> None:
        assert parse_fabric(text, walls_vocabulary()) == expected

    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("Mystery wall, filled cavity", id="unknown-construction"),
            pytest.param("Cavity wall, mystery insulation", id="unknown-insulation"),
            pytest.param(
                "Cavity wall, 250 mm loft insulation", id="thickness-not-read"
            ),
        ],
    )
    def test_phrase_in_no_list_is_unknown(self, text: str) -> None:
        assert parse_fabric(text, walls_vocabulary()).kind == "unknown"

    @pytest.mark.parametrize(
        ("text", "insulation_mm"),
        [
            pytest.param("Pitched, 250 mm loft insulation", 250, id="plain"),
            pytest.param("Pitched, 300+ mm loft insulation", 300, id="plus"),
            pytest.param("Pitched, >=300 mm loft insulation", 300, id="at-least"),
            pytest.param("Pitched, 100mm loft insulation", 100, id="no-space"),
        ],
    )
    def test_thickness_returns_millimetres_and_insulation(
        self, text: str, insulation_mm: int
    ) -> None:
        result = parse_fabric(text, roof_vocabulary())
        assert (result.construction, result.insulation, result.insulation_mm) == (
            "pitched",
            "loft",
            insulation_mm,
        )

    def test_insulation_alone_returns_no_construction(self) -> None:
        result = parse_fabric("no insulation", roof_vocabulary())
        assert result == described(None, "none")


class TestFabricParser:
    """`fabric_parser` gives an element's columns through `parse_column`."""

    def test_columns_are_added_for_each_row(self) -> None:
        frame = pl.LazyFrame(
            {
                "walls_description": [
                    "Cavity wall, filled cavity",
                    "Average thermal transmittance 0.25 W/m2K",
                    None,
                    "Average thermal transmittance -0.25 W/m2K",
                ]
            }
        )
        result = parse_column(frame, fabric_parser(walls_vocabulary())).collect()
        assert result.columns == [
            "walls_description",
            "walls_construction",
            "walls_insulation",
            "walls_u_value",
            "walls_assumed",
        ]
        assert result.drop("walls_description").rows() == [
            ("cavity", "filled_cavity", None, False),
            (None, None, 0.25, None),
            (None, None, None, None),
            (None, None, None, None),
        ]

    def test_thickness_column_follows_insulation_when_vocabulary_has_it(self) -> None:
        frame = pl.LazyFrame({"roof_description": ["Pitched, 250 mm loft insulation"]})
        result = parse_column(frame, fabric_parser(roof_vocabulary())).collect()
        assert result.schema == pl.Schema(
            {
                "roof_description": pl.String(),
                "roof_construction": pl.Enum(["pitched"]),
                "roof_insulation": pl.Enum(["loft", "none"]),
                "roof_insulation_mm": pl.Int64(),
                "roof_u_value": pl.Float64(),
                "roof_assumed": pl.Boolean(),
            }
        )
        assert result.row(0)[1:] == ("pitched", "loft", 250, None, False)

    def test_unknown_phrase_raises_value_error_listing_it(self) -> None:
        frame = pl.LazyFrame({"walls_description": ["Mystery wall, filled cavity"]})
        with pytest.raises(ValueError, match=r"1 unknown phrases.*'Mystery wall"):
            parse_column(frame, fabric_parser(walls_vocabulary()))
