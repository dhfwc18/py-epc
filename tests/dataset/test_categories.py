"""Tests for putting the phrases of a column into categories."""

import json
from typing import Any

import polars as pl
import pytest
from pydantic import ValidationError

from py_epc._dataset._categories import (
    ParsedCategory,
    PhraseVocabulary,
    category_parser,
    parse_category,
)
from py_epc._dataset._phrases import parse_column
from py_epc._dataset._vocabulary import load_vocabulary

# A small windows vocabulary, laid out as the shipped one is.
_WINDOWS_VOCABULARY: dict[str, Any] = {
    "version": 1,
    "format": "phrase",
    "column": "windows_description",
    "output_column": "windows_glazing",
    "misplaced": ["Solid"],
    "not_a_description": ["SAP05:Windows"],
    "categories": {
        "fully_double": {
            "label": "Fully double glazed",
            "phrases": ["Fully double glazed", "Gwydrau dwbl llawn"],
        },
        "single": {"label": "Single glazed", "phrases": ["Single glazed"]},
    },
}

# A small vocabulary of a column of codes, replaced in place.
_TENURE_VOCABULARY: dict[str, Any] = {
    "version": 1,
    "format": "phrase",
    "column": "tenure",
    "output_column": "tenure",
    "not_a_description": ["unknown"],
    "categories": {
        "owner_occupied": {"label": "Owner-occupied", "phrases": ["owner-occupied"]},
    },
}

# Shipped phrase vocabularies, and the column each describes.
_SHIPPED_COLUMNS: dict[str, str] = {
    "windows": "windows_description",
    "mainheatcont": "mainheatcont_description",
    **{
        name: name
        for name in (
            "built_form",
            "energy_tariff",
            "glazed_area",
            "glazed_type",
            "heat_loss_corridor",
            "main_heating_category",
            "mechanical_ventilation",
            "property_type",
            "tenure",
            "transaction_type",
        )
    },
}


def windows_vocabulary(**changes: Any) -> PhraseVocabulary:
    """Return the small windows vocabulary, with fields changed."""
    return PhraseVocabulary.model_validate_json(
        json.dumps({**_WINDOWS_VOCABULARY, **changes})
    )


class TestPhraseVocabulary:
    """`PhraseVocabulary` checks a vocabulary when it is made."""

    @pytest.mark.parametrize(("name", "column"), _SHIPPED_COLUMNS.items())
    def test_shipped_vocabulary_is_valid(self, name: str, column: str) -> None:
        assert load_vocabulary(name, PhraseVocabulary).column == column

    def test_phrase_in_two_categories_raises_validation_error(self) -> None:
        categories = {
            "single": {"label": "a", "phrases": ["Single glazed"]},
            "other": {"label": "b", "phrases": ["single GLAZED"]},
        }
        with pytest.raises(ValidationError, match="is in 'single' and 'other'"):
            windows_vocabulary(categories=categories)

    def test_phrase_both_in_category_and_not_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match=r"both in a category.*'solid'"):
            windows_vocabulary(
                categories={"solid": {"label": "Solid", "phrases": ["SOLID"]}}
            )

    def test_other_format_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="format"):
            windows_vocabulary(format="fabric")


class TestParseCategory:
    """`parse_category` gives the category of a phrase."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            pytest.param(
                "Fully double glazed",
                ParsedCategory("described", "fully_double"),
                id="english",
            ),
            pytest.param(
                "Gwydrau dwbl llawn",
                ParsedCategory("described", "fully_double"),
                id="welsh",
            ),
            pytest.param(
                "single GLAZED", ParsedCategory("described", "single"), id="any-case"
            ),
            pytest.param(
                "Description: Single glazed",
                ParsedCategory("described", "single"),
                id="description-prefix",
            ),
            pytest.param("Solid", ParsedCategory("misplaced"), id="misplaced"),
            pytest.param(
                "Solid, no insulation",
                ParsedCategory("misplaced"),
                id="misplaced-first-part",
            ),
            pytest.param(
                "SAP05:Windows", ParsedCategory("not_a_description"), id="placeholder"
            ),
            pytest.param(
                "????????", ParsedCategory("not_a_description"), id="question-marks"
            ),
            pytest.param(None, ParsedCategory("not_a_description"), id="none"),
            pytest.param("Stained glass", ParsedCategory("unknown"), id="unknown"),
        ],
    )
    def test_phrase_returns_category(
        self, text: str | None, expected: ParsedCategory
    ) -> None:
        assert parse_category(text, windows_vocabulary()) == expected


class TestCategoryParser:
    """`category_parser` gives a column's categories through `parse_column`."""

    def test_category_column_is_added_after_description(self) -> None:
        frame = pl.LazyFrame(
            {"windows_description": ["Single glazed", None, "SAP05:Windows"]}
        )
        parser = category_parser(windows_vocabulary())
        result = parse_column(frame, parser).collect()
        assert result.to_dict(as_series=False) == {
            "windows_description": ["Single glazed", None, "SAP05:Windows"],
            "windows_glazing": ["single", None, None],
        }

    def test_category_type_is_enum_of_keys_in_vocabulary_order(self) -> None:
        frame = pl.LazyFrame({"windows_description": ["Single glazed"]})
        result = parse_column(frame, category_parser(windows_vocabulary())).collect()
        assert result.schema["windows_glazing"] == pl.Enum(["fully_double", "single"])

    def test_column_of_codes_is_replaced_in_place(self) -> None:
        vocabulary = PhraseVocabulary.model_validate_json(
            json.dumps(_TENURE_VOCABULARY)
        )
        frame = pl.LazyFrame({"tenure": ["owner-occupied", "unknown"], "row": [1, 2]})
        result = parse_column(frame, category_parser(vocabulary)).collect()
        assert result.to_dict(as_series=False) == {
            "tenure": ["owner_occupied", None],
            "row": [1, 2],
        }

    def test_unknown_phrase_raises_value_error_listing_it(self) -> None:
        frame = pl.LazyFrame({"windows_description": ["Stained glass"]})
        with pytest.raises(ValueError, match=r"1 unknown phrases.*'Stained glass'"):
            parse_column(frame, category_parser(windows_vocabulary()))
