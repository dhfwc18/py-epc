"""Tests for what every vocabulary has, and reading the shipped ones."""

import json
from typing import Any

import pytest
from pydantic import ValidationError

from py_epc._dataset._vocabulary import (
    Category,
    Vocabulary,
    categories_by_phrase,
    check_categories,
    load_vocabulary,
)

# Fields every vocabulary has.
_VOCABULARY_FIELDS: dict[str, Any] = {
    "version": 1,
    "column": "windows_description",
    "misplaced": ["Solid"],
    "not_a_description": ["SAP05:Windows"],
}


def categories(**phrases: tuple[str, ...]) -> dict[str, Category]:
    """Return categories with the given phrases, labelled by their keys."""
    return {key: Category(label=key, phrases=items) for key, items in phrases.items()}


class TestVocabulary:
    """`Vocabulary` tells the phrases it describes none of."""

    @pytest.mark.parametrize(
        ("phrase", "expected"),
        [
            pytest.param("SAP05:Windows", "not_a_description", id="placeholder"),
            pytest.param("sap05:WINDOWS", "not_a_description", id="any-case"),
            pytest.param(
                "SAP05:Windows, fully", "not_a_description", id="placeholder-first-part"
            ),
            pytest.param("Solid", "misplaced", id="misplaced"),
            pytest.param("Solid, no insulation", "misplaced", id="first-part"),
            pytest.param("Solid brick", None, id="first-part-must-be-whole"),
            pytest.param("Single glazed", None, id="other"),
        ],
    )
    def test_phrase_returns_undescribed_kind(
        self, phrase: str, expected: str | None
    ) -> None:
        vocabulary = Vocabulary.model_validate_json(json.dumps(_VOCABULARY_FIELDS))
        assert vocabulary.undescribed_kind(phrase) == expected

    def test_unknown_field_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs"):
            Vocabulary.model_validate_json(
                json.dumps({**_VOCABULARY_FIELDS, "name": "windows"})
            )

    def test_other_version_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="version"):
            Vocabulary.model_validate_json(
                json.dumps({**_VOCABULARY_FIELDS, "version": 2})
            )


class TestCheckCategories:
    """`check_categories` rejects keys and phrases a vocabulary cannot use."""

    def test_valid_categories_pass(self) -> None:
        check_categories(categories(single=("Single glazed",)), "glazing")

    def test_key_not_snake_case_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="glazing key 'Single' is not snake"):
            check_categories(categories(Single=("Single glazed",)), "glazing")

    def test_phrase_in_two_categories_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="is in 'single' and 'other'"):
            check_categories(
                categories(single=("Single glazed",), other=("single GLAZED",)),
                "glazing",
            )


class TestCategoriesByPhrase:
    """`categories_by_phrase` keys each category by its phrases."""

    def test_phrases_are_casefolded(self) -> None:
        result = categories_by_phrase(
            categories(single=("Single glazed", "Gwydrau sengl"))
        )
        assert result == {"single glazed": "single", "gwydrau sengl": "single"}


class TestLoadVocabulary:
    """`load_vocabulary` reads a shipped vocabulary."""

    def test_unknown_name_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="no vocabulary named 'doors'"):
            load_vocabulary("doors", Vocabulary)

    def test_vocabulary_of_other_kind_raises_validation_error(self) -> None:
        # The shared fields alone reject the fields a kind adds.
        with pytest.raises(ValidationError, match="Extra inputs"):
            load_vocabulary("windows", Vocabulary)
