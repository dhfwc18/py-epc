"""Put the phrases of a column into categories listed by a vocabulary."""

from dataclasses import dataclass
from typing import Literal, Self

import polars as pl
from pydantic import PrivateAttr, model_validator

from py_epc._dataset._phrases import (
    ColumnParser,
    OutputColumn,
    ParsedPhrase,
    prepare_phrase,
)
from py_epc._dataset._vocabulary import (
    Category,
    Vocabulary,
    categories_by_phrase,
    check_categories,
)


class PhraseVocabulary(Vocabulary):
    """The categories of one column, each matched by whole phrases."""

    format: Literal["phrase"]
    """Kind of vocabulary: one category for each whole phrase."""

    output_column: str
    """
    Column the category is given as, such as ``"windows_glazing"``. For a
    column of codes, such as ``"tenure"``, it is the column itself, which is
    replaced in place.
    """

    categories: dict[str, Category]
    """Categories, keyed by name, such as ``"fully_double"``."""

    _categories_by_phrase: dict[str, str] = PrivateAttr()

    @model_validator(mode="after")
    def _check(self) -> Self:
        check_categories(self.categories, self.output_column)
        undescribed = {p.casefold() for p in self.misplaced + self.not_a_description}
        described = categories_by_phrase(self.categories)
        if repeated := sorted(undescribed & described.keys()):
            raise ValueError(f"phrases both in a category and not: {repeated}")
        return self

    def model_post_init(self, context: object, /) -> None:
        super().model_post_init(context)
        self._categories_by_phrase = categories_by_phrase(self.categories)

    def category_of(self, phrase: str) -> str | None:
        """Return the key of the category `phrase` means, in any case."""
        return self._categories_by_phrase.get(phrase.casefold())


@dataclass(frozen=True)
class ParsedCategory(ParsedPhrase):
    """The category of one phrase."""

    category: str | None = None
    """Key of the phrase's category, such as ``"fully_double"``."""


def parse_category(text: str | None, vocabulary: PhraseVocabulary) -> ParsedCategory:
    """
    Return the category of a cleaned phrase.

    The whole phrase is matched, in any case, after `prepare_phrase`.

    Parameters
    ----------
    text : str or None
        A phrase, already cleaned by `clean_text`.
    vocabulary : PhraseVocabulary
        Categories of the column.

    Returns
    -------
    ParsedCategory
        The phrase's category. Its kind is ``"unknown"`` if the phrase is in
        no list of the vocabulary.
    """
    phrase = prepare_phrase(text)
    if phrase is None:
        return ParsedCategory("not_a_description")
    if kind := vocabulary.undescribed_kind(phrase):
        return ParsedCategory(kind)
    category = vocabulary.category_of(phrase)
    if category is None:
        return ParsedCategory("unknown")
    return ParsedCategory("described", category)


def category_parser(vocabulary: PhraseVocabulary) -> ColumnParser[ParsedCategory]:
    """
    Return a parser giving the category of each phrase of a column.

    The category is given as `PhraseVocabulary.output_column`, which
    replaces the column in place if it has the column's name.

    Parameters
    ----------
    vocabulary : PhraseVocabulary
        Categories of the column.

    Returns
    -------
    ColumnParser
        A parser for `parse_column`.
    """

    def parse_phrase(text: str | None) -> ParsedCategory:
        return parse_category(text, vocabulary)

    return ColumnParser(
        column=vocabulary.column,
        parse_phrase=parse_phrase,
        outputs=(
            OutputColumn(
                vocabulary.output_column, lambda parsed: parsed.category, pl.String()
            ),
        ),
        replaces_column=vocabulary.output_column == vocabulary.column,
    )
