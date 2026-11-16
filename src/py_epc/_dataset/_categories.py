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

    format: Literal["phrase", "phrase_list"]
    """The format of vocabulary."""

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
    replaces the column in place if it has the column's name. Categories
    are a `polars.Enum`, listed in the vocabulary's order.

    Parameters
    ----------
    vocabulary : PhraseVocabulary
        Categories of the column.

    Returns
    -------
    ColumnParser
        A parser for `parse_column`.

    Raises
    ------
    ValueError
        If the vocabulary is not of format ``"phrase"``.
    """
    _check_format(vocabulary, "phrase")

    def parse_phrase(text: str | None) -> ParsedCategory:
        return parse_category(text, vocabulary)

    return ColumnParser(
        column=vocabulary.column,
        parse_phrase=parse_phrase,
        outputs=(
            OutputColumn(
                vocabulary.output_column,
                lambda parsed: parsed.category,
                pl.Enum(list(vocabulary.categories)),
            ),
        ),
        replaces_column=vocabulary.output_column == vocabulary.column,
    )


@dataclass(frozen=True)
class ParsedCategoryList(ParsedPhrase):
    """The categories of one list of phrases."""

    categories: tuple[str, ...] = ()
    """Keys of the items' categories, in published order, each once."""


def parse_category_list(
    text: str | None, vocabulary: PhraseVocabulary
) -> ParsedCategoryList:
    """
    Return the categories of a cleaned list of phrases.

    Items are separated by ``"|"``, and each is matched as a whole phrase
    by `parse_category`. An item that is not a description is skipped, and
    a misplaced or unknown item makes the whole list so.

    Parameters
    ----------
    text : str or None
        A list of phrases, already cleaned by `clean_text`, such as
        ``"Description: Air source heat pump | Description: Wind turbine |"``.
    vocabulary : PhraseVocabulary
        Categories of the column's items.

    Returns
    -------
    ParsedCategoryList
        The items' categories. Its kind is ``"unknown"`` if an item is in no
        list of the vocabulary.
    """
    if text is None:
        return ParsedCategoryList("not_a_description")
    categories: list[str] = []
    for item in text.split("|"):
        parsed = parse_category(item, vocabulary)
        if parsed.kind == "not_a_description":
            continue
        if parsed.kind != "described" or parsed.category is None:
            return ParsedCategoryList(parsed.kind)
        if parsed.category not in categories:
            categories.append(parsed.category)
    if not categories:
        return ParsedCategoryList("not_a_description")
    return ParsedCategoryList("described", tuple(categories))


def category_list_parser(
    vocabulary: PhraseVocabulary,
) -> ColumnParser[ParsedCategoryList]:
    """
    Return a parser giving the categories of each list of phrases.

    The categories are given as `PhraseVocabulary.output_column`, a list
    of a `polars.Enum`, which replaces the column in place if it has the
    column's name.

    Parameters
    ----------
    vocabulary : PhraseVocabulary
        Categories of the column's items, of format ``"phrase_list"``.

    Returns
    -------
    ColumnParser
        A parser for `parse_column`.

    Raises
    ------
    ValueError
        If the vocabulary is not of format ``"phrase_list"``.
    """
    _check_format(vocabulary, "phrase_list")

    def parse_phrase(text: str | None) -> ParsedCategoryList:
        return parse_category_list(text, vocabulary)

    return ColumnParser(
        column=vocabulary.column,
        parse_phrase=parse_phrase,
        outputs=(
            OutputColumn(
                vocabulary.output_column,
                lambda parsed: list(parsed.categories),
                pl.List(pl.Enum(list(vocabulary.categories))),
            ),
        ),
        replaces_column=vocabulary.output_column == vocabulary.column,
    )


def _check_format(vocabulary: PhraseVocabulary, expected: str) -> None:
    """Raise if `vocabulary` is not of the format its parser reads."""
    if vocabulary.format != expected:
        raise ValueError(
            f"vocabulary of {vocabulary.column!r} is of format"
            f" {vocabulary.format!r}, not {expected!r}"
        )
