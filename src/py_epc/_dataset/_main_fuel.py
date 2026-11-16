"""Parse the main fuel, and whether it feeds a community heating scheme."""

import re
from dataclasses import dataclass
from typing import Final

import polars as pl

from py_epc._dataset._categories import (
    ParsedCategory,
    PhraseVocabulary,
    parse_category,
)
from py_epc._dataset._phrases import ColumnParser, OutputColumn, prepare_phrase

# How a phrase says whether its fuel feeds a community scheme.
_COMMUNITY_PATTERN: Final = re.compile(
    r"\((?P<not_community>not )?community\)$|^community heating schemes:",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedMainFuel(ParsedCategory):
    """The main fuel of one phrase, and whether it feeds a community scheme."""

    community: bool | None = None
    """Whether the fuel feeds a community scheme, if the phrase says."""


def parse_main_fuel(text: str | None, vocabulary: PhraseVocabulary) -> ParsedMainFuel:
    """
    Return the main fuel of a cleaned phrase, and whether it is community.

    The fuel is the phrase's category, as `parse_category` gives it. The
    phrase says the fuel feeds a community scheme if it ends in
    ``"(community)"`` or starts with ``"Community heating schemes:"``, and
    says it does not if it ends in ``"(not community)"``.

    Parameters
    ----------
    text : str or None
        A phrase, already cleaned by `clean_text`.
    vocabulary : PhraseVocabulary
        Categories of the main fuel.

    Returns
    -------
    ParsedMainFuel
        The phrase's fuel and community flag. Its kind is ``"unknown"`` if
        the phrase is in no list of the vocabulary.

    Examples
    --------
    >>> from py_epc._dataset._vocabulary import load_vocabulary
    >>> vocabulary = load_vocabulary("main_fuel", PhraseVocabulary)
    >>> parse_main_fuel("mains gas (not community)", vocabulary)
    ParsedMainFuel(kind='described', category='mains_gas', community=False)
    """
    parsed = parse_category(text, vocabulary)
    if parsed.kind != "described":
        return ParsedMainFuel(parsed.kind)
    phrase = prepare_phrase(text) or ""
    found = _COMMUNITY_PATTERN.search(phrase)
    community = None if found is None else found["not_community"] is None
    return ParsedMainFuel("described", parsed.category, community)


def main_fuel_parser(vocabulary: PhraseVocabulary) -> ColumnParser[ParsedMainFuel]:
    """
    Return a parser giving the main fuel and its community flag.

    The fuel is given as `PhraseVocabulary.output_column`, as
    `category_parser` gives it, and ``main_fuel_community`` is added after
    the frame's columns.

    Parameters
    ----------
    vocabulary : PhraseVocabulary
        Categories of the main fuel.

    Returns
    -------
    ColumnParser
        A parser for `parse_column`.
    """

    def parse_phrase(text: str | None) -> ParsedMainFuel:
        return parse_main_fuel(text, vocabulary)

    return ColumnParser(
        column=vocabulary.column,
        parse_phrase=parse_phrase,
        outputs=(
            OutputColumn(
                vocabulary.output_column,
                lambda parsed: parsed.category,
                pl.Enum(list(vocabulary.categories)),
            ),
            OutputColumn(
                "main_fuel_community", lambda parsed: parsed.community, pl.Boolean()
            ),
        ),
        replaces_column=vocabulary.output_column == vocabulary.column,
    )
