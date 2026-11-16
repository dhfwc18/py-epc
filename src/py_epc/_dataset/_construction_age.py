"""Parse the years a dwelling was built from its construction age band."""

import re
from dataclasses import dataclass
from typing import Final

import polars as pl

from py_epc._dataset._phrases import (
    ColumnParser,
    OutputColumn,
    ParsedPhrase,
    prepare_phrase,
)

# Columns given: the first and last years the dwelling may have been built.
_AGE_FROM_COLUMN: Final = "construction_age_from"
_AGE_TO_COLUMN: Final = "construction_age_to"

# Column of the date each certificate was lodged.
_LODGEMENT_DATE_COLUMN: Final = "lodgement_date"

# Earliest exact year taken as true. Earlier years in the data, such as 0,
# 200 or 1008 to 1017, are typing errors.
_EARLIEST_YEAR: Final = 1200

# Years after its lodgement a build year may be, as a new dwelling may be
# lodged in the year before it is finished.
_YEARS_AFTER_LODGEMENT: Final = 1

# The register's name some England and Wales bands start with.
_REGISTER_PREFIX_PATTERN: Final = re.compile(r"^england and wales:\s*", re.IGNORECASE)

# Bands of years, such as "1900-1929", "before 1900", "Pre-1900",
# "2008 onwards" and "Post-2006".
_AGE_RANGE_PATTERN: Final = re.compile(r"(?P<from>[0-9]{4})\s*-\s*(?P<to>[0-9]{4})")
_AGE_BEFORE_PATTERN: Final = re.compile(
    r"(?:before\s+|pre-?\s*)(?P<year>[0-9]{4})", re.IGNORECASE
)
_AGE_ONWARDS_PATTERN: Final = re.compile(r"(?P<year>[0-9]{4})\s+onwards", re.IGNORECASE)
_AGE_AFTER_PATTERN: Final = re.compile(r"post-?\s*(?P<year>[0-9]{4})", re.IGNORECASE)

# An exact year, of any number of digits so that errors are recognised.
_EXACT_YEAR_PATTERN: Final = re.compile(r"[0-9]+")

# Phrases that describe no age band.
_AGE_BAND_NOT_A_DESCRIPTION_PHRASES: Final = frozenset({"not applicable"})


@dataclass(frozen=True)
class ParsedAgeBand(ParsedPhrase):
    """The years one published age band covers."""

    from_year: int | None = None
    """First year the dwelling may have been built, or None if open."""

    to_year: int | None = None
    """Last year the dwelling may have been built, or None if open."""


def parse_age_band(text: str | None) -> ParsedAgeBand:
    """
    Return the years a cleaned age band covers.

    A band gives its first and last years, with None for an open end, and an
    exact year gives itself as both. A band whose first year is after its
    last, or an exact year before 1200 or not of four digits, is invalid.

    Parameters
    ----------
    text : str or None
        An age band, already cleaned by `clean_text`, such as
        ``"England and Wales: 1900-1929"``, ``"before 1919"`` or ``"2015"``.

    Returns
    -------
    ParsedAgeBand
        The years. Its kind is ``"unknown"`` if the phrase has no form
        above.

    Examples
    --------
    >>> parse_age_band("England and Wales: before 1900")
    ParsedAgeBand(kind='described', from_year=None, to_year=1899)
    """
    phrase = prepare_phrase(text)
    if phrase is None:
        return ParsedAgeBand("not_a_description")
    phrase = _REGISTER_PREFIX_PATTERN.sub("", phrase).strip()
    if phrase.casefold() in _AGE_BAND_NOT_A_DESCRIPTION_PHRASES:
        return ParsedAgeBand("not_a_description")
    if found := _AGE_RANGE_PATTERN.fullmatch(phrase):
        first, last = int(found["from"]), int(found["to"])
        return ParsedAgeBand("described" if first <= last else "invalid", first, last)
    if found := _AGE_BEFORE_PATTERN.fullmatch(phrase):
        return ParsedAgeBand("described", None, int(found["year"]) - 1)
    if found := _AGE_ONWARDS_PATTERN.fullmatch(phrase):
        return ParsedAgeBand("described", int(found["year"]), None)
    if found := _AGE_AFTER_PATTERN.fullmatch(phrase):
        return ParsedAgeBand("described", int(found["year"]) + 1, None)
    if _EXACT_YEAR_PATTERN.fullmatch(phrase):
        year = int(phrase)
        if len(phrase) != 4 or year < _EARLIEST_YEAR:
            return ParsedAgeBand("invalid", year, year)
        return ParsedAgeBand("described", year, year)
    return ParsedAgeBand("unknown")


def construction_age_parser() -> ColumnParser[ParsedAgeBand]:
    """
    Return a parser giving the years each dwelling may have been built.

    The columns given are ``construction_age_from`` and
    ``construction_age_to``, null at an open end. Use
    `null_construction_age_after_lodgement` after `parse_column` to drop
    years the certificate's lodgement rules out.

    Returns
    -------
    ColumnParser
        A parser of ``construction_age_band`` for `parse_column`.
    """
    return ColumnParser(
        column="construction_age_band",
        parse_phrase=parse_age_band,
        outputs=(
            OutputColumn(_AGE_FROM_COLUMN, lambda parsed: parsed.from_year, pl.Int64()),
            OutputColumn(_AGE_TO_COLUMN, lambda parsed: parsed.to_year, pl.Int64()),
        ),
    )


def null_construction_age_after_lodgement(frame: pl.LazyFrame) -> pl.LazyFrame:
    """
    Return `frame` with no construction age starting after its lodgement.

    A dwelling may be lodged in the year before it is finished, so a band
    starting more than a year after ``lodgement_date`` cannot be true. Both
    of its years are given null.

    Parameters
    ----------
    frame : polars.LazyFrame
        Rows with ``lodgement_date``, and the columns of
        `construction_age_parser`.

    Returns
    -------
    polars.LazyFrame
        The rows, in the same order.
    """
    latest = pl.col(_LODGEMENT_DATE_COLUMN).dt.year() + _YEARS_AFTER_LODGEMENT
    from_year = pl.col(_AGE_FROM_COLUMN)
    possible = from_year.is_null() | (from_year <= latest)
    return frame.with_columns(
        pl.when(possible).then(pl.col(column)).alias(column)
        for column in (_AGE_FROM_COLUMN, _AGE_TO_COLUMN)
    )
