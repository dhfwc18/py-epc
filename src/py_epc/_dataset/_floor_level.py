"""Parse the floor a flat or maisonette is on from its floor level."""

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

# A numbered floor, such as "1st", or "21st or above", which gives 21.
_FLOOR_ORDINAL_PATTERN: Final = re.compile(
    r"(?P<number>[0-9]+)(?:st|nd|rd|th)(?:\s+or above)?", re.IGNORECASE
)

# Positions a floor may have in its building, from lowest to highest.
_FLOOR_POSITIONS: Final = ("basement", "ground", "mid", "top")

# Floors named by their position, keyed by phrase, with the floor's number.
_FLOOR_POSITION_PHRASES: Final[dict[str, tuple[int | None, str]]] = {
    "basement": (-1, "basement"),
    "ground": (0, "ground"),
    "ground floor": (0, "ground"),
    "mid floor": (None, "mid"),
    "top floor": (None, "top"),
}


@dataclass(frozen=True)
class ParsedFloorLevel(ParsedPhrase):
    """The floor one published floor level gives."""

    number: int | None = None
    """Number of the floor, with the ground floor 0 and a basement -1."""

    position: str | None = None
    """Position of the floor in its building, such as ``"top"``."""


def parse_floor_level(text: str | None) -> ParsedFloorLevel:
    """
    Return the floor a cleaned floor level gives.

    A numbered floor gives its number only, as it may be a middle or top
    floor. ``"21st or above"`` gives 21. A floor named by its position,
    such as ``"Ground"``, gives the position, and the number where the name
    gives it.

    Parameters
    ----------
    text : str or None
        A floor level, already cleaned by `clean_text`, such as ``"2nd"``
        or ``"top floor"``.

    Returns
    -------
    ParsedFloorLevel
        The floor. Its kind is ``"unknown"`` if the phrase has no form
        above.

    Examples
    --------
    >>> parse_floor_level("Ground")
    ParsedFloorLevel(kind='described', number=0, position='ground')
    """
    phrase = prepare_phrase(text)
    if phrase is None:
        return ParsedFloorLevel("not_a_description")
    if found := _FLOOR_ORDINAL_PATTERN.fullmatch(phrase):
        return ParsedFloorLevel("described", int(found["number"]))
    if (named := _FLOOR_POSITION_PHRASES.get(phrase.casefold())) is not None:
        number, position = named
        return ParsedFloorLevel("described", number, position)
    return ParsedFloorLevel("unknown")


def floor_level_parser() -> ColumnParser[ParsedFloorLevel]:
    """
    Return a parser giving the floor each dwelling is on.

    The columns given are ``floor_level_number`` and
    ``floor_level_position``, a `polars.Enum` listed from lowest to
    highest.

    Returns
    -------
    ColumnParser
        A parser of ``floor_level`` for `parse_column`.
    """
    return ColumnParser(
        column="floor_level",
        parse_phrase=parse_floor_level,
        outputs=(
            OutputColumn(
                "floor_level_number", lambda parsed: parsed.number, pl.Int64()
            ),
            OutputColumn(
                "floor_level_position",
                lambda parsed: parsed.position,
                pl.Enum(list(_FLOOR_POSITIONS)),
            ),
        ),
    )
