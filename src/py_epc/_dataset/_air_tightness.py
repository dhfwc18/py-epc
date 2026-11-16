"""Parse the phrases of a dwelling's air tightness."""

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

# Tests an air permeability may be measured by, as published in brackets,
# such as "[AP50]".
_AIR_TIGHTNESS_TESTS: Final = ("ap50", "ap4")

# Air permeability, its test, value and basis, in English or Welsh.
_AIR_PERMEABILITY_PATTERN: Final = re.compile(
    r"(?:air permeability|athreiddedd aer)\s*"
    rf"(?:\[(?P<test>{'|'.join(_AIR_TIGHTNESS_TESTS)})\]\s*=\s*)?"
    r"(?P<value>-?[0-9]+(?:\.[0-9]+)?)\s*m[3?]/h\.m[2?]\s*"
    r"\((?P<basis>[^)]*)\)",
    re.IGNORECASE,
)

# Basis of an air permeability, keyed by the phrase in its brackets.
_AIR_TIGHTNESS_BASIS_PHRASES: Final = {
    "assumed": "assumed",
    "rhagdybiaeth": "assumed",
    "as tested": "tested",
    "assessed average": "assessed_average",
    "": None,
}

# Phrases saying the air permeability was not tested, given as the basis.
_AIR_TIGHTNESS_NOT_TESTED_PHRASES: Final = frozenset({"(not tested)", "(heb ei brofi)"})
_NOT_TESTED_BASIS: Final = "not_tested"


@dataclass(frozen=True)
class ParsedAirTightness(ParsedPhrase):
    """What one phrase says of a dwelling's air tightness."""

    permeability: float | None = None
    """Air permeability, in cubic metres an hour per square metre."""

    basis: str | None = None
    """How it was found, such as ``"tested"`` or ``"assumed"``."""

    test: str | None = None
    """Test it was measured by, such as ``"ap50"``, if published."""


def parse_air_tightness(text: str | None) -> ParsedAirTightness:
    """
    Return what a cleaned phrase says of a dwelling's air tightness.

    A phrase such as ``"Air permeability [AP50] = 5.4 m3/h.m2 (as tested)"``
    gives the permeability, the test and the basis. ``"(not tested)"`` gives
    only the basis. A permeability of zero or less is invalid. Phrases are
    matched in any case, in English or Welsh.

    Parameters
    ----------
    text : str or None
        A phrase, already cleaned by `clean_text`.

    Returns
    -------
    ParsedAirTightness
        What the phrase says. Its kind is ``"unknown"`` if it matches no
        known phrase, or has an unknown basis or test.
    """
    phrase = prepare_phrase(text)
    if phrase is None:
        return ParsedAirTightness("not_a_description")
    if phrase.casefold() in _AIR_TIGHTNESS_NOT_TESTED_PHRASES:
        return ParsedAirTightness("described", basis=_NOT_TESTED_BASIS)
    match = _AIR_PERMEABILITY_PATTERN.fullmatch(phrase)
    if match is None:
        return ParsedAirTightness("unknown")
    basis_phrase = match["basis"].strip().casefold()
    if basis_phrase not in _AIR_TIGHTNESS_BASIS_PHRASES:
        return ParsedAirTightness("unknown")
    permeability = float(match["value"])
    return ParsedAirTightness(
        "described" if permeability > 0 else "invalid",
        permeability=permeability,
        basis=_AIR_TIGHTNESS_BASIS_PHRASES[basis_phrase],
        test=match["test"].casefold() if match["test"] else None,
    )


def air_tightness_parser() -> ColumnParser[ParsedAirTightness]:
    """
    Return a parser giving a dwelling's air permeability, basis and test.

    The columns given are ``air_tightness_m3_per_h_m2``, then
    ``air_tightness_basis`` and ``air_tightness_test``, each a
    `polars.Enum`.

    Returns
    -------
    ColumnParser
        A parser of ``air_tightness_description`` for `parse_column`.
    """
    bases = [
        *dict.fromkeys(b for b in _AIR_TIGHTNESS_BASIS_PHRASES.values() if b),
        _NOT_TESTED_BASIS,
    ]
    return ColumnParser(
        column="air_tightness_description",
        parse_phrase=parse_air_tightness,
        outputs=(
            OutputColumn(
                "air_tightness_m3_per_h_m2",
                lambda parsed: parsed.permeability,
                pl.Float64(),
            ),
            OutputColumn(
                "air_tightness_basis", lambda parsed: parsed.basis, pl.Enum(bases)
            ),
            OutputColumn(
                "air_tightness_test",
                lambda parsed: parsed.test,
                pl.Enum(list(_AIR_TIGHTNESS_TESTS)),
            ),
        ),
    )
