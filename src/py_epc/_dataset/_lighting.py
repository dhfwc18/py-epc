"""Parse the phrases of a dwelling's lighting."""

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

# Share of fixed outlets with low energy lighting, in English or Welsh.
_LIGHTING_PERCENT_PATTERN: Final = re.compile(
    r"(?:low energy lighting|goleuadau ynni-isel)\s+(?:in\s+|mewn\s+)?"
    r"(?P<percent>[0-9]+(?:\.[0-9]+)?)\s*%.*",
    re.IGNORECASE,
)

# Phrases saying all fixed outlets, or none, have low energy lighting.
_LIGHTING_ALL_PHRASES: Final = frozenset(
    {
        "low energy lighting in all fixed outlets",
        "goleuadau ynni-isel ym mhob un o'r mannau gosod",
        "goleuadau ynni-isel ym mhob un o?r mannau gosod",
    }
)
_LIGHTING_NONE_PHRASES: Final = frozenset(
    {"no low energy lighting", "dim goleuadau ynni-isel"}
)

# Lighting efficiency, given instead of a share by newer assessments, keyed
# by phrase. Its values are listed in order, best first.
_LIGHTING_EFFICIENCY_PHRASES: Final = {
    "excellent lighting efficiency": "excellent",
    "excelent lighting efficiency": "excellent",
    "effeithlonrwydd goleuo rhagorol": "excellent",
    "good lighting efficiency": "good",
    "effeithlonrwydd goleuo da": "good",
    "below average lighting efficiency": "below_average",
    "effeithlonrwydd goleuo is na'r cyfartaledd": "below_average",
}

# Phrases that describe no lighting, such as placeholders and pieces of a
# Welsh phrase.
_LIGHTING_NOT_A_DESCRIPTION_PHRASES: Final = frozenset(
    {"sap05:lighting", "'r mannau gosod", "r mannau gosod", "***sample***"}
)


@dataclass(frozen=True)
class ParsedLighting(ParsedPhrase):
    """What one phrase says of a dwelling's lighting."""

    low_energy_percent: float | None = None
    """Share of fixed outlets with low energy lighting, from 0 to 100."""

    efficiency: str | None = None
    """Lighting efficiency, such as ``"good"``, from newer assessments."""


def parse_lighting(text: str | None) -> ParsedLighting:
    """
    Return what a cleaned phrase says of a dwelling's lighting.

    A share is read from phrases such as ``"Low energy lighting in 50% of
    fixed outlets"``, with all outlets as 100 and none as 0. A share above
    100 is invalid. Newer assessments give an efficiency instead, such as
    ``"Good lighting efficiency"``. Phrases are matched in any case, in
    English or Welsh.

    Parameters
    ----------
    text : str or None
        A phrase, already cleaned by `clean_text`.

    Returns
    -------
    ParsedLighting
        What the phrase says. Its kind is ``"unknown"`` if it matches no
        known phrase.
    """
    phrase = prepare_phrase(text)
    if phrase is None:
        return ParsedLighting("not_a_description")
    folded = phrase.casefold()
    if folded in _LIGHTING_NOT_A_DESCRIPTION_PHRASES:
        return ParsedLighting("not_a_description")
    if folded in _LIGHTING_ALL_PHRASES:
        return ParsedLighting("described", low_energy_percent=100.0)
    if folded in _LIGHTING_NONE_PHRASES:
        return ParsedLighting("described", low_energy_percent=0.0)
    if efficiency := _LIGHTING_EFFICIENCY_PHRASES.get(folded):
        return ParsedLighting("described", efficiency=efficiency)
    if match := _LIGHTING_PERCENT_PATTERN.fullmatch(phrase):
        percent = float(match["percent"])
        return ParsedLighting(
            "described" if percent <= 100 else "invalid", low_energy_percent=percent
        )
    return ParsedLighting("unknown")


def lighting_parser() -> ColumnParser[ParsedLighting]:
    """
    Return a parser giving the share and efficiency of a dwelling's lighting.

    The columns given are ``lighting_low_energy_percent`` and
    ``lighting_efficiency``, a `polars.Enum` listed best first.

    Returns
    -------
    ColumnParser
        A parser of ``lighting_description`` for `parse_column`.
    """
    efficiencies = list(dict.fromkeys(_LIGHTING_EFFICIENCY_PHRASES.values()))
    return ColumnParser(
        column="lighting_description",
        parse_phrase=parse_lighting,
        outputs=(
            OutputColumn(
                "lighting_low_energy_percent",
                lambda parsed: parsed.low_energy_percent,
                pl.Float64(),
            ),
            OutputColumn(
                "lighting_efficiency",
                lambda parsed: parsed.efficiency,
                pl.Enum(efficiencies),
            ),
        ),
    )
