"""Parse the phrases of a building's fabric: its walls, roof and floor."""

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal, Self

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

# A U-value published instead of a description, in English or Welsh. The
# sign is read so that a negative value is found; "NaN" means none was given.
_U_VALUE_PATTERN: Final = re.compile(
    r"(?:average thermal transmittance|trawsyriannedd thermol cyfartalog)"
    r"\s*=?\s*(?P<value>-?[0-9]+(?:\.[0-9]+)?|nan)\s*w/m2k",
    re.IGNORECASE,
)

# A note that the assessor assumed the element rather than inspected it, in
# English or Welsh.
_ASSUMED_PATTERN: Final = re.compile(r"\((?:assumed|rhagdybiaeth)\)", re.IGNORECASE)

# A thickness of insulation, such as "250 mm", "300+ mm" or ">= 300 mm".
_THICKNESS_PATTERN: Final = re.compile(
    r"(?:>=\s*)?(?P<mm>[0-9]+)\s*\+?\s*mm\b", re.IGNORECASE
)


class FabricVocabulary(Vocabulary):
    """The phrases of one element of a building's fabric, such as its walls."""

    format: Literal["fabric"]
    """Kind of vocabulary: a construction, then its insulation."""

    prefix: str
    """Start of the names of the columns given, such as ``"walls"`` for
    ``"walls_construction"``."""

    construction: dict[str, Category]
    """Constructions, keyed by name, matched by a phrase's first part."""

    insulation: dict[str, Category]
    """Insulation, keyed by name, matched by the rest of the phrase."""

    insulation_has_thickness: bool = False
    """Whether insulation may give a thickness, such as ``"250 mm loft
    insulation"``, given as ``<prefix>_insulation_mm``."""

    _construction_by_phrase: dict[str, str] = PrivateAttr()
    _insulation_by_phrase: dict[str, str] = PrivateAttr()

    @model_validator(mode="after")
    def _check(self) -> Self:
        check_categories(self.construction, "construction")
        check_categories(self.insulation, "insulation")
        return self

    def model_post_init(self, context: object, /) -> None:
        super().model_post_init(context)
        self._construction_by_phrase = categories_by_phrase(self.construction)
        self._insulation_by_phrase = categories_by_phrase(self.insulation)

    def construction_of(self, part: str) -> str | None:
        """Return the key of the construction `part` means, in any case."""
        return self._construction_by_phrase.get(part.casefold())

    def insulation_of(self, part: str) -> str | None:
        """Return the key of the insulation `part` means, in any case."""
        return self._insulation_by_phrase.get(part.casefold())


@dataclass(frozen=True)
class ParsedFabric(ParsedPhrase):
    """What one phrase says of an element of a building's fabric."""

    construction: str | None = None
    """Key of the construction, such as ``"cavity"``."""

    insulation: str | None = None
    """Key of the insulation, such as ``"filled_cavity"``."""

    insulation_mm: int | None = None
    """Thickness of the insulation, in millimetres, if given."""

    u_value: float | None = None
    """U-value, in W/m2K, if published instead of a description."""

    assumed: bool | None = None
    """Whether the assessor assumed the element rather than inspected it."""


def parse_fabric(text: str | None, vocabulary: FabricVocabulary) -> ParsedFabric:
    """
    Return what a cleaned phrase says of an element.

    A phrase is a construction, then after the first comma its insulation,
    such as ``"Cavity wall, as built, no insulation (assumed)"``, where
    ``"(assumed)"`` sets `ParsedFabric.assumed`. It may instead be a
    U-value, such as ``"Average thermal transmittance 0.25 W/m2K"``. A
    U-value of zero or less is invalid. Phrases are matched in any case.

    Parameters
    ----------
    text : str or None
        A phrase, already cleaned by `clean_text`.
    vocabulary : FabricVocabulary
        Phrases of the element.

    Returns
    -------
    ParsedFabric
        What the phrase says. Its kind is ``"unknown"`` if a part is in no
        list of the vocabulary.
    """
    phrase = prepare_phrase(text)
    if phrase is None:
        return ParsedFabric("not_a_description")
    if match := _U_VALUE_PATTERN.fullmatch(phrase):
        if match["value"].casefold() == "nan":
            return ParsedFabric("not_a_description")
        u_value = float(match["value"])
        return ParsedFabric("described" if u_value > 0 else "invalid", u_value=u_value)
    assumed = bool(_ASSUMED_PATTERN.search(phrase))
    # Only a trailing comma is removed. A leading comma marks an empty first
    # part, which must stay empty.
    phrase = _ASSUMED_PATTERN.sub("", phrase).strip().rstrip(",").strip()
    if not phrase:
        return ParsedFabric("not_a_description")
    if kind := vocabulary.undescribed_kind(phrase):
        return ParsedFabric(kind)
    head, _, rest = (part.strip() for part in phrase.partition(","))
    # A phrase of insulation alone, such as "no insulation", says nothing of
    # the construction.
    if not rest and vocabulary.construction_of(head) is None:
        head, rest = "", head
    insulation_mm = None
    if vocabulary.insulation_has_thickness and (
        thickness := _THICKNESS_PATTERN.search(rest)
    ):
        insulation_mm = int(thickness["mm"])
        rest = (rest[: thickness.start()] + rest[thickness.end() :]).strip()
    construction = vocabulary.construction_of(head) if head else None
    stated = bool(rest) and not vocabulary.is_not_a_description(rest)
    insulation = vocabulary.insulation_of(rest) if stated else None
    if (head and construction is None) or (stated and insulation is None):
        return ParsedFabric("unknown")
    return ParsedFabric(
        "described",
        construction=construction,
        insulation=insulation,
        insulation_mm=insulation_mm,
        assumed=assumed,
    )


def fabric_parser(vocabulary: FabricVocabulary) -> ColumnParser[ParsedFabric]:
    """
    Return a parser giving what each phrase says of an element.

    The columns given, each named after `FabricVocabulary.prefix`, are
    ``<prefix>_construction`` and ``<prefix>_insulation``, as a
    `polars.Enum` listed in the vocabulary's order, then
    ``<prefix>_insulation_mm`` if the vocabulary has
    `FabricVocabulary.insulation_has_thickness`, then ``<prefix>_u_value``
    and ``<prefix>_assumed``.

    Parameters
    ----------
    vocabulary : FabricVocabulary
        Phrases of the element.

    Returns
    -------
    ColumnParser
        A parser for `parse_column`.
    """

    def parse_phrase(text: str | None) -> ParsedFabric:
        return parse_fabric(text, vocabulary)

    def output(
        field: str, read_value: Callable[[ParsedFabric], object], dtype: pl.DataType
    ) -> OutputColumn[ParsedFabric]:
        return OutputColumn(f"{vocabulary.prefix}_{field}", read_value, dtype)

    thickness = (
        (output("insulation_mm", lambda parsed: parsed.insulation_mm, pl.Int64()),)
        if vocabulary.insulation_has_thickness
        else ()
    )
    return ColumnParser(
        column=vocabulary.column,
        parse_phrase=parse_phrase,
        outputs=(
            output(
                "construction",
                lambda parsed: parsed.construction,
                pl.Enum(list(vocabulary.construction)),
            ),
            output(
                "insulation",
                lambda parsed: parsed.insulation,
                pl.Enum(list(vocabulary.insulation)),
            ),
            *thickness,
            output("u_value", lambda parsed: parsed.u_value, pl.Float64()),
            output("assumed", lambda parsed: parsed.assumed, pl.Boolean()),
        ),
    )
