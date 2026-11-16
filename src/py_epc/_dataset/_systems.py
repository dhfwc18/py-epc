"""Parse the phrases of a dwelling's systems, such as its main heating."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final, Literal, Self

import polars as pl
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from py_epc._dataset._models import MODEL_CONFIG
from py_epc._dataset._phrases import (
    ColumnParser,
    OutputColumn,
    ParsedPhrase,
    prepare_phrase,
)
from py_epc._dataset._vocabulary import Vocabulary

# Start of the names of each system's columns after the prefix, in order, so
# the second system's fuel is "<prefix>_second_fuel".
_SYSTEM_COLUMN_PREFIXES: Final = ("", "second_")


class SystemPart(BaseModel):
    """Phrases for one part of a system, and the fields each sets."""

    model_config = MODEL_CONFIG

    phrases: tuple[str, ...]
    """Published phrases for the part, in any case."""

    values: dict[str, str]
    """Fields the part sets, such as ``{"system": "boiler"}``."""

    otherwise: dict[str, str] | None = None
    """
    Fields it sets instead if `values` would overwrite the current system,
    such as warm air being an emitter after a heat pump. Empty if the part
    then adds nothing, such as a general fuel after a specific one.
    """

    assumed: bool = False
    """Whether the part says the assessor assumed it rather than inspected
    it."""


class SystemVocabulary(Vocabulary):
    """
    The parts of the phrases of one system, such as the main heating.

    A phrase is parts separated by commas. A part listed as misplaced makes
    the phrase misplaced, and a part listed as no description is skipped.
    """

    format: Literal["system"]
    """Kind of vocabulary: parts, each setting fields of a system."""

    prefix: str
    """Start of the names of the columns given, such as ``"mainheat"`` for
    ``"mainheat_fuel"``."""

    fields: tuple[str, ...]
    """Fields a system has, such as ``"system"``, ``"emitter"`` and
    ``"fuel"``. The first names the system, so naming it again starts
    another."""

    labels: dict[str, dict[str, str]]
    """Readable name of each value, keyed by field, then by value."""

    max_systems: int = Field(default=1, ge=1, le=len(_SYSTEM_COLUMN_PREFIXES))
    """Most systems a phrase may describe, such as 2 for main heating."""

    parts: tuple[SystemPart, ...]
    """Every part, and the fields it sets."""

    _parts_by_phrase: dict[str, SystemPart] = PrivateAttr()

    @model_validator(mode="after")
    def _check(self) -> Self:
        if set(self.labels) != set(self.fields):
            raise ValueError(f"labels are for {sorted(self.labels)}, not every field")
        seen: set[str] = set()
        for part in self.parts:
            for values in (part.values, part.otherwise or {}):
                for name, value in values.items():
                    if name not in self.fields:
                        raise ValueError(f"part {part.phrases[0]!r} sets {name!r}")
                    if value not in self.labels[name]:
                        raise ValueError(f"{name} value {value!r} has no label")
            for phrase in part.phrases:
                if phrase.casefold() in seen:
                    raise ValueError(f"phrase {phrase!r} is in two parts")
                seen.add(phrase.casefold())
        return self

    def model_post_init(self, context: object, /) -> None:
        super().model_post_init(context)
        self._parts_by_phrase = {
            phrase.casefold(): part for part in self.parts for phrase in part.phrases
        }

    def part_of(self, phrase: str) -> SystemPart | None:
        """Return the part with `phrase` among its phrases, in any case."""
        return self._parts_by_phrase.get(phrase.casefold())


@dataclass(frozen=True)
class ParsedSystems(ParsedPhrase):
    """What one phrase says of a dwelling's systems."""

    systems: tuple[Mapping[str, str], ...] = ()
    """Fields of each system, in published order."""

    assumed: bool | None = None
    """Whether the assessor assumed a system rather than inspected it."""


def parse_systems(text: str | None, vocabulary: SystemVocabulary) -> ParsedSystems:
    """
    Return what a cleaned phrase says of a dwelling's systems.

    A phrase is parts separated by commas, such as ``"Boiler and radiators,
    mains gas"``. Each part sets fields of the current system. A part that
    would change a field already set, or names the system again, starts the
    next system, unless it has `SystemPart.otherwise` values that fit. Any
    other value repeated, such as a second ``"electric"``, changes nothing.

    Parameters
    ----------
    text : str or None
        A phrase, already cleaned by `clean_text`.
    vocabulary : SystemVocabulary
        Parts of the system's phrases.

    Returns
    -------
    ParsedSystems
        What the phrase says. Its kind is ``"unknown"`` if a part is in no
        list of the vocabulary, or it describes more systems than
        `SystemVocabulary.max_systems`.
    """
    phrase = prepare_phrase(text)
    if phrase is None:
        return ParsedSystems("not_a_description")
    marker = vocabulary.fields[0]
    systems: list[dict[str, str]] = [{}]
    assumed = False
    for piece in (piece.strip() for piece in phrase.split(",")):
        if not piece or vocabulary.is_not_a_description(piece):
            continue
        if vocabulary.is_misplaced(piece):
            return ParsedSystems("misplaced")
        part = vocabulary.part_of(piece)
        if part is None:
            return ParsedSystems("unknown")
        values = part.values
        if _starts_new_system(values, systems[-1], marker):
            if part.otherwise is not None and not _starts_new_system(
                part.otherwise, systems[-1], marker
            ):
                values = part.otherwise
            else:
                systems.append({})
        systems[-1].update(values)
        assumed = assumed or part.assumed
    described = tuple(system for system in systems if system)
    if not described:
        return ParsedSystems("not_a_description")
    if len(described) > vocabulary.max_systems:
        return ParsedSystems("unknown")
    return ParsedSystems("described", systems=described, assumed=assumed)


def system_parser(vocabulary: SystemVocabulary) -> ColumnParser[ParsedSystems]:
    """
    Return a parser giving the fields of each system a phrase describes.

    For each field, such as ``"fuel"``, the column ``<prefix>_<field>`` is
    given, as a `polars.Enum` of its labelled values. If the vocabulary
    allows two systems, ``<prefix>_second_<field>`` columns follow, then
    ``<prefix>_system_count``. ``<prefix>_assumed`` comes last.

    Parameters
    ----------
    vocabulary : SystemVocabulary
        Parts of the system's phrases.

    Returns
    -------
    ColumnParser
        A parser for `parse_column`.
    """

    def parse_phrase(text: str | None) -> ParsedSystems:
        return parse_systems(text, vocabulary)

    def output(
        name: str, read_value: Callable[[ParsedSystems], object], dtype: pl.DataType
    ) -> OutputColumn[ParsedSystems]:
        return OutputColumn(f"{vocabulary.prefix}_{name}", read_value, dtype)

    def field_value(number: int, field: str) -> Callable[[ParsedSystems], object]:
        def read_value(parsed: ParsedSystems) -> str | None:
            if number < len(parsed.systems):
                return parsed.systems[number].get(field)
            return None

        return read_value

    fields = tuple(
        output(
            f"{start}{field}",
            field_value(number, field),
            pl.Enum(list(vocabulary.labels[field])),
        )
        for number, start in enumerate(
            _SYSTEM_COLUMN_PREFIXES[: vocabulary.max_systems]
        )
        for field in vocabulary.fields
    )
    count = (
        (output("system_count", lambda parsed: len(parsed.systems), pl.Int64()),)
        if vocabulary.max_systems > 1
        else ()
    )
    return ColumnParser(
        column=vocabulary.column,
        parse_phrase=parse_phrase,
        outputs=(
            *fields,
            *count,
            output("assumed", lambda parsed: parsed.assumed, pl.Boolean()),
        ),
    )


def _starts_new_system(
    values: Mapping[str, str], system: Mapping[str, str], marker: str
) -> bool:
    """
    Return whether `values` belong to a new system rather than `system`.

    They do if they would change a field `system` already has, or name the
    system again, as `marker`, the field that names one.
    """
    return any(
        name in system and (system[name] != value or name == marker)
        for name, value in values.items()
    )
