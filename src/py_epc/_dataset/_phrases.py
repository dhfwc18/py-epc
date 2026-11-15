"""Parse a column of phrases, each distinct phrase once."""

import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal

import polars as pl

from py_epc._dataset._distinct import join_distinct

# What a phrase turned out to be.
type PhraseKind = Literal[
    "described", "not_a_description", "misplaced", "invalid", "unknown"
]

# A prefix some Scottish values start with, which says nothing.
_DESCRIPTION_PREFIX_PATTERN: Final = re.compile(r"^description:\s*", re.IGNORECASE)

# A run of question marks where a separator was lost before publication,
# such as "Cavity wall???as built", read as a comma.
_LOST_SEPARATOR_PATTERN: Final = re.compile(r"\s*\?{3,}\s*")

# Most unknown phrases an error or warning lists.
_SHOWN_UNKNOWN_PHRASES: Final = 20

# Working column, never returned: the output that replaces the column.
_REPLACEMENT_COLUMN: Final = "_replacement"


@dataclass(frozen=True)
class ParsedPhrase:
    """What one phrase turned out to be. Each parser adds its own values."""

    kind: PhraseKind
    """What the phrase turned out to be."""


@dataclass(frozen=True)
class OutputColumn[P: ParsedPhrase]:
    """One column a parser gives, read from each parsed phrase."""

    name: str
    """Name of the column, such as ``"walls_construction"``."""

    read_value: Callable[[P], object]
    """Function reading the column's value from a parsed phrase."""

    dtype: pl.DataType
    """Type of the column, such as ``polars.String()``."""


@dataclass(frozen=True)
class ColumnParser[P: ParsedPhrase]:
    """How to parse one column of phrases, and the columns it gives."""

    column: str
    """Column of phrases, such as ``"walls_description"``."""

    parse_phrase: Callable[[str | None], P]
    """Function parsing one phrase, already cleaned by `clean_text`."""

    outputs: tuple[OutputColumn[P], ...]
    """Columns given, in order."""

    replaces_column: bool = False
    """Whether the first output replaces `column` in place, as for a column
    of codes such as ``"tenure"``. The others are added after the frame's
    columns."""

    def __post_init__(self) -> None:
        names = [output.name for output in self.outputs]
        if not names:
            raise ValueError(f"parser of {self.column!r} has no outputs")
        if len(set(names)) != len(names):
            raise ValueError(f"parser of {self.column!r} repeats an output: {names}")
        if self.replaces_column and names[0] != self.column:
            raise ValueError(
                f"parser replacing {self.column!r} must name its first output so"
            )


def prepare_phrase(text: str | None) -> str | None:
    """
    Return a cleaned phrase ready to parse, or None if it says nothing.

    A leading ``"Description:"`` is removed, and a run of three or more
    question marks, where a separator was lost, becomes a comma. Text that
    is empty, or only question marks, gives None.

    Parameters
    ----------
    text : str or None
        A phrase, already cleaned by `clean_text`.

    Returns
    -------
    str or None
        The phrase, trimmed, or None.

    Examples
    --------
    >>> prepare_phrase("Pitched??????250 mm loft insulation")
    'Pitched, 250 mm loft insulation'
    """
    if text is None:
        return None
    phrase = _DESCRIPTION_PREFIX_PATTERN.sub("", text.strip()).strip()
    if not phrase.strip("?"):
        return None
    return _LOST_SEPARATOR_PATTERN.sub(", ", phrase).strip()


def parse_column[P: ParsedPhrase](
    frame: pl.LazyFrame, parser: ColumnParser[P], *, strict: bool = True
) -> pl.LazyFrame:
    """
    Return `frame` with the columns `parser` gives for each phrase.

    Each distinct phrase of `ColumnParser.column` is parsed once. A
    described phrase gives its values. Any other kind gives null in every
    output, so a placeholder, a misplaced phrase or an invalid value is
    never mistaken for a description. A column that is not text, such as
    Scotland's numbered glazed area, is read as text.

    Parameters
    ----------
    frame : polars.LazyFrame
        Rows whose `ColumnParser.column` is already cleaned by `clean_text`.
    parser : ColumnParser
        How to parse the column.
    strict : bool, default True
        Whether an unknown phrase raises an error. If false, it gives null,
        with a warning listing it.

    Returns
    -------
    polars.LazyFrame
        The rows, in the same order, with the outputs added, or with the
        column replaced by the first if `ColumnParser.replaces_column` is
        true.

    Raises
    ------
    ValueError
        If `strict` is true and a phrase is unknown. The message lists them.
    """
    names = [output.name for output in parser.outputs]
    if parser.replaces_column:
        names[0] = _REPLACEMENT_COLUMN
    unknown: list[str] = []

    def build(values: list[str]) -> pl.DataFrame:
        parsed = [parser.parse_phrase(value) for value in values]
        unknown.extend(
            value
            for value, phrase in zip(values, parsed, strict=True)
            if phrase.kind == "unknown"
        )
        return pl.DataFrame(
            [
                pl.Series(
                    name,
                    [
                        (
                            output.read_value(phrase)
                            if phrase.kind == "described"
                            else None
                        )
                        for phrase in parsed
                    ],
                    output.dtype,
                )
                for name, output in zip(names, parser.outputs, strict=True)
            ]
        )

    frame = frame.with_columns(pl.col(parser.column).cast(pl.String))
    frame = join_distinct(frame, parser.column, build)
    if unknown:
        shown = "; ".join(
            repr(value) for value in sorted(unknown)[:_SHOWN_UNKNOWN_PHRASES]
        )
        message = f"{len(unknown)} unknown phrases in {parser.column!r}: {shown}"
        if strict:
            raise ValueError(message)
        warnings.warn(f"{message}, given null", stacklevel=2)
    if parser.replaces_column:
        frame = frame.with_columns(
            pl.col(_REPLACEMENT_COLUMN).alias(parser.column)
        ).drop(_REPLACEMENT_COLUMN)
    return frame
