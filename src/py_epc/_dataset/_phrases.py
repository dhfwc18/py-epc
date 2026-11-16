"""Parse a column of phrases, each distinct phrase once."""

import re
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, get_args

import polars as pl

from py_epc._dataset._distinct import TableBuilder, join_distinct

# What a phrase turned out to be.
type PhraseKind = Literal[
    "described", "not_a_description", "misplaced", "invalid", "unknown"
]

# Type of a column of kinds, listed in the order of `PhraseKind`.
PHRASE_KIND_DTYPE: Final = pl.Enum(list(get_args(PhraseKind.__value__)))

# A prefix some Scottish values start with, which says nothing.
_DESCRIPTION_PREFIX_PATTERN: Final = re.compile(r"^description:\s*", re.IGNORECASE)

# A run of question marks where a separator was lost before publication,
# such as "Cavity wall???as built", read as a comma.
_LOST_SEPARATOR_PATTERN: Final = re.compile(r"\s*\?{3,}\s*")

# Most unknown phrases an error or warning lists.
_SHOWN_UNKNOWN_PHRASES: Final = 20

# Prefix of the working columns, never returned, each holding the output
# that replaces its column.
_REPLACEMENT_PREFIX: Final = "_replacement_"


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


def parse_columns(
    frame: pl.LazyFrame,
    parsers: Sequence[ColumnParser[Any]],
    *,
    strict: bool = True,
    kind_columns: Mapping[str, str] | None = None,
) -> pl.LazyFrame:
    """
    Return `frame` with the columns each parser gives for each phrase.

    The distinct phrases of every `ColumnParser.column` are read in one pass,
    and each is parsed once. A described phrase gives its values. Any other
    kind gives null in every output, so a placeholder, a misplaced phrase or an
    invalid value is never mistaken for a description. A column that is not
    text, such as Scotland's numbered glazed area, is read as text.

    Parameters
    ----------
    frame : polars.LazyFrame
        Rows whose parsed columns are already cleaned by `clean_text`.
    parsers : Sequence[ColumnParser]
        How to parse each column, one parser for each.
    strict : bool, default True
        Whether an unknown phrase raises an error. If false, it gives null,
        with a warning listing it.
    kind_columns : Mapping[str, str], optional
        Name of a column to add after a parsed column's outputs, keyed by
        the parsed column, giving each row's `PhraseKind`, or null if its
        phrase is null. Its type is `PHRASE_KIND_DTYPE`.

    Returns
    -------
    polars.LazyFrame
        The rows, in the same order, with each parser's outputs added, or
        with its column replaced by the first if
        `ColumnParser.replaces_column` is true.

    Raises
    ------
    ValueError
        If two parsers read one column, a kind column is keyed by no parsed
        column or named as an output, or `strict` is true and a phrase is
        unknown.
    """
    kind_columns = kind_columns or {}
    columns = [parser.column for parser in parsers]
    if repeated := sorted({c for c in columns if columns.count(c) > 1}):
        raise ValueError(f"columns read by more than one parser: {repeated}")
    if unparsed := sorted(kind_columns.keys() - set(columns)):
        raise ValueError(f"kind columns keyed by no parsed column: {unparsed}")
    outputs = {output.name for parser in parsers for output in parser.outputs}
    if clashing := sorted(outputs & set(kind_columns.values())):
        raise ValueError(f"kind columns named as outputs: {clashing}")
    unknown: dict[str, list[str]] = {}

    def builder(parser: ColumnParser[Any]) -> TableBuilder:
        names = [output.name for output in parser.outputs]
        if parser.replaces_column:
            names[0] = _replacement_column(parser.column)
        kind_column = kind_columns.get(parser.column)

        def build(values: list[str]) -> pl.DataFrame:
            parsed = [parser.parse_phrase(value) for value in values]
            if found := [
                value
                for value, phrase in zip(values, parsed, strict=True)
                if phrase.kind == "unknown"
            ]:
                unknown[parser.column] = found
            table = [
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
            if kind_column is not None:
                kinds = [phrase.kind for phrase in parsed]
                table.append(pl.Series(kind_column, kinds, PHRASE_KIND_DTYPE))
            return pl.DataFrame(table)

        return build

    frame = frame.with_columns(pl.col(column).cast(pl.String) for column in columns)
    frame = join_distinct(frame, {parser.column: builder(parser) for parser in parsers})
    if unknown:
        message = ". ".join(
            f"{len(found)} unknown phrases in {column!r}: "
            + "; ".join(repr(value) for value in sorted(found)[:_SHOWN_UNKNOWN_PHRASES])
            for column, found in unknown.items()
        )
        if strict:
            raise ValueError(message)
        warnings.warn(f"{message}, given null", stacklevel=2)
    replaced = [parser.column for parser in parsers if parser.replaces_column]
    return frame.with_columns(
        pl.col(_replacement_column(column)).alias(column) for column in replaced
    ).drop(_replacement_column(column) for column in replaced)


def parse_column[P: ParsedPhrase](
    frame: pl.LazyFrame,
    parser: ColumnParser[P],
    *,
    strict: bool = True,
    kind_column: str | None = None,
) -> pl.LazyFrame:
    """
    Return `frame` with the columns `parser` gives for each phrase.

    The same as `parse_columns`, for one column.

    Parameters
    ----------
    frame : polars.LazyFrame
        Rows whose `ColumnParser.column` is already cleaned by `clean_text`.
    parser : ColumnParser
        How to parse the column.
    strict : bool, default True
        Whether an unknown phrase raises an error. If false, it gives null,
        with a warning listing it.
    kind_column : str, optional
        If given, a column of this name is added after the outputs, giving
        each row's `PhraseKind`, or null if its phrase is null. Its type is
        `PHRASE_KIND_DTYPE`.

    Returns
    -------
    polars.LazyFrame
        The rows, in the same order, with the outputs added, or with the
        column replaced by the first if `ColumnParser.replaces_column` is
        true.

    Raises
    ------
    ValueError
        If `kind_column` is named as an output, or `strict` is true and a
        phrase is unknown.
    """
    kind_columns = {} if kind_column is None else {parser.column: kind_column}
    return parse_columns(frame, [parser], strict=strict, kind_columns=kind_columns)


def _replacement_column(column: str) -> str:
    """Return the working column holding the output that replaces `column`."""
    return f"{_REPLACEMENT_PREFIX}{column}"
