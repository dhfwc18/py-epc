"""
Clean published text into ASCII, with the pound sign the only exception.

Notes
-----
Accents are removed so that every cleaned value is plain ASCII. Datasets are
often read by tools that handle other characters badly, such as spreadsheets
opening CSV files, older databases and plotting libraries. Plain ASCII also
makes values easier to match, search and compare.

The normalisation of place names is a practical choice rather than a judgement
on the accents themselves. Diacritical marks in Welsh and Scottish Gaelic, such
as the circumflex in the Welsh name for Anglesey or the grave accents found in
many Gaelic place names, are integral to how these languages are written and
reflect the rich and diverse heritage of the places they name. Their omission
here should not be taken to suggest that the original accented spellings are
any less correct or important.

When processing the data register with Py-EPC, the original spellings are
preserved in the processed builds, which retain every value exactly as
published. Entries in the final datasets also retain identifiers such as ONS
area codes, allowing each place to be identified unambiguously. Where the
original spelling matters, such as on maps, in reports or in public-facing
materials, the published spelling can be retrieved accordingly.
"""

import html
import re
import unicodedata
import warnings
from collections.abc import Iterable, Sequence
from typing import Final

import polars as pl

from py_epc._dataset._distinct import Converter, map_distinct

# The one character kept that is not ASCII.
_POUND_SIGN: Final = "\N{POUND SIGN}"

# Characters that only appear when UTF-8 text was decoded as Windows-1252,
# such as "\N{LATIN CAPITAL LETTER A WITH CIRCUMFLEX}\N{SUPERSCRIPT TWO}"
# for a superscript two.
_MOJIBAKE_MARKERS: Final = (
    "\N{LATIN CAPITAL LETTER A WITH CIRCUMFLEX}",
    "\N{LATIN CAPITAL LETTER A WITH TILDE}",
    "\N{LATIN SMALL LETTER A WITH CIRCUMFLEX}\N{EURO SIGN}",
    "\N{LATIN SMALL LETTER A WITH CIRCUMFLEX}\N{SINGLE LOW-9 QUOTATION MARK}",
)

# Most times text is decoded again to repair it, as some was decoded twice.
_REPAIR_ROUNDS: Final = 2

# Characters lost before publication, shown as question marks, where the
# original is clear. An inverted question mark stands for a lost apostrophe,
# such as in a name ending "'s".
_LOST_CHARACTERS: Final = (
    ("W/m??K", "W/m2K"),
    ("W/m?K", "W/m2K"),
    ("\N{LATIN SMALL LETTER A WITH CIRCUMFLEX}??", "'"),
    ("\N{INVERTED QUESTION MARK}", "'"),
)

# Characters with an ASCII form that Unicode normalisation does not give.
_ASCII_FORMS: Final = {
    "\N{LEFT SINGLE QUOTATION MARK}": "'",
    "\N{RIGHT SINGLE QUOTATION MARK}": "'",
    "\N{ACUTE ACCENT}": "'",
    "\N{LEFT DOUBLE QUOTATION MARK}": '"',
    "\N{RIGHT DOUBLE QUOTATION MARK}": '"',
    "\N{HYPHEN}": "-",
    "\N{NON-BREAKING HYPHEN}": "-",
    "\N{EN DASH}": "-",
    "\N{EM DASH}": "-",
    "\N{MINUS SIGN}": "-",
    "\N{HORIZONTAL ELLIPSIS}": "...",
    "\N{MULTIPLICATION SIGN}": "x",
    "\N{FRACTION SLASH}": "/",
    "\N{BULLET}": " ",
    "\N{EURO SIGN}": "EUR",
    "\N{LATIN SMALL LETTER SHARP S}": "ss",
}

# Invisible characters, removed.
_INVISIBLE_CHARACTERS: Final = ("\N{ZERO WIDTH SPACE}",)

_ASCII_TABLE: Final = str.maketrans(
    {**_ASCII_FORMS, **dict.fromkeys(_INVISIBLE_CHARACTERS, "")}
)

# Runs of whitespace, made one space.
_WHITESPACE_PATTERN: Final = re.compile(r"\s+")

# A character `clean_text` keeps as it is.
_CLEAN_CHARACTER_PATTERN: Final = rf"[!-%'->@-~{_POUND_SIGN}]"

# Values `clean_text` returns as they are.
_CLEAN_VALUE_PATTERN: Final = (
    rf"^{_CLEAN_CHARACTER_PATTERN}(?: ?{_CLEAN_CHARACTER_PATTERN})*$"
)


def clean_text(value: str | None) -> str | None:
    """
    Return text in ASCII, with the pound sign the only exception.

    The steps, in order:

    1. HTML entities are decoded, such as ``"&#0178;"``.
    2. Text that was decoded wrongly is repaired, such as an A with a
       circumflex before a superscript two.
    3. Characters lost as ``"??"`` are restored where the original is clear.
    4. Characters become their ASCII form, such as a superscript two
       becoming ``"2"`` and a curly quote becoming ``"'"``, and invisible
       characters, such as a zero-width space, are removed.
    5. Accents are removed.
    6. Runs of whitespace become one space, with none at either end.

    Case is kept. Empty text becomes None.

    A character with no ASCII form is left as it is, so that
    `unknown_characters` can report it rather than lose it.

    Parameters
    ----------
    value : str or None
        Text as published.

    Returns
    -------
    str or None
        The cleaned text, or None if empty.

    Examples
    --------
    >>> clean_text("Average thermal transmittance 0.30 W/m&#0178;K ")
    'Average thermal transmittance 0.30 W/m2K'
    """
    if value is None:
        return None
    text = _repair(html.unescape(value))
    for lost, original in _LOST_CHARACTERS:
        text = text.replace(lost, original)

    text = unicodedata.normalize("NFKC", text.translate(_ASCII_TABLE))
    text = text.translate(_ASCII_TABLE)
    text = "".join(
        character
        for character in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(character)
    )
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()
    return text or None


def unknown_characters(text: str) -> set[str]:
    """Return the characters of `text` that are neither ASCII nor a pound."""
    return {
        character
        for character in text
        if not character.isascii() and character != _POUND_SIGN
    }


def normalise_text_columns(
    frame: pl.LazyFrame, columns: Sequence[str], *, strict: bool = True
) -> pl.LazyFrame:
    """
    Return `frame` with each of `columns` cleaned by `clean_text`.

    Only values that cleaning may change are cleaned, so a column of
    addresses, with millions of distinct values that are nearly all plain
    ASCII already, is cleaned quickly. The distinct values that may change
    are read for every column in one pass, each is cleaned once, then joined
    back to every row with it, and only those values are held in memory.

    Parameters
    ----------
    frame : polars.LazyFrame
        Rows to clean.
    columns : Sequence[str]
        Text columns to clean. Other columns are left as they are.
    strict : bool, default True
        Whether a character with no ASCII form raises an error. If false, it
        is removed, with a warning naming it.

    Returns
    -------
    polars.LazyFrame
        The rows, in the same order, with the columns cleaned.

    Raises
    ------
    ValueError
        If `strict` is true and a value has a character with no ASCII form.
        The message names every such character and its column.
    """
    unknown: dict[str, set[str]] = {}

    def converter(column: str) -> Converter:
        def convert(values: list[str]) -> list[str | None]:
            cleaned, found = _clean_all(values, strict=strict)
            if found:
                unknown[column] = found
            return cleaned

        return convert

    frame = map_distinct(
        frame,
        {column: converter(column) for column in columns},
        is_unchanged=_is_clean,
    )
    if unknown:
        listed = "; ".join(
            f"{column}: {sorted(found)}" for column, found in unknown.items()
        )
        if strict:
            raise ValueError(f"characters with no ASCII form: {listed}")
        warnings.warn(f"removed characters with no ASCII form: {listed}", stacklevel=2)
    return frame


def _is_clean(value: pl.Expr) -> pl.Expr:
    """Return whether `clean_text` cannot change each value, null for null."""
    return value.str.contains(_CLEAN_VALUE_PATTERN)


def _clean_all(
    values: Iterable[str], *, strict: bool
) -> tuple[list[str | None], set[str]]:
    """Clean each value, returning the results and any unknown characters."""
    cleaned: list[str | None] = []
    found: set[str] = set()
    for value in values:
        text = clean_text(value)
        if text is not None:
            unknown = unknown_characters(text)
            found |= unknown
            if unknown and not strict:
                kept = "".join(c for c in text if c not in unknown)
                text = _WHITESPACE_PATTERN.sub(" ", kept).strip() or None
        cleaned.append(text)
    return cleaned, found


def _repair(text: str) -> str:
    """Return `text` decoded again where it was decoded wrongly."""
    for _ in range(_REPAIR_ROUNDS):
        if not any(marker in text for marker in _MOJIBAKE_MARKERS):
            break
        try:
            text = text.encode("cp1252").decode("utf-8")
        except UnicodeError:
            break
    return text
