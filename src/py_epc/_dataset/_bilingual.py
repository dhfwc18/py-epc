"""
Keep the English of England and Wales values published in English and Welsh.

Notes
-----
Some certificates lodged from 2009 to 2011 give each description in both
languages, alternating English and Welsh parts separated by ``"|"``. Only
England and Wales does this. In Scotland, ``"|"`` separates the parts of a
building instead, which `split_building_parts` handles.
"""

import warnings
from collections.abc import Sequence
from typing import Final

import polars as pl

# Separator between the English and Welsh parts of a bilingual value.
_LANGUAGE_SEPARATOR: Final = "|"

_WELSH_PART_PATTERN: Final = r"\|[^|]*(?:\||$)"

# Most values with an odd number of parts an error or warning lists.
_SHOWN_ODD_VALUES: Final = 20


def english_text(value: pl.Expr) -> pl.Expr:
    """
    Return the English of each value published in English and Welsh.

    The English parts are the first, third and so on, and are joined as
    published. A value without ``"|"`` is returned as it is. A value of no
    English gives null.

    Parameters
    ----------
    value : polars.Expr
        Text as published.

    Returns
    -------
    polars.Expr
        The English text.

    Examples
    --------
    >>> import polars as pl
    >>> value = pl.lit("Room heaters, |Gwresogyddion, |electric|trydan")
    >>> pl.select(english_text(value)).item()
    'Room heaters, electric'
    """
    english = value.str.replace_all(_WELSH_PART_PATTERN, "").str.strip_chars()
    return pl.when(~_is_bilingual(value)).then(value).when(english != "").then(english)


def keep_english_text(
    frame: pl.LazyFrame, columns: Sequence[str], *, strict: bool = True
) -> pl.LazyFrame:
    """
    Return `frame` with each of `columns` in English, by `english_text`.

    The columns are first read once to find any bilingual value with an odd
    number of parts, whose English and Welsh cannot be told apart.

    Parameters
    ----------
    frame : polars.LazyFrame
        Rows of an England and Wales register, as published.
    columns : Sequence[str]
        Text columns that may hold bilingual values.
    strict : bool, default True
        Whether a value with an odd number of parts raises an error. If
        false, it gives null, with a warning listing it.

    Returns
    -------
    polars.LazyFrame
        The rows, in the same order, with the columns in English.

    Raises
    ------
    ValueError
        If `strict` is true and a value has an odd number of parts. The
        message lists them, with their columns.
    """
    counts = (
        frame.select(_has_odd_parts(pl.col(column)).sum() for column in columns)
        .collect(engine="streaming")
        .row(0, named=True)
    )
    odd = {column: count for column, count in counts.items() if count}
    if odd:
        listed = "; ".join(
            f"{column}: {count} such as {_odd_values(frame, column)}"
            for column, count in odd.items()
        )
        message = f"bilingual values with an odd number of parts: {listed}"
        if strict:
            raise ValueError(message)
        warnings.warn(f"{message}, given null", stacklevel=2)
    return frame.with_columns(
        pl.when(~_has_odd_parts(pl.col(column)))
        .then(english_text(pl.col(column)))
        .alias(column)
        for column in columns
    )


def _is_bilingual(value: pl.Expr) -> pl.Expr:
    """Return whether each value has English and Welsh parts."""
    return value.str.contains(_LANGUAGE_SEPARATOR, literal=True)


def _has_odd_parts(value: pl.Expr) -> pl.Expr:
    """Return whether each bilingual value has an odd number of parts."""
    separators = value.str.count_matches(_LANGUAGE_SEPARATOR, literal=True)
    return (_is_bilingual(value) & (separators % 2 == 0)).fill_null(False)


def _odd_values(frame: pl.LazyFrame, column: str) -> list[str]:
    """Return the first values of `column` with an odd number of parts."""
    return (
        frame.select(column)
        .filter(_has_odd_parts(pl.col(column)))
        .unique()
        .sort(column)
        .head(_SHOWN_ODD_VALUES)
        .collect(engine="streaming")[column]
        .to_list()
    )
