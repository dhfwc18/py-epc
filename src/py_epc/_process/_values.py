"""Parse raw text values into each column's type, as Polars expressions."""

from typing import Final

import polars as pl

from py_epc._process._register import Column, Register

# A plain decimal number, such as "345.00", ".30", "11." or "-0.5", with an
# optional exponent, such as "1.5e3". Words such as "inf" and "nan" are not
# numbers here, although Polars would cast them.
_NUMBER_PATTERN: Final = r"^-?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][-+]?[0-9]+)?$"


def clean(raw: pl.Expr, register: Register) -> pl.Expr:
    """
    Return `raw` with surrounding spaces removed, and missing values as null.

    Parameters
    ----------
    raw : polars.Expr
        Text values, as read from the file.
    register : Register
        Register whose `Register.missing_values` mean a value is missing.

    Returns
    -------
    polars.Expr
        Text values, or null where a value is missing.
    """
    stripped = raw.str.strip_chars()
    missing = stripped.is_in(list(register.missing_values))
    return pl.when(missing.not_()).then(stripped)


def parse(raw: pl.Expr, column: Column, register: Register) -> pl.Expr:
    """
    Return `raw` cleaned, then parsed into the column's type.

    A value that is present but does not parse becomes null. Use `is_invalid`
    to find such values.

    Parameters
    ----------
    raw : polars.Expr
        Text values, as read from the file.
    column : Column
        Column whose `Column.type` and `Column.formats` say how to parse.
    register : Register
        Register whose missing, true and false values apply.

    Returns
    -------
    polars.Expr
        Values of the column's type: ``String`` for text, ``Int64`` for
        integers, ``Float64`` for numbers, ``Date``, ``Datetime`` or
        ``Boolean``.
    """
    value = clean(raw, register)
    match column.type:
        case "text":
            return value
        case "number":
            return _number(value)
        case "integer":
            return _integer(value)
        case "date":
            return pl.coalesce(
                value.str.to_date(format, strict=False) for format in column.formats
            )
        case "datetime":
            return pl.coalesce(
                value.str.to_datetime(format, strict=False) for format in column.formats
            )
        case "flag":
            return (
                pl.when(value.is_in(list(register.true_values)))
                .then(pl.lit(True))
                .when(value.is_in(list(register.false_values)))
                .then(pl.lit(False))
            )


def is_invalid(raw: pl.Expr, column: Column, register: Register) -> pl.Expr:
    """
    Return whether each value is present but does not parse.

    Missing values are not invalid.

    Parameters
    ----------
    raw : polars.Expr
        Text values, as read from the file.
    column : Column
        Column the values belong to.
    register : Register
        Register whose missing, true and false values apply.

    Returns
    -------
    polars.Expr
        Boolean values, true where a value is invalid.
    """
    return clean(raw, register).is_not_null() & parse(raw, column, register).is_null()


def _number(value: pl.Expr) -> pl.Expr:
    """Return text values as numbers, or null where not a plain number."""
    return pl.when(value.str.contains(_NUMBER_PATTERN)).then(
        value.cast(pl.Float64, strict=False)
    )


def _integer(value: pl.Expr) -> pl.Expr:
    """Return text values as integers, or null where not a whole number."""
    # "100.0" and "11." are whole numbers. "11.5" is not, and is never
    # truncated. Numbers up to 2**53 are exact, which covers every UPRN.
    number = _number(value)
    return pl.when(number == number.floor()).then(number.cast(pl.Int64, strict=False))
