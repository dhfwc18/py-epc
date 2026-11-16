"""Change text columns by converting each of their distinct values once."""

from collections.abc import Callable, Mapping, Sequence
from typing import Final

import polars as pl

# Working columns, never returned.
_RAW_COLUMN: Final = "_raw"
_CONVERTED_COLUMN: Final = "_converted"

# Function given a column's distinct values, returning each one's new value.
type Converter = Callable[[list[str]], Sequence[str | None]]


def map_distinct(
    frame: pl.LazyFrame,
    converters: Mapping[str, Converter],
    *,
    is_unchanged: Callable[[pl.Expr], pl.Expr] | None = None,
) -> pl.LazyFrame:
    """
    Return `frame` with each column converted, one distinct value at a time.

    The distinct values of every column in `converters` are read in one
    pass, and each column's are given to its converter once. Each row is
    then given its value's result by a join. A column of millions of rows
    with thousands of distinct values is converted quickly, and only the
    distinct values are held in memory. Rows keep their order, and the
    columns keep their places. Null stays null.

    Parameters
    ----------
    frame : polars.LazyFrame
        Rows to convert.
    converters : Mapping[str, Converter]
        Function converting each text column, given its distinct values as
        a list, returning a list of the same length with each value's
        result.
    is_unchanged : Callable, optional
        Function given a column, returning whether each value is left as it
        is, so it is never read or given to the converter. By default every
        value is converted.

    Returns
    -------
    polars.LazyFrame
        The rows, with the columns converted.

    Raises
    ------
    ValueError
        If a converter returns a list of a different length.
    """
    order = frame.collect_schema().names()

    def to_convert(column: str) -> pl.Expr:
        value = pl.col(column)
        return (
            value if is_unchanged is None else pl.when(~is_unchanged(value)).then(value)
        )

    values = _distinct_values(
        frame, {column: to_convert(column) for column in converters}
    )
    for column, convert in converters.items():
        converted = convert(values[column])
        if len(converted) != len(values[column]):
            raise ValueError(
                f"converter of {column!r} returned {len(converted)} values"
                f" for {len(values[column])}"
            )
        mapping = pl.LazyFrame(
            {_RAW_COLUMN: values[column], _CONVERTED_COLUMN: converted},
            schema={_RAW_COLUMN: pl.String, _CONVERTED_COLUMN: pl.String},
        )
        frame = (
            frame.join(
                mapping,
                left_on=to_convert(column),
                right_on=_RAW_COLUMN,
                how="left",
                maintain_order="left",
                coalesce=False,
            )
            .with_columns(
                pl.when(pl.col(_RAW_COLUMN).is_not_null())
                .then(pl.col(_CONVERTED_COLUMN))
                .otherwise(pl.col(column))
                .alias(column)
            )
            .drop(_RAW_COLUMN, _CONVERTED_COLUMN)
        )
    return frame.select(order)


# Function given a column's distinct values, returning a frame with one row
# for each, in the same order, holding the columns to add.
type TableBuilder = Callable[[list[str]], pl.DataFrame]


def join_distinct(
    frame: pl.LazyFrame, builders: Mapping[str, TableBuilder]
) -> pl.LazyFrame:
    """
    Return `frame` with columns built from each distinct value of columns.

    Like `map_distinct`, but each builder returns several columns for each
    value, which are added after the frame's columns, in the order of
    `builders`. The columns built from are kept. Rows with null in a column
    get null in every column built from it.

    Parameters
    ----------
    frame : polars.LazyFrame
        Rows to add columns to.
    builders : Mapping[str, TableBuilder]
        Function building columns from each text column, given its distinct
        values as a list, returning a frame with one row for each, in the
        same order.

    Returns
    -------
    polars.LazyFrame
        The rows, in the same order, with the built columns added.

    Raises
    ------
    ValueError
        If a builder returns a frame with a different number of rows, or a
        column the frame or another builder already has.
    """
    values = _distinct_values(frame, {column: pl.col(column) for column in builders})
    names = set(frame.collect_schema().names())
    for column, build in builders.items():
        table = build(values[column])
        if table.height != len(values[column]):
            raise ValueError(
                f"builder of {column!r} returned {table.height} rows"
                f" for {len(values[column])}"
            )
        if existing := sorted(names & set(table.columns)):
            raise ValueError(
                f"builder of {column!r} returned columns the frame has: {existing}"
            )
        names |= set(table.columns)
        mapping = table.with_columns(
            pl.Series(_RAW_COLUMN, values[column], dtype=pl.String)
        )
        frame = frame.join(
            mapping.lazy(),
            left_on=column,
            right_on=_RAW_COLUMN,
            how="left",
            maintain_order="left",
        )
    return frame


def _distinct_values(
    frame: pl.LazyFrame, values: Mapping[str, pl.Expr]
) -> dict[str, list[str]]:
    """
    Return the distinct values of each expression but null, in one pass.

    Each expression's values are read by its own query, and the queries are run
    together, so the work they share, such as earlier joins, is done once.
    """
    queries = [
        frame.select(value.alias(name).unique()) for name, value in values.items()
    ]
    distinct = pl.collect_all(queries, engine="streaming")
    return {
        name: table.to_series().drop_nulls().to_list()
        for name, table in zip(values, distinct, strict=True)
    }
