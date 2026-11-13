"""Change a text column by converting each of its distinct values once."""

from collections.abc import Callable, Sequence
from typing import Final

import polars as pl

# Working column, never returned: the value before it is converted.
_RAW_COLUMN: Final = "_raw"

# Function given a column's distinct values, returning each one's new value.
type Converter = Callable[[list[str]], Sequence[str | None]]


def map_distinct(frame: pl.LazyFrame, column: str, convert: Converter) -> pl.LazyFrame:
    """
    Return `frame` with `column` converted, one distinct value at a time.

    The column's distinct values are collected and given to `convert` once,
    then each row is given its value's result by a join. A column of
    millions of rows with thousands of distinct values is converted quickly,
    and only the distinct values are held in memory. Rows keep their order,
    and the column keeps its place. Null stays null.

    Parameters
    ----------
    frame : polars.LazyFrame
        Rows to convert.
    column : str
        Text column to convert.
    convert : Converter
        Function given the distinct values, as a list, returning a list of
        the same length with each value's result.

    Returns
    -------
    polars.LazyFrame
        The rows, with the column converted.

    Raises
    ------
    ValueError
        If `convert` returns a list of a different length.
    """
    order = frame.collect_schema().names()
    values = (
        frame.select(pl.col(column).unique())
        .collect(engine="streaming")
        .to_series()
        .drop_nulls()
        .to_list()
    )
    converted = convert(values)
    if len(converted) != len(values):
        raise ValueError(
            f"converter returned {len(converted)} values for {len(values)}"
        )
    mapping = pl.LazyFrame(
        {_RAW_COLUMN: values, column: converted},
        schema={_RAW_COLUMN: pl.String, column: pl.String},
    )
    return (
        frame.rename({column: _RAW_COLUMN})
        .join(mapping, on=_RAW_COLUMN, how="left", maintain_order="left")
        .drop(_RAW_COLUMN)
        .select(order)
    )


# Function given a column's distinct values, returning a frame with one row
# for each, in the same order, holding the columns to add.
type TableBuilder = Callable[[list[str]], pl.DataFrame]


def join_distinct(
    frame: pl.LazyFrame, column: str, build: TableBuilder
) -> pl.LazyFrame:
    """
    Return `frame` with columns built from each distinct value of `column`.

    Like `map_distinct`, but `build` returns several columns for each value,
    which are added after the frame's columns. `column` itself is kept. Rows
    with null in `column` get null in every added column.

    Parameters
    ----------
    frame : polars.LazyFrame
        Rows to add columns to.
    column : str
        Text column whose values the new columns are built from.
    build : TableBuilder
        Function given the distinct values, as a list, returning a frame
        with one row for each, in the same order.

    Returns
    -------
    polars.LazyFrame
        The rows, in the same order, with the built columns added.

    Raises
    ------
    ValueError
        If `build` returns a frame with a different number of rows, or a
        column `frame` already has.
    """
    values = (
        frame.select(pl.col(column).unique())
        .collect(engine="streaming")
        .to_series()
        .drop_nulls()
        .to_list()
    )
    table = build(values)
    if table.height != len(values):
        raise ValueError(f"builder returned {table.height} rows for {len(values)}")
    existing = sorted(set(table.columns) & set(frame.collect_schema().names()))
    if existing:
        raise ValueError(f"builder returned columns the frame has: {existing}")
    mapping = table.with_columns(pl.Series(_RAW_COLUMN, values, dtype=pl.String))
    return frame.join(
        mapping.lazy(),
        left_on=column,
        right_on=_RAW_COLUMN,
        how="left",
        maintain_order="left",
    )
