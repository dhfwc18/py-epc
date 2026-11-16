"""Summarise one prepared year of a dataset, for its manifest."""

from collections.abc import Mapping
from typing import Final, get_args

import polars as pl
from pydantic import BaseModel

from py_epc._dataset._definition import DatasetDefinition
from py_epc._dataset._models import MODEL_CONFIG
from py_epc._dataset._parts import MisalignedParts
from py_epc._dataset._phrases import PhraseKind
from py_epc._dataset._prepare import KIND_COLUMN_SUFFIX

# Name of the count of rows in a query of `count_kinds`.
_ROWS_COLUMN: Final = "rows"

# Field of `PhraseKindCounts` counting rows with no phrase.
_MISSING_FIELD: Final = "missing"


class PhraseKindCounts(BaseModel):
    """Rows of one parsed column, counted by the kind of their phrase."""

    model_config = MODEL_CONFIG

    described: int
    """Rows whose phrase is described, and gives values."""

    not_a_description: int
    """Rows whose phrase describes nothing, such as a placeholder."""

    misplaced: int
    """Rows whose phrase belongs to another column."""

    invalid: int
    """Rows whose phrase is of a known form but an impossible value."""

    unknown: int
    """Rows whose phrase is of no known form, given null."""

    missing: int
    """Rows with no phrase."""


class DatasetYearSummary(BaseModel):
    """What one year of a dataset holds, and what its preparation found."""

    model_config = MODEL_CONFIG

    year: int
    """Lodgement year."""

    rows: int
    """Number of certificates."""

    kinds: dict[str, PhraseKindCounts]
    """Rows of each parsed column, counted by kind, keyed by the column."""

    misaligned_parts: dict[str, MisalignedParts]
    """Certificates whose parts do not line up, keyed by building element."""


def count_kinds(frame: pl.LazyFrame, definition: DatasetDefinition) -> pl.LazyFrame:
    """
    Return a query counting each parsed column's rows by kind.

    Parameters
    ----------
    frame : polars.LazyFrame
        One year of certificates, prepared by `prepare_certificates`.
    definition : DatasetDefinition
        The definition the year was prepared by.

    Returns
    -------
    polars.LazyFrame
        One row giving the number of rows, then a struct of counts for each
        parsed column, named after it, for `summarise_year`.
    """
    return frame.select(
        pl.len().alias(_ROWS_COLUMN),
        *(_kind_counts(parser.column) for parser in definition.parsers),
    )


def summarise_year(
    year: int, counts: pl.DataFrame, misaligned_parts: Mapping[str, MisalignedParts]
) -> DatasetYearSummary:
    """
    Return the summary of one year of a dataset.

    Parameters
    ----------
    year : int
        Lodgement year.
    counts : polars.DataFrame
        The result of the query `count_kinds` returns.
    misaligned_parts : Mapping[str, MisalignedParts]
        Counts of `misaligned_parts` for each building element, keyed by its
        name, if the source has any.

    Returns
    -------
    DatasetYearSummary
        The summary.
    """
    row = counts.row(0, named=True)
    rows = row.pop(_ROWS_COLUMN)
    return DatasetYearSummary(
        year=year,
        rows=rows,
        kinds={
            column: PhraseKindCounts.model_validate(values)
            for column, values in row.items()
        },
        misaligned_parts=dict(misaligned_parts),
    )


def _kind_counts(column: str) -> pl.Expr:
    """Return a struct counting the rows of a parsed column by kind."""
    kind = pl.col(f"{column}{KIND_COLUMN_SUFFIX}")
    return pl.struct(
        *((kind == name).sum().alias(name) for name in get_args(PhraseKind.__value__)),
        kind.is_null().sum().alias(_MISSING_FIELD),
    ).alias(column)
