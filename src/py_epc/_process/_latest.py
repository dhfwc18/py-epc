"""Mark each property's latest certificate in a build."""

import os
from pathlib import Path
from typing import Final

import polars as pl
from pydantic import BaseModel, ConfigDict

from py_epc._process._register import LatestRule

# Name of the column added to every certificate.
LATEST_COLUMN: Final = "is_latest"

# Most rows sorted or grouped at once. Memory stays flat as data grow.
_ROWS_PER_BUCKET: Final = 2_000_000

# Working columns, never written.
_YEAR_COLUMN: Final = "_year"
_ROW_COLUMN: Final = "_row"
_PROPERTY_KEY_COLUMN: Final = "_property_key"


class LatestSummary(BaseModel):
    """What marking the latest certificates found in a build."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        hide_input_in_errors=True,
        use_attribute_docstrings=True,
    )

    properties: int
    """Number of properties. A certificate without one counts as its own."""

    without_property: int
    """Certificates with no value in any property column."""

    blank_certificate_numbers: int
    """Certificates with no certificate number."""

    duplicate_certificate_numbers: int
    """Certificate numbers given to more than one certificate."""

    superseded_disagreements: int | None
    """Certificates where `is_latest` and the publisher's flag disagree."""


def property_key(rule: LatestRule) -> pl.Expr:
    """
    Return the property each certificate belongs to, as text.

    The first property column with a value is used, prefixed with its name,
    such as ``"uprn:906700000001"``, so values from different columns never
    match. The key is null when every property column is empty.
    """
    return pl.coalesce(
        pl.concat_str(pl.lit(f"{name}:"), pl.col(name).cast(pl.String))
        for name in rule.property_columns
    )


def mark_latest(certificates: Path, rule: LatestRule) -> LatestSummary:
    """
    Add `is_latest` to every certificate in a build's yearly files.

    A property's certificates are sorted by the rule's order columns, oldest
    first, with missing values first; ties are broken by year and row. The
    last one is the latest. A certificate without a property is its own
    property, and so is the latest.

    To find the latest, a few columns of every year are read once per
    bucket of properties, about two million certificates each, so memory
    stays flat however large the build. Each yearly file is then rewritten
    with the flag.

    Parameters
    ----------
    certificates : Path
        Folder of the yearly files, ``lodgement_year=<year>/data.parquet``.
    rule : LatestRule
        How to find each property's latest certificate.

    Returns
    -------
    LatestSummary
        Counts of properties, gaps and disagreements.
    """
    files = {
        int(path.parent.name.removeprefix("lodgement_year=")): path
        for path in certificates.glob("lodgement_year=*/data.parquet")
    }
    marked = _find_latest(files, rule)
    for year, path in files.items():
        _write_flag(path, marked.filter(pl.col(_YEAR_COLUMN) == year))
    return _summarise(files, marked, rule)


def _rows(files: dict[int, Path], rule: LatestRule, *columns: str) -> pl.LazyFrame:
    """Return the property key, year, row and `columns` of each certificate."""
    return pl.concat(
        pl.scan_parquet(path)
        .select(property_key(rule).alias(_PROPERTY_KEY_COLUMN), *columns)
        .with_row_index(_ROW_COLUMN)
        .with_columns(pl.lit(year, pl.Int32).alias(_YEAR_COLUMN))
        for year, path in files.items()
    )


def _find_latest(files: dict[int, Path], rule: LatestRule) -> pl.DataFrame:
    """Return the year, row and `is_latest` of every certificate."""
    superseded = [rule.superseded_column] if rule.superseded_column else []
    rows = _rows(files, rule, *rule.order_columns, *superseded)
    key = pl.col(_PROPERTY_KEY_COLUMN)
    order = [_PROPERTY_KEY_COLUMN, *rule.order_columns, _YEAR_COLUMN, _ROW_COLUMN]
    columns = [_YEAR_COLUMN, _ROW_COLUMN, *superseded]
    buckets = _bucket_count(files)
    parts = [
        rows.filter(key.is_null())
        .select(*columns, pl.lit(True).alias(LATEST_COLUMN))
        .collect(engine="streaming")
    ]
    # Every certificate of a property is in the same bucket, as the bucket
    # comes from the property's key; properties are still told apart by
    # their whole key within a bucket.
    parts.extend(
        rows.filter(key.is_not_null() & _in_bucket(key, bucket, buckets))
        .sort(order, nulls_last=False)
        .select(*columns, key.is_last_distinct().alias(LATEST_COLUMN))
        .collect(engine="streaming")
        for bucket in range(buckets)
    )
    return pl.concat(parts)


def _bucket_count(files: dict[int, Path]) -> int:
    """Return how many buckets keep each one near `_ROWS_PER_BUCKET`."""
    rows = sum(
        pl.scan_parquet(path).select(pl.len()).collect().item()
        for path in files.values()
    )
    return max(1, -(-rows // _ROWS_PER_BUCKET))


def _in_bucket(values: pl.Expr, bucket: int, buckets: int) -> pl.Expr:
    """Return whether each value's hash puts it in `bucket` of `buckets`."""
    return values.hash() % buckets == bucket


def _write_flag(path: Path, marked: pl.DataFrame) -> None:
    """Rewrite one yearly file with `is_latest` as its last column."""
    # Sorted by row, the flags line up with the file, so they are added side
    # by side, without a join that would hold the whole year in memory.
    flags = marked.sort(_ROW_COLUMN).select(LATEST_COLUMN)
    rows = pl.scan_parquet(path).select(pl.len()).collect().item()
    if flags.height != rows:
        raise RuntimeError(f"{path} has {rows} rows but {flags.height} flags")
    temporary = path.with_name(f".{path.name}.part")
    pl.concat([pl.scan_parquet(path), flags.lazy()], how="horizontal").sink_parquet(
        temporary, compression="zstd"
    )
    os.replace(temporary, path)


def _summarise(
    files: dict[int, Path], marked: pl.DataFrame, rule: LatestRule
) -> LatestSummary:
    """Count properties, gaps and disagreements."""
    latest = pl.col(LATEST_COLUMN)
    rows = _rows(files, rule, rule.certificate_column)
    counts = rows.select(
        pl.col(_PROPERTY_KEY_COLUMN).null_count().alias("without"),
        pl.col(rule.certificate_column).null_count().alias("blank"),
    ).collect(engine="streaming")
    number = pl.col(rule.certificate_column)
    buckets = _bucket_count(files)
    duplicates = sum(
        rows.filter(number.is_not_null() & _in_bucket(number, bucket, buckets))
        .group_by(number)
        .len()
        .filter(pl.col("len") > 1)
        .select(pl.len())
        .collect(engine="streaming")
        .item()
        for bucket in range(buckets)
    )
    disagreements = None
    if rule.superseded_column:
        superseded = pl.col(rule.superseded_column)
        disagreements = marked.select(
            ((latest & superseded) | (latest.not_() & superseded.not_())).sum()
        ).item()
    return LatestSummary(
        # Each property has exactly one latest certificate.
        properties=marked.select(latest.sum()).item(),
        without_property=int(counts["without"][0]),
        blank_certificate_numbers=int(counts["blank"][0]),
        duplicate_certificate_numbers=int(duplicates),
        superseded_disagreements=disagreements,
    )
