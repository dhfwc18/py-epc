"""Process a raw snapshot into typed Parquet, one file per lodgement year."""

import csv
import os
import warnings
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Final, Literal

import polars as pl
from pydantic import AwareDatetime, BaseModel, ConfigDict

from py_epc._enums import EPCOrigin, EPCSegment, parse_member
from py_epc._files import atomic_folder, ensure_gitignore
from py_epc._manifest import DownloadManifest, read_manifest, write_manifest
from py_epc._process._archive import find_snapshot, for_each_member
from py_epc._process._latest import LatestSummary, mark_latest
from py_epc._process._register import Column, Register, load_register
from py_epc._process._values import is_invalid, parse

# Most example values kept for each column with invalid values.
_EXAMPLES: Final = 5

# Folder in a build that holds the certificates, one folder per year.
_CERTIFICATES: Final = "certificates"

# Settings shared by the manifest models, which are read-only.
_MODEL_CONFIG = ConfigDict(
    frozen=True,
    extra="forbid",
    strict=True,
    hide_input_in_errors=True,
    use_attribute_docstrings=True,
)


class ProcessWarning(UserWarning):
    """Processing finished, but some values were invalid or out of place."""


class InvalidValues(BaseModel):
    """Values of one column that were present but did not parse."""

    model_config = _MODEL_CONFIG

    count: int
    """Number of invalid values, which became null."""

    examples: tuple[str, ...]
    """Up to five distinct invalid values, as published."""


class YearSummary(BaseModel):
    """What processing found in one yearly file."""

    model_config = _MODEL_CONFIG

    year: int
    """Year of the file, from its name in the archive."""

    rows: int
    """Number of certificates in the file."""

    invalid: dict[str, InvalidValues]
    """Invalid values, keyed by processed column name. Empty if none."""

    year_mismatches: int
    """Rows whose year column is in a different year from the file."""


class ProcessManifest(BaseModel):
    """The record of one build, written once every year is complete."""

    model_config = _MODEL_CONFIG

    version: Literal[1] = 1
    """Version of the manifest format."""

    source_key: str
    """Config key of the source, such as ``"epc.scotland.domestic"``."""

    snapshot: str
    """Name of the raw snapshot processed, such as ``"20261111T093000Z"``."""

    raw_sha256: str
    """SHA-256 of the raw archive, from its download manifest."""

    register_version: int
    """Version of the register definition's format."""

    package_version: str
    """Version of py-epc that processed the snapshot."""

    processed_at: AwareDatetime
    """When processing started, in UTC."""

    years: tuple[YearSummary, ...]
    """Summary of each yearly file, in archive order."""

    latest: LatestSummary | None
    """What marking the latest certificates found, if the register says how."""


def process(
    key: str,
    root: str | os.PathLike[str],
    *,
    snapshot: str | None = None,
    overwrite: bool = False,
) -> Path:
    """
    Process a raw snapshot of a source into typed Parquet.

    Each yearly file in the snapshot's archive is checked against the
    source's register definition, parsed into typed columns with their
    processed names, and written to
    ``<root>/processed/<key parts>/<snapshot>/certificates/
    lodgement_year=<year>/data.parquet``. The build folder appears only once
    every year is written, with a manifest of what was found. The first call
    writes ``<root>/processed/.gitignore``, which makes Git ignore every
    build.

    If the register says how to find each property's latest certificate,
    every certificate gets an ``is_latest`` column, as its last column.

    Values that do not parse become null. They are counted in the manifest,
    and a `ProcessWarning` names the years and columns that have them. A
    certificate number given to more than one certificate is also warned.

    Parameters
    ----------
    key : str
        Config key of the source, such as ``"epc.scotland.domestic"``.
    root : str | os.PathLike[str]
        Folder that holds the relevant data, as given to `download`.
    snapshot : str, optional
        Name of the raw snapshot to process. By default, the newest.
    overwrite : bool, default False
        Whether to replace an existing build of the snapshot. The old build
        is kept until the new one is complete.

    Returns
    -------
    Path
        Folder of the build.

    Raises
    ------
    KeyError
        If no register definition is shipped for the source.
    FileNotFoundError
        If there is no completed raw snapshot, or none with that name.
    FileExistsError
        If the snapshot is already processed and `overwrite` is false.
    ValueError
        If the archive does not match its manifest, or a file's header does
        not match the register definition.
    """
    register = load_register(key)
    raw = find_snapshot(root, key, snapshot)
    download_manifest = read_manifest(raw, DownloadManifest)
    processed_folder = Path(root).expanduser() / "processed"
    build = processed_folder.joinpath(*key.split("."), raw.name)
    if build.exists() and not overwrite:
        raise FileExistsError(f"{build} exists; pass overwrite=True to replace it")
    processed_at = datetime.now(UTC).replace(microsecond=0)
    years: list[YearSummary] = []
    with atomic_folder(build, replace=overwrite) as partial:
        ensure_gitignore(processed_folder)

        def handle(year: int, path: Path) -> None:
            years.append(_process_year(register, year, path, partial))

        for_each_member(raw, register.member_regex, handle, scratch=partial)
        latest = (
            mark_latest(partial / _CERTIFICATES, register.latest)
            if register.latest is not None
            else None
        )
        manifest = ProcessManifest(
            source_key=key,
            snapshot=raw.name,
            raw_sha256=download_manifest.sha256,
            register_version=register.version,
            package_version=version("py-epc"),
            processed_at=processed_at,
            years=tuple(years),
            latest=latest,
        )
        write_manifest(partial, manifest)
    _warn(key, years, latest)
    return build


def process_epc(
    origin: EPCOrigin | str,
    segment: EPCSegment | str,
    root: str | os.PathLike[str],
    *,
    snapshot: str | None = None,
    overwrite: bool = False,
) -> Path:
    """
    Process a raw EPC snapshot for an origin and segment into typed Parquet.

    This is `process` for the source ``"epc.<origin>.<segment>"``, and works
    the same way.

    Parameters
    ----------
    origin : EPCOrigin | str
        Register the snapshot is from, such as ``"scotland"``.
    segment : EPCSegment | str
        Type of certificate, such as ``"domestic"``.
    root : str | os.PathLike[str]
        Folder that holds the relevant data, as given to `download_epc`.
    snapshot : str, optional
        Name of the raw snapshot to process. By default, the newest.
    overwrite : bool, default False
        Whether to replace an existing build of the snapshot.

    Returns
    -------
    Path
        Folder of the build.

    Raises
    ------
    ValueError
        If the origin or segment is unknown, or processing fails, as in
        `process`.
    KeyError
        If no register definition is shipped for the origin and segment.
    FileNotFoundError
        If there is no completed raw snapshot, as in `process`.
    FileExistsError
        If the snapshot is already processed, as in `process`.

    Examples
    --------
    >>> import py_epc
    >>> py_epc.process_epc("scotland", "domestic", "data")  # doctest: +SKIP
    PosixPath('data/processed/epc/scotland/domestic/20261111T093000Z')
    """
    origin = parse_member(EPCOrigin, origin)
    segment = parse_member(EPCSegment, segment)
    key = f"epc.{origin}.{segment}"
    return process(key, root, snapshot=snapshot, overwrite=overwrite)


def _process_year(
    register: Register, year: int, path: Path, build: Path
) -> YearSummary:
    """Write one yearly file as Parquet in `build`, and summarise it."""
    columns = register.match_header(_read_header(path))
    raw = pl.scan_csv(
        path,
        has_header=False,
        skip_rows=1 + register.label_rows,
        schema=dict.fromkeys(columns, pl.String),
        encoding="utf8-lossy",
    )
    target = build / _CERTIFICATES / f"lodgement_year={year}" / "data.parquet"
    raw.select(
        parse(pl.col(source), column, register).alias(column.name)
        for source, column in columns.items()
    ).sink_parquet(target, compression="zstd", mkdir=True)
    return _summarise(register, year, raw, columns)


def _read_header(path: Path) -> list[str]:
    """Return the column names in a file's first row."""
    # utf-8-sig drops the byte order mark, so the first name is clean.
    with path.open(encoding="utf-8-sig", newline="") as file:
        return next(csv.reader(file), [])


def _summarise(
    register: Register, year: int, raw: pl.LazyFrame, columns: dict[str, Column]
) -> YearSummary:
    """Count rows, invalid values and year mismatches in one yearly file."""
    checked = {
        source: column for source, column in columns.items() if column.type != "text"
    }
    year_source = next(
        (s for s, c in columns.items() if c.name == register.year_column), None
    )
    counts = raw.select(
        pl.len().alias("rows"),
        *(
            is_invalid(pl.col(source), column, register).sum().alias(source)
            for source, column in checked.items()
        ),
        (
            (
                parse(pl.col(year_source), columns[year_source], register).dt.year()
                != year
            ).sum()
            if year_source is not None
            else pl.lit(0)
        ).alias("year_mismatches"),
    ).collect(engine="streaming")
    invalid: dict[str, InvalidValues] = {}
    for source, column in checked.items():
        count = int(counts[source][0])
        if count:
            examples = (
                raw.filter(is_invalid(pl.col(source), column, register))
                .select(pl.col(source).str.strip_chars())
                .unique(maintain_order=True)
                .head(_EXAMPLES)
                .collect(engine="streaming")
                .to_series()
                .to_list()
            )
            invalid[column.name] = InvalidValues(count=count, examples=tuple(examples))
    return YearSummary(
        year=year,
        rows=int(counts["rows"][0]),
        invalid=invalid,
        year_mismatches=int(counts["year_mismatches"][0]),
    )


def _warn(key: str, years: list[YearSummary], latest: LatestSummary | None) -> None:
    """Warn about invalid values, year mismatches and repeated numbers."""
    problems: list[str] = []
    for summary in sorted(years, key=lambda summary: summary.year):
        found = [f"{name} ({values.count})" for name, values in summary.invalid.items()]
        if summary.year_mismatches:
            found.append(f"year mismatches ({summary.year_mismatches})")
        if found:
            problems.append(f"{summary.year}: {', '.join(found)}")
    if problems:
        warnings.warn(
            f"{key} has values that did not parse or are in the wrong year;"
            f" see the build manifest. {'; '.join(problems)}",
            ProcessWarning,
            stacklevel=3,
        )
    if latest is not None and latest.duplicate_certificate_numbers:
        warnings.warn(
            f"{key} gives {latest.duplicate_certificate_numbers} certificate"
            " numbers to more than one certificate; see the build manifest.",
            ProcessWarning,
            stacklevel=3,
        )
