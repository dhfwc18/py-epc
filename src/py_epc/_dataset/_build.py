"""Build a dataset from a processed build, one lodgement year at a time."""

import os
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Final, Literal

import polars as pl
from pydantic import AwareDatetime, BaseModel

from py_epc._dataset._definition import DatasetDefinition, load_dataset_definition
from py_epc._dataset._models import MODEL_CONFIG
from py_epc._dataset._parts import misaligned_parts
from py_epc._dataset._prepare import prepare_certificates
from py_epc._dataset._summary import DatasetYearSummary, count_kinds, summarise_year
from py_epc._enums import EPCOrigin, EPCSegment, parse_member
from py_epc._files import atomic_folder, ensure_gitignore
from py_epc._manifest import find_completed, read_manifest, write_manifest
from py_epc._process._build import CERTIFICATES_FOLDER, ProcessManifest

# Folder under the root that holds the datasets, beside "raw" and "processed".
DATASETS_FOLDER: Final = "datasets"


class DatasetManifest(BaseModel):
    """The record of one dataset, written once every year is complete."""

    model_config = MODEL_CONFIG

    version: Literal[1] = 1
    """Version of the manifest format."""

    source_key: str
    """Config key of the source, such as ``"epc.scotland.domestic"``."""

    snapshot: str
    """Name of the processed build, after its raw snapshot, such as
    ``"20261111T093000Z"``."""

    raw_sha256: str
    """SHA-256 of the raw archive, from the build's manifest."""

    definition_version: int
    """Version of the dataset definition's format."""

    package_version: str
    """Version of py-epc that built the dataset."""

    built_at: AwareDatetime
    """When building started, in UTC."""

    years: tuple[DatasetYearSummary, ...]
    """Summary of each lodgement year, in order."""


def build_dataset(
    key: str,
    root: str | os.PathLike[str],
    *,
    snapshot: str | None = None,
    overwrite: bool = False,
    strict: bool = True,
) -> Path:
    """
    Build the dataset of a source from its processed build.

    Each lodgement year of the build is prepared by the source's dataset
    definition, with `prepare_certificates`, and written to
    ``<root>/datasets/<key parts>/<snapshot>/certificates/
    lodgement_year=<year>/data.parquet``. The dataset folder appears only
    once every year is written, with a manifest counting each parsed
    column's rows by kind. The first call writes
    ``<root>/datasets/.gitignore``, which makes Git ignore every dataset.

    Parameters
    ----------
    key : str
        Config key of the source, such as ``"epc.scotland.domestic"``.
    root : str | os.PathLike[str]
        Folder that holds the relevant data, as given to `process`.
    snapshot : str, optional
        Name of the processed build to read. By default, the newest.
    overwrite : bool, default False
        Whether to replace an existing dataset of the build. The old one is
        kept until the new one is complete.
    strict : bool, default True
        Whether a value that cannot be read raises an error, as in
        `prepare_certificates`. If false, it gives null, with a warning.

    Returns
    -------
    Path
        Folder of the dataset.

    Raises
    ------
    KeyError
        If no dataset definition is shipped for the source.
    FileNotFoundError
        If there is no completed processed build, or none with that name.
    FileExistsError
        If the build already has a dataset and `overwrite` is false.
    ValueError
        If `strict` is true and a value cannot be read.
    """
    definition = load_dataset_definition(key)
    root_folder = Path(root).expanduser()
    build = find_completed(root_folder.joinpath("processed", *key.split(".")), snapshot)
    process_manifest = read_manifest(build, ProcessManifest)
    datasets_folder = root_folder / DATASETS_FOLDER
    dataset = datasets_folder.joinpath(*key.split("."), build.name)
    if dataset.exists() and not overwrite:
        raise FileExistsError(f"{dataset} exists; pass overwrite=True to replace it")
    built_at = datetime.now(UTC).replace(microsecond=0)
    with atomic_folder(dataset, replace=overwrite) as partial:
        ensure_gitignore(datasets_folder)
        years = tuple(
            _build_year(definition, year, build, partial, strict=strict)
            for year in sorted(summary.year for summary in process_manifest.years)
        )
        manifest = DatasetManifest(
            source_key=key,
            snapshot=build.name,
            raw_sha256=process_manifest.raw_sha256,
            definition_version=definition.version,
            package_version=version("py-epc"),
            built_at=built_at,
            years=years,
        )
        write_manifest(partial, manifest)
    return dataset


def build_epc_dataset(
    origin: EPCOrigin | str,
    segment: EPCSegment | str,
    root: str | os.PathLike[str],
    *,
    snapshot: str | None = None,
    overwrite: bool = False,
    strict: bool = True,
) -> Path:
    """
    Build the EPC dataset of an origin and segment from its processed build.

    This is `build_dataset` for the source ``"epc.<origin>.<segment>"``, and
    works the same way.

    Parameters
    ----------
    origin : EPCOrigin | str
        Register the build is from, such as ``"scotland"``.
    segment : EPCSegment | str
        Type of certificate, such as ``"domestic"``.
    root : str | os.PathLike[str]
        Folder that holds the relevant data, as given to `process_epc`.
    snapshot : str, optional
        Name of the processed build to read. By default, the newest.
    overwrite : bool, default False
        Whether to replace an existing dataset of the build.
    strict : bool, default True
        Whether a value that cannot be read raises an error.

    Returns
    -------
    Path
        Folder of the dataset.

    Raises
    ------
    ValueError
        If the origin or segment is unknown, or a value cannot be read, as in
        `build_dataset`.
    KeyError
        If no dataset definition is shipped for the origin and segment.
    FileNotFoundError
        If there is no completed processed build, as in `build_dataset`.
    FileExistsError
        If the build already has a dataset, as in `build_dataset`.

    Examples
    --------
    >>> from py_epc import build_epc_dataset
    >>> build_epc_dataset("scotland", "domestic", "data")  # doctest: +SKIP
    PosixPath('data/datasets/epc/scotland/domestic/20261111T093000Z')
    """
    origin = parse_member(EPCOrigin, origin)
    segment = parse_member(EPCSegment, segment)
    key = f"epc.{origin}.{segment}"
    return build_dataset(
        key, root, snapshot=snapshot, overwrite=overwrite, strict=strict
    )


def _build_year(
    definition: DatasetDefinition,
    year: int,
    build: Path,
    dataset: Path,
    *,
    strict: bool,
) -> DatasetYearSummary:
    """Write one prepared year in `dataset`, and summarise it."""
    year_file = Path(CERTIFICATES_FOLDER, f"lodgement_year={year}", "data.parquet")
    certificates = pl.scan_parquet(build / year_file)
    misaligned = {
        element.name: misaligned_parts(certificates, element)
        for element in definition.building_elements
    }
    prepared = prepare_certificates(certificates, definition, strict=strict)
    _, counts = pl.collect_all(
        [
            prepared.sink_parquet(
                dataset / year_file, compression="zstd", mkdir=True, lazy=True
            ),
            count_kinds(prepared, definition),
        ],
        engine="streaming",
    )
    return summarise_year(year, counts, misaligned)
