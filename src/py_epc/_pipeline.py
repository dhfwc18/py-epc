"""Run every step for an EPC origin and segment in one call."""

import os
import shutil
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Final, Literal, overload

import polars as pl
from pydantic import AwareDatetime, BaseModel

from py_epc._dataset._build import DATASETS_FOLDER, build_dataset
from py_epc._dataset._definition import load_dataset_definition
from py_epc._dataset._models import MODEL_CONFIG
from py_epc._download import download, download_ons_postcode_directory
from py_epc._enums import EPC_REGISTERS, EPCOrigin, EPCSegment, parse_member
from py_epc._files import ensure_gitignore
from py_epc._manifest import DownloadManifest, find_completed, read_manifest
from py_epc._output import OutputType, write_output
from py_epc._process._build import ProcessManifest, process
from py_epc._process._postcodes import (
    PostcodeManifest,
    process_ons_postcode_directory,
)
from py_epc._process._register import load_register
from py_epc._scan import Stage, scan_epc

# Config key of the ONS Postcode Directory, which gives the areas of postcodes.
_POSTCODE_DIRECTORY_KEY: Final = "ons.postcode-directory"

# Folder under the root that holds the outputs, beside "datasets".
OUTPUTS_FOLDER: Final = "outputs"

# Folders under the root that `clean_up` deletes a register's snapshot from.
_CLEANED_STAGES: Final = ("raw", "processed", DATASETS_FOLDER)

# Suffix of the manifest written beside each output file, after its stem.
_OUTPUT_MANIFEST_SUFFIX: Final = ".manifest.json"


class OutputSource(BaseModel):
    """One downloaded source an output was made from."""

    model_config = MODEL_CONFIG

    source_key: str
    """Config key of the source, such as ``"epc.scotland.domestic"``."""

    snapshot: str
    """Name of its raw snapshot, such as ``"20261111T093000Z"``."""

    raw_sha256: str
    """SHA-256 of its raw file, from its build's manifest."""


class OutputManifest(BaseModel):
    """The record of one output file, written beside it once it is complete."""

    model_config = MODEL_CONFIG

    version: Literal[1] = 1
    """Version of the manifest format."""

    origin: str
    """Origin read, such as ``"gb"``."""

    segment: str
    """Type of certificate, such as ``"domestic"``."""

    stage: str
    """Stage read, ``"processed"`` or ``"dataset"``."""

    latest: bool
    """Whether only each property's latest certificate was kept."""

    geography: bool
    """Whether the areas of each certificate's postcode were added."""

    output_type: str
    """Type of the output file, such as ``"parquet"``."""

    file_name: str
    """Name of the output file, beside this manifest."""

    sources: tuple[OutputSource, ...]
    """Each register's download, then the postcode directory's if
    `geography` is true."""

    package_version: str
    """Version of py-epc that wrote the output."""

    written_at: AwareDatetime
    """When writing started, in UTC."""


@overload
def run_epc_pipeline(
    origin: EPCOrigin | str,
    segment: EPCSegment | str,
    root: str | os.PathLike[str],
    *,
    stage: Stage = ...,
    latest: bool = ...,
    geography: bool | None = ...,
    update: bool = ...,
    token: str | None = ...,
    strict: bool = ...,
    output_type: None = ...,
    output_path: str | os.PathLike[str] | None = ...,
    overwrite: bool = ...,
    clean_up: bool = ...,
) -> pl.LazyFrame: ...


@overload
def run_epc_pipeline(
    origin: EPCOrigin | str,
    segment: EPCSegment | str,
    root: str | os.PathLike[str],
    *,
    stage: Stage = ...,
    latest: bool = ...,
    geography: bool | None = ...,
    update: bool = ...,
    token: str | None = ...,
    strict: bool = ...,
    output_type: OutputType | str,
    output_path: str | os.PathLike[str] | None = ...,
    overwrite: bool = ...,
    clean_up: bool = ...,
) -> Path: ...


def run_epc_pipeline(
    origin: EPCOrigin | str,
    segment: EPCSegment | str,
    root: str | os.PathLike[str],
    *,
    stage: Stage = "dataset",
    latest: bool = False,
    geography: bool | None = None,
    update: bool = False,
    token: str | None = None,
    strict: bool = True,
    output_type: OutputType | str | None = None,
    output_path: str | os.PathLike[str] | None = None,
    overwrite: bool = False,
    clean_up: bool = False,
) -> pl.LazyFrame | Path:
    """
    Make whatever is missing for an origin and segment, then read it.

    For each register of the origin, both for ``"gb"``, the newest completed
    download is used, or the register is downloaded if it has none or
    `update` is true. A new download identical to the newest one is deleted, so
    an unchanged register is not processed or built again. The snapshot is
    processed if it has no build, and with ``stage="dataset"`` built into a
    dataset if it has none. A step already done is never done again. If
    `geography` is on, the ONS Postcode Directory is downloaded and processed
    too, if it has not been.

    Without `output_type`, the certificates are returned as a lazy frame.
    With it, they are written to one file, by default
    ``<root>/outputs/epc-<origin>-<segment>-<stage>.<suffix>``, with
    ``-latest`` before the suffix if `latest` is true, and its path is
    returned. A manifest naming every download the file was made from is
    written beside it, as ``<name>.manifest.json``. The first output writes
    ``<root>/outputs/.gitignore``, which makes Git ignore every output.

    With `clean_up`, the raw download, processed build and dataset of each
    register read are deleted once the output and its manifest are written,
    so only the output is left of them. The postcode directory, other
    snapshots and other sources are kept. The next run downloads the
    registers again.

    Parameters
    ----------
    origin : EPCOrigin | str
        Register to read, such as ``"scotland"``, or ``"gb"`` for both.
    segment : EPCSegment | str
        Type of certificate, such as ``"domestic"``.
    root : str | os.PathLike[str]
        Folder that holds the relevant data, created if missing.
    stage : {"processed", "dataset"}, default "dataset"
        Whether to make and read the processed build, or go on to its
        dataset, as in `scan`.
    latest : bool, default False
        Whether to keep only each property's latest certificate.
    geography : bool, optional
        Whether to add the areas of each certificate's postcode, as in
        `scan_epc`. By default, true for ``"gb"`` and false otherwise.
    update : bool, default False
        Whether to download each register again, even if it has a download.
    token : str, optional
        Bearer token, needed to download England and Wales.
    strict : bool, default True
        Whether a value that cannot be read stops the dataset's build, as in
        `build_dataset`.
    output_type : OutputType | str, optional
        Type of file to write the certificates to, such as ``"parquet"``.
        An SQLite file is much larger, about 1.4 KB a certificate, and
        slower to write.
    output_path : str | os.PathLike[str], optional
        File to write, instead of the default under ``<root>/outputs``.
    overwrite : bool, default False
        Whether to replace an existing output file and its manifest.
    clean_up : bool, default False
        Whether to delete what each register's output was made from, once
        it is written. Needs `output_type`.

    Returns
    -------
    polars.LazyFrame or Path
        The certificates, as `scan_epc` reads them, without `output_type`,
        or the path of the file written with it.

    Raises
    ------
    ValueError
        If the origin, segment, stage or output type is unknown, `clean_up`
        is given without `output_type` or with an `output_path` inside a
        folder it deletes from, England and Wales must be downloaded but no
        token is given, or a step fails.
    KeyError
        If no register definition, or for ``stage="dataset"`` no dataset
        definition, is shipped for the segment, which is checked before any
        download.
    FileExistsError
        If the output file exists and `overwrite` is false, which is checked
        before any download.

    Examples
    --------
    >>> import py_epc
    >>> py_epc.run_epc_pipeline(  # doctest: +SKIP
    ...     "gb", "domestic", "data", latest=True, output_type="parquet"
    ... )
    PosixPath('data/outputs/epc-gb-domestic-dataset-latest.parquet')
    """
    origin = parse_member(EPCOrigin, origin)
    segment = parse_member(EPCSegment, segment)
    output = None if output_type is None else parse_member(OutputType, output_type)
    folder = Path(root).expanduser()
    registers = EPC_REGISTERS if origin is EPCOrigin.GB else (origin,)
    keys = [f"epc.{register}.{segment}" for register in registers]
    for key in keys:
        load_register(key)
        if stage == "dataset":
            load_dataset_definition(key)
    target = None
    if output is not None:
        target = (
            Path(output_path).expanduser()
            if output_path is not None
            else _default_output_path(folder, origin, segment, stage, latest, output)
        )
        if target.exists() and not overwrite:
            raise FileExistsError(f"{target} exists; pass overwrite=True to replace it")
    if clean_up:
        _check_clean_up(folder, target)
    builds = [
        _prepare_register(
            key, folder, stage=stage, update=update, token=token, strict=strict
        )
        for key in keys
    ]
    with_geography = origin is EPCOrigin.GB if geography is None else geography
    postcodes = _prepare_postcode_directory(folder) if with_geography else None
    frame = scan_epc(
        origin,
        segment,
        folder,
        snapshot=None if origin is EPCOrigin.GB else builds[0].name,
        stage=stage,
        latest=latest,
        geography=with_geography,
    )
    if output is None or target is None:
        return frame
    written_at = datetime.now(UTC).replace(microsecond=0)
    if output_path is None:
        (folder / OUTPUTS_FOLDER).mkdir(parents=True, exist_ok=True)
        ensure_gitignore(folder / OUTPUTS_FOLDER)
    write_output(frame, target, output, overwrite=overwrite)
    sources = [_output_source(build, ProcessManifest) for build in builds]
    if postcodes is not None:
        sources.append(_output_source(postcodes, PostcodeManifest))
    manifest = OutputManifest(
        origin=origin.value,
        segment=segment.value,
        stage=stage,
        latest=latest,
        geography=with_geography,
        output_type=output.value,
        file_name=target.name,
        sources=tuple(sources),
        package_version=version("py-epc-uk"),
        written_at=written_at,
    )
    output_manifest_path(target).write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    if clean_up:
        for key, build in zip(keys, builds, strict=True):
            _remove_snapshot(folder, key, build.name)
    return target


def output_manifest_path(path: str | os.PathLike[str]) -> Path:
    """
    Return the path of the manifest written beside an output file.

    Parameters
    ----------
    path : str | os.PathLike[str]
        The output file, such as ``"epc-gb-domestic-dataset.parquet"``.

    Returns
    -------
    Path
        The manifest, such as ``"epc-gb-domestic-dataset.manifest.json"``.
    """
    output = Path(path)
    return output.with_name(f"{output.stem}{_OUTPUT_MANIFEST_SUFFIX}")


def _default_output_path(
    root: Path,
    origin: EPCOrigin,
    segment: EPCSegment,
    stage: Stage,
    latest: bool,
    output_type: OutputType,
) -> Path:
    """Return the default file of an output under ``<root>/outputs``."""
    name = f"epc-{origin}-{segment}-{stage}{'-latest' if latest else ''}"
    return root / OUTPUTS_FOLDER / f"{name}{output_type.suffix}"


def _prepare_register(
    key: str,
    root: Path,
    *,
    stage: Stage,
    update: bool,
    token: str | None,
    strict: bool,
) -> Path:
    """Make a register's missing steps, returning its processed build."""
    parts = key.split(".")
    previous = _completed(root.joinpath("raw", *parts))
    if previous is not None and not update:
        raw = previous
    else:
        raw = download(key, root, token=token).parent
        if previous is not None and _same_download(previous, raw):
            shutil.rmtree(raw)
            raw = previous
    if _completed(root.joinpath("processed", *parts), raw.name) is None:
        process(key, root, snapshot=raw.name)
    if stage == "dataset" and (
        _completed(root.joinpath(DATASETS_FOLDER, *parts), raw.name) is None
    ):
        build_dataset(key, root, snapshot=raw.name, strict=strict)
    return root.joinpath("processed", *parts, raw.name)


def _prepare_postcode_directory(root: Path) -> Path:
    """Make the processed postcode directory if missing, and return it."""
    parts = _POSTCODE_DIRECTORY_KEY.split(".")
    processed = _completed(root.joinpath("processed", *parts))
    if processed is not None:
        return processed
    raw = _completed(root.joinpath("raw", *parts))
    if raw is None:
        raw = download_ons_postcode_directory(root).parent
    return process_ons_postcode_directory(root, snapshot=raw.name)


def _same_download(kept: Path, new: Path) -> bool:
    """Return whether two raw snapshots downloaded the same file."""
    return (
        read_manifest(kept, DownloadManifest).sha256
        == read_manifest(new, DownloadManifest).sha256
    )


def _output_source(
    build: Path, model: type[ProcessManifest] | type[PostcodeManifest]
) -> OutputSource:
    """Return the record of the download a processed build was made from."""
    manifest = read_manifest(build, model)
    return OutputSource(
        source_key=manifest.source_key,
        snapshot=manifest.snapshot,
        raw_sha256=manifest.raw_sha256,
    )


def _check_clean_up(root: Path, target: Path | None) -> None:
    """Raise ValueError unless `clean_up` can keep the output."""
    if target is None:
        raise ValueError("clean_up needs output_type, or nothing would be left")
    for stage in _CLEANED_STAGES:
        stage_folder = (root / stage).resolve()
        if target.resolve().is_relative_to(stage_folder):
            raise ValueError(
                f"output {target} is inside {stage_folder}, which clean_up deletes from"
            )


def _remove_snapshot(root: Path, key: str, snapshot: str) -> None:
    """Delete a source's raw snapshot, build and dataset, where they exist."""
    for stage in _CLEANED_STAGES:
        folder = root.joinpath(stage, *key.split("."), snapshot)
        if folder.exists():
            shutil.rmtree(folder)


def _completed(folder: Path, name: str | None = None) -> Path | None:
    """Return the newest completed subfolder, or the named one, or None."""
    try:
        return find_completed(folder, name)
    except FileNotFoundError:
        return None
