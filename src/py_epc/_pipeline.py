"""Run every step for an EPC origin and segment in one call."""

import os
from pathlib import Path

import polars as pl

from py_epc._dataset._build import DATASETS_FOLDER, build_dataset
from py_epc._dataset._definition import load_dataset_definition
from py_epc._download import download, download_ons_postcode_directory
from py_epc._enums import EPC_REGISTERS, EPCOrigin, EPCSegment, parse_member
from py_epc._manifest import find_completed
from py_epc._process._build import process
from py_epc._process._postcodes import process_ons_postcode_directory
from py_epc._process._register import load_register
from py_epc._scan import Stage, scan_epc

# Config key of the ONS Postcode Directory, which gives the areas of postcodes.
_POSTCODE_DIRECTORY_KEY = "ons.postcode-directory"


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
) -> pl.LazyFrame:
    """
    Make whatever is missing for an origin and segment, then read it.

    For each register of the origin, both for ``"gb"``, the newest completed
    download is used, or the register is downloaded if it has none or
    `update` is true. Its snapshot is processed if it has no build, and with
    ``stage="dataset"`` built into a dataset if it has none. A step already
    done is never done again. If `geography` is on, the ONS Postcode
    Directory is downloaded and processed too, if it has not been.

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
        Whether to download each register again, even if it has a download,
        and make its build and dataset from the new one.
    token : str, optional
        Bearer token, needed to download England and Wales.
    strict : bool, default True
        Whether a value that cannot be read stops the dataset's build, as in
        `build_dataset`.

    Returns
    -------
    polars.LazyFrame
        The certificates, as `scan_epc` reads them.

    Raises
    ------
    ValueError
        If the origin, segment or stage is unknown, England and Wales must
        be downloaded but no token is given, or a step fails.
    KeyError
        If no register definition, or for ``stage="dataset"`` no dataset
        definition, is shipped for the segment, which is checked before any
        download.

    Examples
    --------
    >>> import py_epc
    >>> homes = py_epc.run_epc_pipeline(  # doctest: +SKIP
    ...     "gb", "domestic", "data", latest=True, token="..."
    ... )
    """
    origin = parse_member(EPCOrigin, origin)
    segment = parse_member(EPCSegment, segment)
    registers = EPC_REGISTERS if origin is EPCOrigin.GB else (origin,)
    keys = [f"epc.{register}.{segment}" for register in registers]
    for key in keys:
        load_register(key)
        if stage == "dataset":
            load_dataset_definition(key)
    snapshots = [
        _prepare_register(
            key, root, stage=stage, update=update, token=token, strict=strict
        )
        for key in keys
    ]
    with_geography = origin is EPCOrigin.GB if geography is None else geography
    if with_geography:
        _prepare_postcode_directory(root)
    return scan_epc(
        origin,
        segment,
        root,
        snapshot=None if origin is EPCOrigin.GB else snapshots[0],
        stage=stage,
        latest=latest,
        geography=with_geography,
    )


def _prepare_register(
    key: str,
    root: str | os.PathLike[str],
    *,
    stage: Stage,
    update: bool,
    token: str | None,
    strict: bool,
) -> str:
    """Make a register's missing steps, returning the name of its snapshot."""
    folder = Path(root).expanduser()
    parts = key.split(".")
    raw = None if update else _completed(folder.joinpath("raw", *parts))
    if raw is None:
        raw = download(key, folder, token=token).parent
    if _completed(folder.joinpath("processed", *parts), raw.name) is None:
        process(key, folder, snapshot=raw.name)
    if stage == "dataset" and (
        _completed(folder.joinpath(DATASETS_FOLDER, *parts), raw.name) is None
    ):
        build_dataset(key, folder, snapshot=raw.name, strict=strict)
    return raw.name


def _prepare_postcode_directory(root: str | os.PathLike[str]) -> None:
    """Download and process the ONS Postcode Directory, if not done yet."""
    folder = Path(root).expanduser()
    parts = _POSTCODE_DIRECTORY_KEY.split(".")
    if _completed(folder.joinpath("processed", *parts)) is not None:
        return
    raw = _completed(folder.joinpath("raw", *parts))
    if raw is None:
        raw = download_ons_postcode_directory(folder).parent
    process_ons_postcode_directory(folder, snapshot=raw.name)


def _completed(folder: Path, name: str | None = None) -> Path | None:
    """Return the newest completed subfolder, or the named one, or None."""
    try:
        return find_completed(folder, name)
    except FileNotFoundError:
        return None
