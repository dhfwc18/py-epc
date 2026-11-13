"""Process the ONS Postcode Directory into the areas each postcode is in."""

import csv
import os
import re
import zipfile
from collections.abc import Iterable
from datetime import UTC, datetime
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Final, Literal, Self

import polars as pl
from pydantic import AwareDatetime, BaseModel, ConfigDict, model_validator

from py_epc._config import Config
from py_epc._files import atomic_folder, ensure_gitignore
from py_epc._manifest import DownloadManifest, read_manifest, write_manifest
from py_epc._process._archive import find_snapshot, for_each_member

# Config key of the ONS Postcode Directory.
_SOURCE_KEY: Final = "ons.postcode-directory"

# Name of the processed file in a build, and of its postcode column.
POSTCODES_FILE: Final = "postcodes.parquet"
POSTCODE_COLUMN: Final = "postcode"

# Folder in the archive holding the lookups of codes to names.
_DOCUMENTS_FOLDER: Final = "Documents"

# A UK postcode without spaces.
_POSTCODE_PATTERN: Final = r"^([A-Z]{1,2}[0-9][A-Z0-9]?[0-9][A-Z]{2}|GIR0AA)$"

# Settings shared by the models, which are read-only and reject unknown keys.
_MODEL_CONFIG = ConfigDict(
    frozen=True,
    extra="forbid",
    strict=True,
    hide_input_in_errors=True,
    use_attribute_docstrings=True,
)


class LookupDocument(BaseModel):
    """Where to find the names of a field's codes, in ``Documents/``."""

    model_config = _MODEL_CONFIG

    file: str
    """Pattern a lookup file's name starts with. Matches are combined."""

    code: str
    """Pattern of the column holding the code to keep."""

    name: str
    """Pattern of the column holding the name."""

    key: str | None = None
    """Pattern of the column holding the directory's codes, if not `code`."""


class PostcodeField(BaseModel):
    """One area or value that can be chosen for each postcode."""

    model_config = _MODEL_CONFIG

    description: str
    """What the field records."""

    column: str
    """Pattern of the directory's column, whose name changes each release."""

    type: Literal["text", "integer", "number", "month"] = "text"
    """Type of the values. A month, published as YYYYMM, is its first day."""

    missing_values: tuple[str, ...] = ()
    """Values that mean a value is missing, in this field only."""

    lookup: LookupDocument | None = None
    """Where to find names for the codes, if any are published."""


class PostcodeDirectory(BaseModel):
    """How to read the ONS Postcode Directory, and the fields to choose."""

    model_config = _MODEL_CONFIG

    version: Literal[1]
    """Version of the definition format."""

    source_key: str
    """Config key of the source, ``"ons.postcode-directory"``."""

    member_pattern: str
    """Pattern of the data file's name in the archive, capturing the year."""

    postcode_column: str
    """Name of the directory's postcode column."""

    excluded_prefixes: tuple[str, ...]
    """Postcode areas left out, such as Northern Ireland's, for the licence."""

    missing_pattern: str
    """Pattern of values that mean missing in every field, as pseudo codes."""

    fields: dict[str, PostcodeField]
    """Fields that can be chosen, keyed by name."""

    @model_validator(mode="after")
    def _check(self) -> Self:
        if re.compile(self.member_pattern).groups != 1:
            raise ValueError("member_pattern must have one group, for the year")
        for name, field in self.fields.items():
            if not re.fullmatch(r"[a-z][a-z0-9]*", name):
                raise ValueError(f"field name {name!r} is not lower case")
            patterns = [field.column]
            if field.lookup is not None:
                patterns += [field.lookup.file, field.lookup.code, field.lookup.name]
            for pattern in patterns:
                re.compile(pattern)
        return self


class FieldSummary(BaseModel):
    """Where one chosen field came from, in one build."""

    model_config = _MODEL_CONFIG

    column: str
    """Name of the directory's column, such as ``"lad26cd"``."""

    lookup_files: tuple[str, ...]
    """Lookup documents read for the field. Empty if none was needed."""

    unmatched: int
    """
    Postcodes with a code that no lookup file has, so the lookup's value
    is null: the name, or the code itself for a field found through another
    code, such as ITL1 through the local administrative unit.
    """


class PostcodeManifest(BaseModel):
    """The record of one processed postcode directory."""

    model_config = _MODEL_CONFIG

    version: Literal[1] = 1
    """Version of the manifest format."""

    source_key: str
    """Config key of the source, ``"ons.postcode-directory"``."""

    snapshot: str
    """Name of the raw snapshot processed, such as ``"20261111T093000Z"``."""

    raw_sha256: str
    """SHA-256 of the raw archive, from its download manifest."""

    package_version: str
    """Version of py-epc that processed the snapshot."""

    processed_at: AwareDatetime
    """When processing started, in UTC."""

    release: str
    """Release of the directory, from its file name, such as ``"AUG_2026"``."""

    rows: int
    """Number of postcodes kept."""

    excluded_rows: int
    """Number of postcodes left out, by their excluded prefixes."""

    names: bool
    """Whether names were added beside the codes."""

    fields: dict[str, FieldSummary]
    """Where each chosen field came from, keyed by field name."""


def normalise_postcode(postcode: pl.Expr) -> pl.Expr:
    """
    Return postcodes in the ONS form, such as ``"AB10 1AA"``, or null.

    Letters are made upper case and every space is removed, then one space
    is put before the last three characters. A value that is not a valid
    UK postcode, such as ``"N/A"`` or a partial postcode, becomes null, so
    it can never match another.

    Parameters
    ----------
    postcode : polars.Expr
        Postcodes as text, in any case and spacing, such as ``"ab101aa"``.

    Returns
    -------
    polars.Expr
        Postcodes in the ONS form, or null where not valid.

    Examples
    --------
    >>> import polars as pl
    >>> postcodes = pl.Series(["ab101aa", " AB10  1AA ", "N/A"])
    >>> pl.select(normalise_postcode(pl.lit(postcodes))).to_series().to_list()
    ['AB10 1AA', 'AB10 1AA', None]
    """
    compact = postcode.str.to_uppercase().str.replace_all(r"\s+", "")
    spaced = pl.concat_str(
        compact.str.slice(0, compact.str.len_chars() - 3),
        pl.lit(" "),
        compact.str.slice(-3),
    )
    return pl.when(compact.str.contains(_POSTCODE_PATTERN)).then(spaced)


def load_postcode_directory() -> PostcodeDirectory:
    """Return the shipped definition of the ONS Postcode Directory."""
    path = files("py_epc._process") / "_registers" / "ons-postcode-directory.json"
    return PostcodeDirectory.model_validate_json(path.read_bytes())


def process_ons_postcode_directory(
    root: str | os.PathLike[str],
    *,
    fields: Iterable[str] | None = None,
    names: bool | None = None,
    snapshot: str | None = None,
    overwrite: bool = False,
) -> Path:
    """
    Process the ONS Postcode Directory into the chosen areas of each postcode.

    Only the postcode and the chosen fields are read. A field with published
    names gives ``<field>_code`` and, if `names` is true, ``<field>_name``,
    read from the matching lookups in the archive's ``Documents/`` folder.
    Without names, a lookup is read only for a field whose code is found
    through another code, such as ITL1. Any other field gives one column,
    ``<field>``. Northern Ireland's postcodes are left out, as their licence
    does not allow free reuse, and pseudo codes such as ``"S99999999"``
    become null.

    The result is written to
    ``<root>/processed/ons/postcode-directory/<snapshot>/postcodes.parquet``,
    sorted by postcode, with a manifest.

    Parameters
    ----------
    root : str | os.PathLike[str]
        Folder that holds the relevant data, as given to `download`.
    fields : Iterable[str], optional
        Fields to keep, such as ``["lad", "pcon", "itl1"]``. By default,
        ``Config.get("geography.fields")``. With none, only postcodes are
        kept.
    names : bool, optional
        Whether to add names beside the codes. By default,
        ``Config.get("geography.names")``.
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
    KeyError
        If a field is not one of the shipped definition's fields.
    FileNotFoundError
        If there is no completed raw snapshot, or it has no postcode file.
    FileExistsError
        If the snapshot is already processed and `overwrite` is false.
    ValueError
        If a field's column or lookup is not found exactly once in the
        release, or a field is given twice.

    Examples
    --------
    >>> import py_epc
    >>> py_epc.process_ons_postcode_directory("data")  # doctest: +SKIP
    PosixPath('data/processed/ons/postcode-directory/20261111T093000Z')
    """
    directory = load_postcode_directory()
    chosen = _chosen_fields(directory, fields)
    if names is None:
        names = bool(Config.get("geography.names"))
    raw = find_snapshot(root, _SOURCE_KEY, snapshot)
    download_manifest = read_manifest(raw, DownloadManifest)
    processed_folder = Path(root).expanduser() / "processed"
    build = processed_folder.joinpath(*_SOURCE_KEY.split("."), raw.name)
    if build.exists() and not overwrite:
        raise FileExistsError(f"{build} exists; pass overwrite=True to replace it")
    processed_at = datetime.now(UTC).replace(microsecond=0)
    archive = raw / download_manifest.file_name
    found: list[PostcodeManifest] = []
    with atomic_folder(build, replace=overwrite) as partial:
        ensure_gitignore(processed_folder)

        def handle(year: int, path: Path) -> None:
            rows, excluded, summaries = _write_postcodes(
                directory, chosen, names, path, archive, partial / POSTCODES_FILE
            )
            found.append(
                PostcodeManifest(
                    source_key=_SOURCE_KEY,
                    snapshot=raw.name,
                    raw_sha256=download_manifest.sha256,
                    package_version=version("py-epc"),
                    processed_at=processed_at,
                    release=_release(path.name),
                    rows=rows,
                    excluded_rows=excluded,
                    names=names,
                    fields=summaries,
                )
            )

        member = re.compile(directory.member_pattern)
        for_each_member(raw, member, handle, scratch=partial)
        if len(found) != 1:
            raise FileNotFoundError(
                f"{archive.name} has {len(found)} files matching"
                f" {directory.member_pattern!r}; expected 1"
            )
        write_manifest(partial, found[0])
    return build


def _chosen_fields(
    directory: PostcodeDirectory, fields: Iterable[str] | None
) -> tuple[str, ...]:
    """Return the chosen fields, from `fields` or the config."""
    if fields is None:
        configured = Config.get("geography.fields")
        if not isinstance(configured, list | tuple):
            raise TypeError("config setting 'geography.fields' is not a list")
        fields = [str(name) for name in configured]
    chosen = tuple(fields)
    unknown = [name for name in chosen if name not in directory.fields]
    if unknown:
        choices = ", ".join(repr(name) for name in directory.fields)
        raise KeyError(f"Unknown fields {unknown}; expected some of {choices}.")
    repeated = sorted({name for name in chosen if chosen.count(name) > 1})
    if repeated:
        raise ValueError(f"fields given more than once: {repeated}")
    return chosen


def _write_postcodes(
    directory: PostcodeDirectory,
    chosen: tuple[str, ...],
    names: bool,
    path: Path,
    archive: Path,
    target: Path,
) -> tuple[int, int, dict[str, FieldSummary]]:
    """Write the chosen fields of each postcode, returning what was found."""
    header = _read_header(path)
    if directory.postcode_column not in header:
        raise ValueError(f"{path.name} has no {directory.postcode_column!r} column")
    columns = {
        name: _one_match(directory.fields[name].column, header, f"{name!r} column")
        for name in chosen
    }
    postcode = normalise_postcode(pl.col(directory.postcode_column))
    excluded = pl.any_horizontal(
        postcode.str.starts_with(prefix) for prefix in directory.excluded_prefixes
    )
    raw = pl.scan_csv(path, infer_schema=False, encoding="utf8-lossy").select(
        directory.postcode_column, *sorted(set(columns.values()))
    )
    excluded_rows = raw.filter(excluded).select(pl.len()).collect().item()
    frame = raw.filter(excluded.not_()).select(
        postcode.alias(POSTCODE_COLUMN),
        *(
            _value(directory, directory.fields[name], columns[name]).alias(name)
            for name in chosen
        ),
    )
    output = [POSTCODE_COLUMN]
    lookup_files: dict[str, tuple[str, ...]] = {}
    # Counts of postcodes whose value no lookup has, keyed by field.
    unmatched: dict[str, pl.Expr] = {}
    for name in chosen:
        lookup = directory.fields[name].lookup
        lookup_files[name] = ()
        if lookup is None:
            output.append(name)
            continue
        if not names and lookup.key is None:
            # The directory's own value is the code, so no lookup is needed.
            frame = frame.with_columns(pl.col(name).alias(f"{name}_code"))
            output.append(f"{name}_code")
            continue
        table, lookup_files[name] = _read_lookup(archive, lookup, name)
        frame = frame.join(table, left_on=name, right_on="_key", how="left")
        code = pl.col("_code") if lookup.key else pl.col(name)
        frame = frame.with_columns(
            code.alias(f"{name}_code"), pl.col("_name").alias(f"{name}_name")
        ).drop("_code", "_name", strict=False)
        found = f"{name}_name" if names else f"{name}_code"
        unmatched[name] = (pl.col(name).is_not_null() & pl.col(found).is_null()).sum()
        output += [f"{name}_code", f"{name}_name"] if names else [f"{name}_code"]
    counts = frame.select(
        pl.len().alias("_rows"),
        *(count.alias(name) for name, count in unmatched.items()),
    ).collect(engine="streaming")
    frame.select(output).sort(POSTCODE_COLUMN).sink_parquet(target, compression="zstd")
    summaries = {
        name: FieldSummary(
            column=columns[name],
            lookup_files=lookup_files[name],
            unmatched=int(counts[name][0]) if name in unmatched else 0,
        )
        for name in chosen
    }
    return int(counts["_rows"][0]), int(excluded_rows), summaries


def _value(directory: PostcodeDirectory, field: PostcodeField, column: str) -> pl.Expr:
    """Return a field's values: trimmed, missing as null, then typed."""
    text = pl.col(column).str.strip_chars()
    missing = text.str.contains(directory.missing_pattern) | text.is_in(
        list(field.missing_values)
    )
    value = pl.when(missing.not_()).then(text)
    match field.type:
        case "text":
            return value
        case "integer":
            return value.cast(pl.Int64, strict=False)
        case "number":
            return value.cast(pl.Float64, strict=False)
        case "month":
            return pl.concat_str(value, pl.lit("01")).str.to_date(
                "%Y%m%d", strict=False
            )


def _read_lookup(
    archive: Path, lookup: LookupDocument, field: str
) -> tuple[pl.LazyFrame, tuple[str, ...]]:
    """Return each code's name from the lookups matching `lookup.file`."""
    parts: list[pl.DataFrame] = []
    used: list[str] = []
    with zipfile.ZipFile(archive) as file:
        for member in sorted(file.namelist()):
            name = PurePosixPath(member).name
            if not (
                PurePosixPath(member).parts[0] == _DOCUMENTS_FOLDER
                and name.endswith(".csv")
                and re.match(lookup.file, name)
            ):
                continue
            table = pl.read_csv(
                file.read(member), infer_schema=False, encoding="utf8-lossy"
            )
            where = f"{field!r} lookup {name!r}"
            code = _one_match(lookup.code, table.columns, f"code column in {where}")
            key = (
                _one_match(lookup.key, table.columns, f"key column in {where}")
                if lookup.key
                else code
            )
            label = _one_match(lookup.name, table.columns, f"name column in {where}")
            parts.append(
                table.select(
                    pl.col(key).str.strip_chars().alias("_key"),
                    pl.col(code).str.strip_chars().alias("_code"),
                    pl.col(label).str.strip_chars().alias("_name"),
                )
            )
            used.append(name)
    if not parts:
        raise ValueError(f"no lookup for {field!r} matches {lookup.file!r}")
    names = pl.concat(parts).filter(pl.col("_key").is_not_null())
    if lookup.key is None:
        names = names.drop("_code")
    return names.unique("_key", keep="first").lazy(), tuple(used)


def _one_match(pattern: str, names: Iterable[str], what: str) -> str:
    """Return the one name that matches `pattern` in full."""
    matches = [name for name in names if re.fullmatch(pattern, name)]
    if len(matches) != 1:
        raise ValueError(f"{what} must match {pattern!r} exactly once; found {matches}")
    return matches[0]


def _read_header(path: Path) -> list[str]:
    """Return the column names in a file's first row."""
    with path.open(encoding="utf-8-sig", newline="") as file:
        return next(csv.reader(file), [])


def _release(file_name: str) -> str:
    """Return the release in a data file's name, such as ``"AUG_2026"``."""
    match = re.search(r"ONSPD_([A-Z]{3}_\d{4})", file_name)
    return match.group(1) if match else PurePosixPath(file_name).stem
