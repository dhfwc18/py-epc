"""Write a frame of certificates to one output file."""

import os
from enum import StrEnum
from pathlib import Path

import polars as pl


class OutputType(StrEnum):
    """Type of file a frame of certificates is written to."""

    PARQUET = "parquet"
    """One Parquet file, keeping every column's type."""

    @property
    def suffix(self) -> str:
        """Return the file suffix, such as ``".parquet"``."""
        return f".{self.value}"


def write_output(
    frame: pl.LazyFrame,
    path: str | os.PathLike[str],
    output_type: OutputType,
    *,
    overwrite: bool = False,
) -> Path:
    """
    Write a frame of certificates to one file of `output_type`.

    The frame is streamed, so it need not fit in memory. The file is written
    beside `path` as ``.<name>.part`` and renamed to `path` only once it is
    complete, so a failed write never leaves part of a file.

    Parameters
    ----------
    frame : polars.LazyFrame
        Certificates, such as `scan` returns.
    path : str | os.PathLike[str]
        File to write. Its folder is created if missing.
    output_type : OutputType
        Type of file to write.
    overwrite : bool, default False
        Whether to replace an existing file.

    Returns
    -------
    Path
        The file written.

    Raises
    ------
    FileExistsError
        If the file exists and `overwrite` is false.
    """
    target = Path(path).expanduser()
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists; pass overwrite=True to replace it")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.part")
    try:
        match output_type:
            case OutputType.PARQUET:
                frame.sink_parquet(partial, compression="zstd")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target
