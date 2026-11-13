"""Find a raw snapshot, and hand over its yearly files one at a time."""

import hashlib
import os
import re
import shutil
import zipfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from types import TracebackType
from typing import BinaryIO, Final, Self

import py7zr
from py7zr.io import NullIO, Py7zIO, WriterFactory

from py_epc._manifest import DownloadManifest, find_completed, read_manifest

# Function that takes a member's year and the path of its temporary copy.
# The copy is deleted once the function returns.
type MemberHandler = Callable[[int, Path], None]

# Size of each chunk read when copying or hashing, so no file sits in memory.
_CHUNK_BYTES: Final = 1024 * 1024


def find_snapshot(
    root: str | os.PathLike[str],
    key: str,
    snapshot: str | None = None,
) -> Path:
    """
    Return the folder of a completed download of a source.

    A download is complete when its folder holds a manifest. Partial
    folders, ``.<stamp>.part``, and folders without a manifest are ignored.

    Parameters
    ----------
    root : str | os.PathLike[str]
        Folder that holds the relevant data, as given to `download`.
    key : str
        Config key of the source, such as ``"epc.scotland.domestic"``.
    snapshot : str, optional
        Name of the snapshot folder, such as ``"20261111T093000Z"``. By
        default, the newest completed snapshot is returned.

    Returns
    -------
    Path
        Folder of the snapshot, ``<root>/raw/<key parts>/<stamp>/``.

    Raises
    ------
    FileNotFoundError
        If there is no completed snapshot, or none with that name.
    """
    source_folder = Path(root).expanduser().joinpath("raw", *key.split("."))
    return find_completed(source_folder, snapshot)


def for_each_member(
    snapshot: Path,
    pattern: re.Pattern[str],
    handle: MemberHandler,
    scratch: Path,
) -> None:
    r"""
    Call `handle` for each archive member whose name matches `pattern`.

    The archive is the file named in the snapshot's manifest, a ``.7z`` or a
    ``.zip``. It is first checked against the manifest's SHA-256. Each
    matching member is then copied to a temporary file in `scratch`, given
    to `handle` with its year, and deleted before the next member is read.
    Members are handled in archive order, and a ``.7z`` is read in a single
    pass, so each byte is decompressed once.

    Parameters
    ----------
    snapshot : Path
        Folder of the snapshot, as returned by `find_snapshot`.
    pattern : re.Pattern[str]
        Pattern that a member's whole name must match, with one group
        capturing the year, such as ``re.compile(r"(\d{4})\.csv")``.
        Members that do not match are skipped.
    handle : MemberHandler
        Function given the year and the temporary file of each member.
    scratch : Path
        Existing folder for the temporary files, such as the partial folder
        of the build being written.

    Raises
    ------
    FileNotFoundError
        If the snapshot has no manifest, or its archive is missing.
    ValueError
        If the archive does not match its manifest's SHA-256, or is neither
        a ``.7z`` nor a ``.zip``.
    Exception
        Any error raised by `handle`. The temporary file is still deleted.
    """
    manifest = read_manifest(snapshot, DownloadManifest)
    archive = snapshot / manifest.file_name
    suffix = archive.suffix.lower()
    if suffix not in {".7z", ".zip"}:
        raise ValueError(f"{archive.name} is neither a .7z nor a .zip")
    if _sha256(archive) != manifest.sha256:
        raise ValueError(f"{archive} does not match its manifest's SHA-256")
    if suffix == ".7z":
        _for_each_7z_member(archive, pattern, handle, scratch)
    else:
        _for_each_zip_member(archive, pattern, handle, scratch)


def _sha256(path: Path) -> str:
    """Return the SHA-256 of a file, as lowercase hex."""
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _year(pattern: re.Pattern[str], name: str) -> int | None:
    """Return the year in a member's name, or None if it does not match."""
    match = pattern.fullmatch(name)
    return None if match is None else int(match.group(1))


def _scratch_path(scratch: Path, name: str) -> Path:
    """Return the temporary path for a member, never outside `scratch`."""
    # Only the last part of the name is used, so a member named
    # "../x.csv" cannot write outside the scratch folder.
    return scratch / PurePosixPath(name).name


def _for_each_zip_member(
    archive: Path, pattern: re.Pattern[str], handle: MemberHandler, scratch: Path
) -> None:
    """Hand each matching member of a ``.zip`` to `handle`."""
    with zipfile.ZipFile(archive) as file:
        for info in file.infolist():
            year = _year(pattern, info.filename)
            if year is None or info.is_dir():
                continue
            path = _scratch_path(scratch, info.filename)
            try:
                with file.open(info) as member, path.open("xb") as copy:
                    shutil.copyfileobj(member, copy, _CHUNK_BYTES)
                handle(year, path)
            finally:
                path.unlink(missing_ok=True)


def _for_each_7z_member(
    archive: Path, pattern: re.Pattern[str], handle: MemberHandler, scratch: Path
) -> None:
    """Hand each matching member of a ``.7z`` to `handle`, in one pass."""
    # Given an open file, not a path, py7zr extracts one member at a time.
    with (
        _MemberWriters(pattern, handle, scratch) as writers,
        archive.open("rb") as raw,
        py7zr.SevenZipFile(raw) as file,
    ):
        file.extract(factory=writers)


class _MemberWriters(WriterFactory):
    """Give py7zr a temporary file for each matching member, else nothing."""

    def __init__(
        self, pattern: re.Pattern[str], handle: MemberHandler, scratch: Path
    ) -> None:
        self._pattern = pattern
        self._handle = handle
        self._scratch = scratch
        self._open: list[_MemberWriter] = []

    def create(self, filename: str) -> Py7zIO:
        """Return where py7zr writes the member named `filename`."""
        year = _year(self._pattern, filename)
        if year is None:
            return NullIO()
        writer = _MemberWriter(
            _scratch_path(self._scratch, filename), year, self._handle
        )
        self._open.append(writer)
        return writer

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # py7zr does not close a writer whose member failed to extract.
        # Any temporary file still here is deleted.
        for writer in self._open:
            writer.discard()


class _MemberWriter(Py7zIO):
    """A temporary file that py7zr fills, handed over once it is complete."""

    def __init__(self, path: Path, year: int, handle: MemberHandler) -> None:
        self._path = path
        self._year = year
        self._handle = handle
        self._file: BinaryIO = path.open("x+b")
        self._size = 0

    def write(self, s: bytes | bytearray) -> int:
        written = self._file.write(s)
        self._size += written
        return written

    def read(self, size: int | None = None) -> bytes:
        return self._file.read(-1 if size is None else size)

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._file.seek(offset, whence)

    def flush(self) -> None:
        self._file.flush()

    def size(self) -> int:
        return self._size

    def close(self) -> None:
        """Hand the complete member to `handle`, then delete it."""
        self._file.close()
        try:
            self._handle(self._year, self._path)
        finally:
            self.discard()

    def discard(self) -> None:
        """Close and delete the temporary file, if still there."""
        self._file.close()
        self._path.unlink(missing_ok=True)
