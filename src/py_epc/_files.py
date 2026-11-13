"""Write folders safely, so a failed step never leaves a partial result."""

import contextlib
import shutil
from collections.abc import Generator
from pathlib import Path


@contextlib.contextmanager
def atomic_folder(folder: Path, *, replace: bool = False) -> Generator[Path]:
    """
    Yield a hidden partial folder that becomes `folder` only on success.

    The partial folder, ``.<name>.part``, is created beside `folder`, with any
    missing parents. When the block ends without an error, it is renamed to
    `folder`. Otherwise it is deleted, so nothing is left behind.

    Parameters
    ----------
    folder : Path
        Folder to create.
    replace : bool, default False
        Whether to replace `folder` if it exists. The old folder is kept
        until the new one is complete, then deleted.

    Yields
    ------
    Path
        The partial folder, to write into.

    Raises
    ------
    FileExistsError
        If the partial folder already exists.
    OSError
        If `folder` already exists and is not empty, and `replace` is false.
    """
    partial = folder.with_name(f".{folder.name}.part")
    partial.mkdir(parents=True)
    try:
        yield partial
        if replace and folder.exists():
            old = folder.with_name(f".{folder.name}.old")
            shutil.rmtree(old, ignore_errors=True)
            folder.rename(old)
            partial.rename(folder)
            shutil.rmtree(old, ignore_errors=True)
        else:
            partial.rename(folder)
    finally:
        shutil.rmtree(partial, ignore_errors=True)


def ensure_gitignore(folder: Path) -> None:
    """Make Git ignore all of `folder`, unless it has a ``.gitignore``."""
    with (
        contextlib.suppress(FileExistsError),
        (folder / ".gitignore").open("x", encoding="utf-8") as file,
    ):
        file.write("# Created by py-epc.\n*\n")
