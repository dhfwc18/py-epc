"""Fixtures shared by the processing tests."""

import hashlib
import zipfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import py7zr
import pytest

from py_epc import Config, DownloadManifest, Source
from py_epc._manifest import write_manifest

# Type of the `make_snapshot` fixture.
type MakeSnapshot = Callable[..., Path]


@pytest.fixture
def make_snapshot(tmp_path: Path) -> MakeSnapshot:
    """
    Return a function that writes a tiny raw snapshot, as `download` would.

    The function takes a config key, such as ``"epc.scotland.domestic"``,
    and the archive's members, as a mapping of member name to content. It
    writes the archive and its manifest to
    ``<tmp_path>/raw/<key parts>/<stamp>/`` and returns that folder.

    The archive is a ``.7z``, as Scotland publishes, or a ``.zip``, as England
    and Wales publishes, chosen with ``archive=``. The snapshot's timestamp
    can be set with ``stamp=``, to write several snapshots of one source.
    """

    def make(
        key: str,
        members: Mapping[str, str | bytes],
        *,
        archive: Literal["7z", "zip"] = "7z",
        stamp: str = "20261111T093000Z",
    ) -> Path:
        source = Config.get(key)
        assert isinstance(source, Source)
        folder = tmp_path.joinpath("raw", *key.split("."), stamp)
        folder.mkdir(parents=True)
        path = folder / f"{source.name}.{archive}"
        if archive == "7z":
            with py7zr.SevenZipFile(path, "w") as file:
                for name, content in members.items():
                    file.writestr(content, name)
        else:
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as file:
                for name, content in members.items():
                    file.writestr(name, content)
        data = path.read_bytes()
        write_manifest(
            folder,
            DownloadManifest(
                source_key=key,
                source_url=source.url,
                publisher=source.publisher,
                retrieved_at=datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(
                    tzinfo=UTC
                ),
                file_name=path.name,
                size_bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
            ),
        )
        return folder

    return make
