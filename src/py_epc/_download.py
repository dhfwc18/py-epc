"""Download a published file, unchanged, with a manifest beside it."""

import contextlib
import hashlib
import os
import shutil
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Final, Literal

import httpx
from pydantic import AwareDatetime, BaseModel, ConfigDict

from py_epc._config import Config
from py_epc._dataset import Dataset

# Timeout of the client `download` creates when none is given.
# Up to 30 s to connect, and up to 120 s for data to arrive.
_DEFAULT_TIMEOUT: Final = httpx.Timeout(30.0, read=120.0)

# Size of each chunk written, so a large file never sits in memory.
_CHUNK_BYTES: Final = 1024 * 1024

# Name of the manifest written beside each downloaded file.
_MANIFEST_NAME: Final = "manifest.json"


class DownloadError(Exception):
    """A published file could not be downloaded."""


class DownloadManifest(BaseModel):
    """The record of one download, written once the file is complete."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        hide_input_in_errors=True,
        use_attribute_docstrings=True,
    )

    version: Literal[1] = 1
    """Version of the manifest format."""

    dataset_key: str
    """Config key of the dataset, such as ``"epc.scotland.domestic"``."""

    source_url: str
    """The dataset's stable URL. A pre-signed URL is never stored."""

    publisher: str
    """Organisation that publishes the file."""

    retrieved_at: AwareDatetime
    """When the download started, in UTC."""

    file_name: str
    """Name of the downloaded file, in the manifest's folder."""

    size_bytes: int
    """Size of the downloaded file, in bytes."""

    sha256: str
    """SHA-256 checksum of the downloaded file, as lowercase hex."""


def download(
    key: str,
    root: str | os.PathLike[str],
    *,
    token: str | None = None,
    client: httpx.Client | None = None,
) -> Path:
    """
    Download a dataset's published file, unchanged.

    The dataset's URL is read from `Config`. The file and its manifest are
    written to a new folder, ``<root>/raw/<key parts>/<retrieved at>/``, such
    as ``raw/epc/scotland/domestic/20261110T093000Z/``. The folder appears only
    once both are complete, so a failed download leaves nothing behind.

    Every call downloads the file again.

    Parameters
    ----------
    key : str
        Config key of the dataset, such as ``"epc.scotland.domestic"``.
    root : str | os.PathLike[str]
        Folder that holds the relevant data.
    token : str, optional
        Bearer token, for a dataset with `Dataset.requires_token` set to true.
        It is sent only to the dataset's URL, never to a server it redirects
        to, and is never stored.
    client : httpx.Client, optional
        HTTP client to use, for example to set a proxy or a timeout. It is
        left open. By default, a client with a 30 s connect and 120 s read
        timeout is created and closed.

    Returns
    -------
    Path
        Path of the downloaded file.

    Raises
    ------
    KeyError
        If there is no config setting with that key.
    TypeError
        If the setting is not a dataset.
    ValueError
        If the dataset requires a token and none is given.
    DownloadError
        If the server refuses the request, the connection fails, or the
        file is not the expected size. Errors never show a redirected URL,
        whose query is a credential.
    """
    dataset = Config.get(key)
    if not isinstance(dataset, Dataset):
        raise TypeError(f"config setting {key!r} is not a dataset")
    if dataset.requires_token and not token:
        raise ValueError(f"dataset {key!r} requires a token; pass token=...")
    headers = {"Authorization": f"Bearer {token}"} if dataset.requires_token else {}

    retrieved_at = datetime.now(UTC).replace(microsecond=0)
    dataset_folder = Path(root).expanduser().joinpath("raw", *key.split("."))
    download_folder = dataset_folder / retrieved_at.strftime("%Y%m%dT%H%M%SZ")
    partial_folder = dataset_folder / f".{download_folder.name}.part"
    partial_folder.mkdir(parents=True)
    try:
        with (
            _client_or_new(client) as http,
            _stream(http, dataset, headers) as response,
        ):
            file_name = dataset.name + PurePosixPath(response.url.path).suffix
            size_bytes, sha256 = _write_file(
                response, partial_folder / file_name, dataset
            )
        manifest = DownloadManifest(
            dataset_key=key,
            source_url=dataset.url,
            publisher=dataset.publisher,
            retrieved_at=retrieved_at,
            file_name=file_name,
            size_bytes=size_bytes,
            sha256=sha256,
        )
        (partial_folder / _MANIFEST_NAME).write_text(
            manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        partial_folder.rename(download_folder)
    finally:
        shutil.rmtree(partial_folder, ignore_errors=True)
    return download_folder / file_name


def _client_or_new(
    client: httpx.Client | None,
) -> contextlib.AbstractContextManager[httpx.Client]:
    """Return the caller's client, left open, or a new one that is closed."""
    if client is not None:
        return contextlib.nullcontext(client)
    return httpx.Client(timeout=_DEFAULT_TIMEOUT)


@contextlib.contextmanager
def _stream(
    http: httpx.Client, dataset: Dataset, headers: dict[str, str]
) -> Generator[httpx.Response]:
    """Open the dataset's URL, following redirects, and check the status."""
    try:
        with http.stream(
            "GET", dataset.url, headers=headers, follow_redirects=True
        ) as response:
            if response.is_error:
                raise DownloadError(
                    f"{dataset.url} answered HTTP {response.status_code}"
                )
            yield response
    except httpx.HTTPError as error:
        # The message may hold the redirected URL, so it is dropped.
        raise DownloadError(f"{dataset.url} failed: {type(error).__name__}") from None


def _write_file(
    response: httpx.Response, path: Path, dataset: Dataset
) -> tuple[int, str]:
    """Write the response body to `path`, returning its size and SHA-256."""
    digest = hashlib.sha256()
    size = 0
    with path.open("xb") as file:
        for chunk in response.iter_bytes(_CHUNK_BYTES):
            file.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    if dataset.expected_bytes is not None and size != dataset.expected_bytes:
        raise DownloadError(
            f"{dataset.url} sent {size} bytes; expected {dataset.expected_bytes}"
        )
    return size, digest.hexdigest()
