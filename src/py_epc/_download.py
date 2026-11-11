"""Download a published file, unchanged, with a manifest beside it."""

import contextlib
import hashlib
import os
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Final

import httpx

from py_epc._config import Config
from py_epc._enums import EPCOrigin, EPCSegment, ONSProduct, OSProduct, parse_member
from py_epc._files import atomic_folder, ensure_gitignore
from py_epc._manifest import DownloadManifest, write_manifest
from py_epc._source import Source

# Timeout of the client `download` creates when none is given.
# Up to 30 s to connect, and up to 120 s for data to arrive.
_DEFAULT_TIMEOUT: Final = httpx.Timeout(30.0, read=120.0)

# Size of each chunk written, so a large file never sits in memory.
_CHUNK_BYTES: Final = 1024 * 1024


class DownloadError(Exception):
    """A published file could not be downloaded."""


def download(
    key: str,
    root: str | os.PathLike[str],
    *,
    token: str | None = None,
    client: httpx.Client | None = None,
) -> Path:
    """
    Download a source's published file, unchanged.

    The source's URL is read from `Config`. The file and its manifest are
    written to a new folder, ``<root>/raw/<key parts>/<retrieved at>/``, such
    as ``raw/epc/scotland/domestic/20261110T093000Z/``. The folder appears only
    once both are complete, so a failed download leaves nothing behind.

    Every call downloads the file again. The first call writes
    ``<root>/raw/.gitignore``, which makes Git ignore every download.

    Parameters
    ----------
    key : str
        Config key of the source, such as ``"epc.scotland.domestic"``.
    root : str | os.PathLike[str]
        Folder that holds the relevant data.
    token : str, optional
        Bearer token, for a source with `Source.requires_token` set to true.
        It is sent only to the source's URL, never to a server it redirects
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
        If the setting is not a source.
    ValueError
        If the source requires a token and none is given.
    DownloadError
        If the server refuses the request, the connection fails, or the
        file is not the expected size. Errors never show a redirected URL,
        whose query is a credential.
    """
    source = Config.get(key)
    if not isinstance(source, Source):
        raise TypeError(f"config setting {key!r} is not a source")
    if source.requires_token and not token:
        raise ValueError(f"source {key!r} requires a token; pass token=...")
    headers = {"Authorization": f"Bearer {token}"} if source.requires_token else {}

    retrieved_at = datetime.now(UTC).replace(microsecond=0)
    raw_folder = Path(root).expanduser() / "raw"
    source_folder = raw_folder.joinpath(*key.split("."))
    download_folder = source_folder / retrieved_at.strftime("%Y%m%dT%H%M%SZ")
    with atomic_folder(download_folder) as partial_folder:
        ensure_gitignore(raw_folder)
        with (
            _client_or_new(client) as http,
            _stream(http, source, headers) as response,
        ):
            file_name = source.name + PurePosixPath(response.url.path).suffix
            size_bytes, sha256 = _write_file(
                response, partial_folder / file_name, source
            )
        manifest = DownloadManifest(
            dataset_key=key,
            source_url=source.url,
            publisher=source.publisher,
            retrieved_at=retrieved_at,
            file_name=file_name,
            size_bytes=size_bytes,
            sha256=sha256,
        )
        write_manifest(partial_folder, manifest)
    return download_folder / file_name


def download_epc(
    origin: EPCOrigin | str,
    segment: EPCSegment | str,
    root: str | os.PathLike[str],
    *,
    token: str | None = None,
    client: httpx.Client | None = None,
) -> Path:
    """
    Download the EPC file published for an origin and segment, unchanged.

    This is `download` for the source ``"epc.<origin>.<segment>"``, and
    works the same way.

    Parameters
    ----------
    origin : EPCOrigin | str
        Register to download from, such as ``"scotland"``.
    segment : EPCSegment | str
        Type of certificate, such as ``"domestic"``.
    root : str | os.PathLike[str]
        Folder that holds the relevant data.
    token : str, optional
        Bearer token, needed for England and Wales. It is sent only to the
        register's URL, never to a server it redirects to, and is never
        stored.
    client : httpx.Client, optional
        HTTP client to use, as in `download`.

    Returns
    -------
    Path
        Path of the downloaded file.

    Raises
    ------
    ValueError
        If the origin or segment is unknown, or the register requires a
        token and none is given.
    KeyError
        If the config has no source for the origin and segment, such as
        Scotland's display certificates.
    DownloadError
        If the download fails, as in `download`.

    Examples
    --------
    >>> import py_epc
    >>> py_epc.download_epc("scotland", "domestic", "data")  # doctest: +SKIP
    PosixPath('data/raw/epc/scotland/domestic/20261110T093000Z/...')
    """
    origin = parse_member(EPCOrigin, origin)
    segment = parse_member(EPCSegment, segment)
    key = f"epc.{origin}.{segment}"
    return download(key, root, token=token, client=client)


def download_ons_postcode_directory(
    root: str | os.PathLike[str],
    *,
    client: httpx.Client | None = None,
) -> Path:
    """
    Download the ONS Postcode Directory, unchanged.

    This is `download` for the source ``"ons.postcode-directory"``, and works
    the same way. ONS publishes each quarterly release at a new URL, so the
    config holds one release. To download another, use `Config.set` to
    change ``"ons.postcode-directory.url"``.

    Parameters
    ----------
    root : str | os.PathLike[str]
        Folder that holds the relevant data.
    client : httpx.Client, optional
        HTTP client to use, as in `download`.

    Returns
    -------
    Path
        Path of the downloaded file.

    Raises
    ------
    KeyError
        If the config has no ONS Postcode Directory.
    DownloadError
        If the download fails, as in `download`.

    Examples
    --------
    >>> import py_epc
    >>> py_epc.download_ons_postcode_directory("data")  # doctest: +SKIP
    PosixPath('data/raw/ons/postcode-directory/20261111T093000Z/...')
    """
    key = f"ons.{ONSProduct.POSTCODE_DIRECTORY}"
    return download(key, root, client=client)


def download_os_open_uprn(
    root: str | os.PathLike[str],
    *,
    client: httpx.Client | None = None,
) -> Path:
    """
    Download OS Open UPRN, unchanged.

    This is `download` for the source ``"os.open-uprn"``, and works the same
    way. Its URL always serves the latest release, as a zipped CSV.

    Parameters
    ----------
    root : str | os.PathLike[str]
        Folder that holds the relevant data.
    client : httpx.Client, optional
        HTTP client to use, as in `download`.

    Returns
    -------
    Path
        Path of the downloaded file.

    Raises
    ------
    KeyError
        If the config has no OS Open UPRN.
    DownloadError
        If the download fails, as in `download`.

    Examples
    --------
    >>> import py_epc
    >>> py_epc.download_os_open_uprn("data")  # doctest: +SKIP
    PosixPath('data/raw/os/open-uprn/20261111T093000Z/...')
    """
    key = f"os.{OSProduct.OPEN_UPRN}"
    return download(key, root, client=client)


def _client_or_new(
    client: httpx.Client | None,
) -> contextlib.AbstractContextManager[httpx.Client]:
    """Return the caller's client, left open, or a new one that is closed."""
    if client is not None:
        return contextlib.nullcontext(client)
    return httpx.Client(timeout=_DEFAULT_TIMEOUT)


@contextlib.contextmanager
def _stream(
    http: httpx.Client, source: Source, headers: dict[str, str]
) -> Generator[httpx.Response]:
    """Open the source's URL, following redirects, and check the status."""
    try:
        with http.stream(
            "GET", source.url, headers=headers, follow_redirects=True
        ) as response:
            if response.is_error:
                raise DownloadError(
                    f"{source.url} answered HTTP {response.status_code}"
                )
            yield response
    except httpx.HTTPError as error:
        # The message may hold the redirected URL, so it is dropped.
        raise DownloadError(f"{source.url} failed: {type(error).__name__}") from None


def _write_file(
    response: httpx.Response, path: Path, source: Source
) -> tuple[int, str]:
    """Write the response body to `path`, returning its size and SHA-256."""
    digest = hashlib.sha256()
    size = 0
    with path.open("xb") as file:
        for chunk in response.iter_bytes(_CHUNK_BYTES):
            file.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    if source.expected_bytes is not None and size != source.expected_bytes:
        raise DownloadError(
            f"{source.url} sent {size} bytes; expected {source.expected_bytes}"
        )
    return size, digest.hexdigest()
