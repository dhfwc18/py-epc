"""Tests for downloading a published file."""

import hashlib
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest

from py_epc import (
    Config,
    DownloadError,
    DownloadManifest,
    EPCOrigin,
    EPCSegment,
    download,
    download_epc,
)

# Key of the dataset every test downloads.
_DATASET_KEY = "epc.scotland.domestic"

# Stable URL of the dataset, as the config holds it.
_STABLE_URL = "https://publisher.invalid/resource/download"

# Pre-signed URL the publisher redirects to. Its query is a credential.
_SIGNED_URL = "https://bucket.invalid/archive.7z?X-Amz-Signature=secret-signature"

# Bytes of the published file, including bytes that are not text.
_PUBLISHED_BYTES = b"published file\x00\xff\n"


class FakePublisher:
    """A publisher that redirects to a pre-signed URL, as both registers do."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.signed_status = 200
        self.signed_error: httpx.HTTPError | None = None

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url == _STABLE_URL:
            return httpx.Response(302, headers={"Location": _SIGNED_URL})
        if self.signed_error is not None:
            raise self.signed_error
        return httpx.Response(self.signed_status, content=_PUBLISHED_BYTES)


@pytest.fixture
def publisher() -> FakePublisher:
    """Return the fake publisher, for tests to change or inspect."""
    Config.set(f"{_DATASET_KEY}.url", _STABLE_URL)
    return FakePublisher()


@pytest.fixture
def client(publisher: FakePublisher) -> Iterator[httpx.Client]:
    """Return a client whose requests go to the fake publisher only."""
    with httpx.Client(transport=httpx.MockTransport(publisher.handle)) as http:
        yield http


def saved_files(root: Path) -> list[Path]:
    """Return every file under `root`."""
    return [path for path in root.rglob("*") if path.is_file()]


class TestDownload:
    """`download` saves a published file unchanged, with a manifest."""

    def test_served_file_is_saved_unchanged(
        self, tmp_path: Path, client: httpx.Client
    ) -> None:
        path = download(_DATASET_KEY, tmp_path, client=client)
        assert path.read_bytes() == _PUBLISHED_BYTES

    def test_served_file_is_named_after_dataset_and_published_suffix(
        self, tmp_path: Path, client: httpx.Client
    ) -> None:
        path = download(_DATASET_KEY, tmp_path, client=client)
        assert path.name == "epc-scotland-domestic.7z"

    def test_served_file_is_saved_in_a_folder_per_download_under_key(
        self, tmp_path: Path, client: httpx.Client
    ) -> None:
        path = download(_DATASET_KEY, tmp_path, client=client)
        assert path.parent.parent == tmp_path / "raw" / "epc" / "scotland" / "domestic"
        assert path.parent.name.endswith("Z")

    def test_root_starting_with_tilde_is_expanded_to_home(
        self, tmp_path: Path, client: httpx.Client, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        path = download(_DATASET_KEY, "~/data", client=client)
        assert path.is_relative_to(tmp_path / "data")

    def test_manifest_records_size_and_sha256(
        self, tmp_path: Path, client: httpx.Client
    ) -> None:
        path = download(_DATASET_KEY, tmp_path, client=client)
        manifest = DownloadManifest.model_validate_json(
            (path.parent / "manifest.json").read_bytes()
        )
        assert manifest.file_name == path.name
        assert manifest.size_bytes == len(_PUBLISHED_BYTES)
        assert manifest.sha256 == hashlib.sha256(_PUBLISHED_BYTES).hexdigest()

    def test_manifest_records_stable_url_not_signed_url(
        self, tmp_path: Path, client: httpx.Client
    ) -> None:
        path = download(_DATASET_KEY, tmp_path, client=client)
        text = (path.parent / "manifest.json").read_text(encoding="utf-8")
        assert DownloadManifest.model_validate_json(text).source_url == _STABLE_URL
        assert "secret-signature" not in text

    def test_folder_holds_only_file_and_manifest(
        self, tmp_path: Path, client: httpx.Client
    ) -> None:
        path = download(_DATASET_KEY, tmp_path, client=client)
        assert set(path.parent.iterdir()) == {path, path.parent / "manifest.json"}

    def test_first_download_makes_git_ignore_raw_folder(
        self, tmp_path: Path, client: httpx.Client
    ) -> None:
        download(_DATASET_KEY, tmp_path, client=client)
        lines = (tmp_path / "raw" / ".gitignore").read_text(encoding="utf-8")
        assert "*" in lines.splitlines()

    def test_existing_gitignore_is_kept(
        self, tmp_path: Path, client: httpx.Client
    ) -> None:
        gitignore = tmp_path / "raw" / ".gitignore"
        gitignore.parent.mkdir()
        gitignore.write_text("mine\n", encoding="utf-8")
        download(_DATASET_KEY, tmp_path, client=client)
        assert gitignore.read_text(encoding="utf-8") == "mine\n"

    def test_token_is_sent_to_dataset_url_only(
        self, tmp_path: Path, client: httpx.Client, publisher: FakePublisher
    ) -> None:
        Config.set(f"{_DATASET_KEY}.requires_token", True)
        download(_DATASET_KEY, tmp_path, token="not-for-s3", client=client)
        stable, signed = publisher.requests
        assert stable.headers["Authorization"] == "Bearer not-for-s3"
        assert "Authorization" not in signed.headers

    def test_missing_token_raises_value_error_and_sends_nothing(
        self, tmp_path: Path, client: httpx.Client, publisher: FakePublisher
    ) -> None:
        Config.set(f"{_DATASET_KEY}.requires_token", True)
        with pytest.raises(ValueError, match="requires a token"):
            download(_DATASET_KEY, tmp_path, client=client)
        assert publisher.requests == []

    def test_refused_request_raises_download_error_and_leaves_nothing(
        self, tmp_path: Path, client: httpx.Client, publisher: FakePublisher
    ) -> None:
        publisher.signed_status = 403
        with pytest.raises(DownloadError, match="HTTP 403"):
            download(_DATASET_KEY, tmp_path, client=client)
        assert saved_files(tmp_path / "raw" / "epc") == []

    @pytest.mark.parametrize(
        ("signed_status", "signed_error"),
        [
            pytest.param(403, None, id="refused"),
            pytest.param(
                200,
                httpx.ReadTimeout(
                    f"timed out: {_SIGNED_URL}",
                    request=httpx.Request("GET", _SIGNED_URL),
                ),
                id="connection-failed",
            ),
        ],
    )
    def test_failed_request_error_hides_signed_url(
        self,
        tmp_path: Path,
        client: httpx.Client,
        publisher: FakePublisher,
        error_text: Callable[[BaseException], str],
        signed_status: int,
        signed_error: httpx.HTTPError | None,
    ) -> None:
        publisher.signed_status = signed_status
        publisher.signed_error = signed_error
        with pytest.raises(DownloadError) as caught:
            download(_DATASET_KEY, tmp_path, client=client)
        assert "secret-signature" not in error_text(caught.value)

    def test_wrong_size_raises_download_error_and_leaves_nothing(
        self, tmp_path: Path, client: httpx.Client
    ) -> None:
        Config.set(f"{_DATASET_KEY}.expected_bytes", len(_PUBLISHED_BYTES) + 1)
        with pytest.raises(DownloadError, match="expected"):
            download(_DATASET_KEY, tmp_path, client=client)
        assert saved_files(tmp_path / "raw" / "epc") == []

    def test_unknown_key_raises_key_error(
        self, tmp_path: Path, client: httpx.Client
    ) -> None:
        with pytest.raises(KeyError, match="No config setting named"):
            download("epc.scotland.display", tmp_path, client=client)

    def test_non_dataset_key_raises_type_error(
        self, tmp_path: Path, client: httpx.Client
    ) -> None:
        with pytest.raises(TypeError, match="not a dataset"):
            download("epc.scotland", tmp_path, client=client)

    def test_given_client_is_left_open(
        self, tmp_path: Path, client: httpx.Client
    ) -> None:
        download(_DATASET_KEY, tmp_path, client=client)
        assert not client.is_closed


class TestDownloadEpc:
    """`download_epc` downloads the EPC dataset for an origin and segment."""

    @pytest.mark.parametrize(
        ("origin", "segment"),
        [
            pytest.param("scotland", "domestic", id="strings"),
            pytest.param(EPCOrigin.SCOTLAND, EPCSegment.DOMESTIC, id="enum-members"),
        ],
    )
    def test_origin_and_segment_download_their_dataset(
        self,
        tmp_path: Path,
        client: httpx.Client,
        origin: EPCOrigin | str,
        segment: EPCSegment | str,
    ) -> None:
        path = download_epc(origin, segment, tmp_path, client=client)
        manifest = DownloadManifest.model_validate_json(
            (path.parent / "manifest.json").read_bytes()
        )
        assert manifest.dataset_key == _DATASET_KEY
        assert path.read_bytes() == _PUBLISHED_BYTES

    def test_token_is_passed_on_to_download(
        self, tmp_path: Path, client: httpx.Client, publisher: FakePublisher
    ) -> None:
        Config.set(f"{_DATASET_KEY}.requires_token", True)
        download_epc("scotland", "domestic", tmp_path, token="t", client=client)
        assert publisher.requests[0].headers["Authorization"] == "Bearer t"

    @pytest.mark.parametrize(
        ("origin", "segment", "match"),
        [
            pytest.param("scottland", "domestic", "unknown EPCOrigin", id="origin"),
            pytest.param("scotland", "homes", "unknown EPCSegment", id="segment"),
        ],
    )
    def test_unknown_name_raises_value_error_naming_choices(
        self,
        tmp_path: Path,
        client: httpx.Client,
        origin: str,
        segment: str,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=f"{match} .*; expected one of"):
            download_epc(origin, segment, tmp_path, client=client)

    def test_unpublished_pair_raises_key_error(
        self, tmp_path: Path, client: httpx.Client
    ) -> None:
        with pytest.raises(KeyError, match=r"epc\.scotland\.display"):
            download_epc("scotland", "display", tmp_path, client=client)
