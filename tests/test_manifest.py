"""Tests for reading and writing manifests."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from py_epc import DownloadManifest
from py_epc._manifest import MANIFEST_NAME, read_manifest, write_manifest

# A complete download manifest.
_MANIFEST = DownloadManifest(
    source_key="epc.scotland.domestic",
    source_url="https://publisher.invalid/resource/download",
    publisher="p",
    retrieved_at=datetime(2026, 11, 11, 9, 30, tzinfo=UTC),
    file_name="epc-scotland-domestic.7z",
    size_bytes=3,
    sha256="0" * 64,
)


class TestWriteManifest:
    """`write_manifest` writes a manifest as JSON in a folder."""

    def test_written_manifest_ends_with_newline(self, tmp_path: Path) -> None:
        write_manifest(tmp_path, _MANIFEST)
        assert (tmp_path / MANIFEST_NAME).read_text(encoding="utf-8").endswith("}\n")


class TestReadManifest:
    """`read_manifest` reads a folder's manifest, checked against a model."""

    def test_written_manifest_reads_back_equal(self, tmp_path: Path) -> None:
        write_manifest(tmp_path, _MANIFEST)
        assert read_manifest(tmp_path, DownloadManifest) == _MANIFEST

    def test_missing_manifest_raises_file_not_found_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_manifest(tmp_path, DownloadManifest)

    def test_invalid_manifest_raises_validation_error(self, tmp_path: Path) -> None:
        (tmp_path / MANIFEST_NAME).write_text('{"version": 1}', encoding="utf-8")
        with pytest.raises(ValidationError):
            read_manifest(tmp_path, DownloadManifest)
