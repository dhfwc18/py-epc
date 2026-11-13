"""Tests for reading a processed build."""

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from py_epc import EPCOrigin, EPCSegment, ProcessManifest, scan, scan_epc
from py_epc._manifest import MANIFEST_FILE, write_manifest
from py_epc._process._latest import LatestSummary

# Key of the source the tests read.
_SOURCE_KEY = "epc.scotland.domestic"

# Latest counts of a build, whose values the tests do not read.
_LATEST = LatestSummary(
    properties=1,
    without_property=0,
    blank_certificate_numbers=0,
    duplicate_certificate_numbers=0,
    superseded_disagreements=None,
)


def write_build(
    root: Path,
    years: dict[int, list[tuple[str, bool]]],
    *,
    snapshot: str = "20261111T093000Z",
    with_latest: bool = True,
) -> Path:
    """Write a build of (certificate number, is latest) pairs per year."""
    folder = root.joinpath("processed", *_SOURCE_KEY.split("."), snapshot)
    for year, rows in years.items():
        target = folder / "certificates" / f"lodgement_year={year}"
        target.mkdir(parents=True)
        frame = pl.DataFrame(
            {
                "certificate_number": [number for number, _ in rows],
                "is_latest": [latest for _, latest in rows],
            }
        )
        if not with_latest:
            frame = frame.drop("is_latest")
        frame.write_parquet(target / "data.parquet")
    write_manifest(
        folder,
        ProcessManifest(
            source_key=_SOURCE_KEY,
            snapshot=snapshot,
            raw_sha256="0" * 64,
            register_version=1,
            package_version="0",
            processed_at=datetime(2026, 11, 11, tzinfo=UTC),
            years=(),
            latest=_LATEST if with_latest else None,
        ),
    )
    return folder


def numbers(frame: pl.LazyFrame) -> list[str]:
    """Return the certificate numbers in a frame, sorted."""
    return sorted(frame.collect()["certificate_number"])


class TestScan:
    """`scan` reads a processed build as a lazy frame."""

    def test_build_returns_every_certificate_with_its_year(
        self, tmp_path: Path
    ) -> None:
        write_build(tmp_path, {2024: [("a", False)], 2025: [("b", True)]})
        frame = scan(_SOURCE_KEY, tmp_path).sort("certificate_number").collect()
        assert frame.select("certificate_number", "lodgement_year").rows() == [
            ("a", 2024),
            ("b", 2025),
        ]

    def test_scan_returns_lazy_frame(self, tmp_path: Path) -> None:
        write_build(tmp_path, {2024: [("a", True)]})
        assert isinstance(scan(_SOURCE_KEY, tmp_path), pl.LazyFrame)

    def test_latest_keeps_only_latest_certificates(self, tmp_path: Path) -> None:
        write_build(tmp_path, {2024: [("a", False), ("b", True)], 2025: [("c", True)]})
        assert numbers(scan(_SOURCE_KEY, tmp_path, latest=True)) == ["b", "c"]

    def test_several_builds_read_newest(self, tmp_path: Path) -> None:
        write_build(tmp_path, {2024: [("old", True)]}, snapshot="20261011T093000Z")
        write_build(tmp_path, {2024: [("new", True)]}, snapshot="20261111T093000Z")
        assert numbers(scan(_SOURCE_KEY, tmp_path)) == ["new"]

    def test_named_snapshot_reads_its_build(self, tmp_path: Path) -> None:
        write_build(tmp_path, {2024: [("old", True)]}, snapshot="20261011T093000Z")
        write_build(tmp_path, {2024: [("new", True)]}, snapshot="20261111T093000Z")
        frame = scan(_SOURCE_KEY, tmp_path, snapshot="20261011T093000Z")
        assert numbers(frame) == ["old"]

    def test_build_without_manifest_is_ignored(self, tmp_path: Path) -> None:
        write_build(tmp_path, {2024: [("done", True)]}, snapshot="20261011T093000Z")
        newer = write_build(
            tmp_path, {2024: [("unfinished", True)]}, snapshot="20261111T093000Z"
        )
        (newer / MANIFEST_FILE).unlink()
        assert numbers(scan(_SOURCE_KEY, tmp_path)) == ["done"]

    def test_root_starting_with_tilde_is_expanded_to_home(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        write_build(tmp_path, {2024: [("a", True)]})
        assert numbers(scan(_SOURCE_KEY, "~")) == ["a"]

    def test_no_build_raises_file_not_found_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            scan(_SOURCE_KEY, tmp_path)

    def test_latest_without_flag_raises_value_error(self, tmp_path: Path) -> None:
        write_build(tmp_path, {2024: [("a", True)]}, with_latest=False)
        with pytest.raises(ValueError, match="has no is_latest"):
            scan(_SOURCE_KEY, tmp_path, latest=True)


class TestScanEpc:
    """`scan_epc` reads the EPC build for an origin and segment."""

    @pytest.mark.parametrize(
        ("origin", "segment"),
        [
            pytest.param("scotland", "domestic", id="strings"),
            pytest.param(EPCOrigin.SCOTLAND, EPCSegment.DOMESTIC, id="enum-members"),
        ],
    )
    def test_origin_and_segment_read_their_build(
        self,
        tmp_path: Path,
        origin: EPCOrigin | str,
        segment: EPCSegment | str,
    ) -> None:
        write_build(tmp_path, {2024: [("a", False), ("b", True)]})
        frame = scan_epc(origin, segment, tmp_path, latest=True)
        assert numbers(frame) == ["b"]

    def test_unknown_segment_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown EPCSegment"):
            scan_epc("scotland", "homes", tmp_path)
