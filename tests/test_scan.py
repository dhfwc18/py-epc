"""Tests for reading a processed build."""

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from py_epc import (
    EPCOrigin,
    EPCSegment,
    PostcodeManifest,
    ProcessManifest,
    scan,
    scan_epc,
)
from py_epc._manifest import MANIFEST_FILE, write_manifest
from py_epc._process._latest import LatestSummary
from py_epc._process._postcodes import FieldSummary

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
    years: dict[int, list[tuple[str, bool]] | list[tuple[str, bool, str]]],
    *,
    snapshot: str = "20261111T093000Z",
    with_latest: bool = True,
) -> Path:
    """
    Write a build of certificates per year.

    Each certificate is a certificate number, whether it is the latest, and
    optionally its postcode, which is ``"AB10 1AA"`` if not given.
    """
    folder = root.joinpath("processed", *_SOURCE_KEY.split("."), snapshot)
    for year, rows in years.items():
        target = folder / "certificates" / f"lodgement_year={year}"
        target.mkdir(parents=True)
        frame = pl.DataFrame(
            {
                "certificate_number": [row[0] for row in rows],
                "postcode": [row[2] if len(row) > 2 else "AB10 1AA" for row in rows],
                "is_latest": [row[1] for row in rows],
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


def write_postcodes(
    root: Path,
    rows: dict[str, tuple[str, str]],
    *,
    snapshot: str = "20261111T093000Z",
) -> Path:
    """Write a processed postcode directory of postcode to (code, name)."""
    folder = root.joinpath("processed", "ons", "postcode-directory", snapshot)
    folder.mkdir(parents=True)
    pl.DataFrame(
        {
            "postcode": list(rows),
            "lad_code": [code for code, _ in rows.values()],
            "lad_name": [name for _, name in rows.values()],
        }
    ).write_parquet(folder / "postcodes.parquet")
    write_manifest(
        folder,
        PostcodeManifest(
            source_key="ons.postcode-directory",
            snapshot=snapshot,
            raw_sha256="0" * 64,
            package_version="0",
            processed_at=datetime(2026, 11, 11, tzinfo=UTC),
            release="AUG_2026",
            rows=len(rows),
            excluded_rows=0,
            names=True,
            fields={
                "lad": FieldSummary(
                    column="lad26cd", lookup_files=("LAD.csv",), unmatched=0
                )
            },
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


class TestScanGeography:
    """`scan` with `geography` adds the areas of certificates' postcodes."""

    def test_matched_postcode_gets_prefixed_areas(self, tmp_path: Path) -> None:
        write_build(tmp_path, {2024: [("a", True, "AB10 1AA")]})
        write_postcodes(tmp_path, {"AB10 1AA": ("S12000033", "Aberdeen City")})
        frame = scan(_SOURCE_KEY, tmp_path, geography=True).collect()
        assert frame.select(
            "certificate_number", "ons_lad_code", "ons_lad_name", "ons_matched"
        ).rows() == [("a", "S12000033", "Aberdeen City", True)]

    def test_unmatched_postcode_gets_nulls_and_false(self, tmp_path: Path) -> None:
        write_build(tmp_path, {2024: [("a", True, "ZZ9 9ZZ"), ("b", True, "N/A")]})
        write_postcodes(tmp_path, {"AB10 1AA": ("S12000033", "Aberdeen City")})
        frame = scan(_SOURCE_KEY, tmp_path, geography=True).sort("certificate_number")
        assert frame.select("ons_lad_code", "ons_matched").collect().rows() == [
            (None, False),
            (None, False),
        ]

    def test_postcode_in_other_form_is_matched(self, tmp_path: Path) -> None:
        write_build(tmp_path, {2024: [("a", True, "ab101aa")]})
        write_postcodes(tmp_path, {"AB10 1AA": ("S12000033", "Aberdeen City")})
        frame = scan(_SOURCE_KEY, tmp_path, geography=True).collect()
        assert frame["ons_lad_code"].to_list() == ["S12000033"]

    def test_certificate_keeps_its_published_postcode(self, tmp_path: Path) -> None:
        write_build(tmp_path, {2024: [("a", True, "ab101aa")]})
        write_postcodes(tmp_path, {"AB10 1AA": ("S12000033", "Aberdeen City")})
        frame = scan(_SOURCE_KEY, tmp_path, geography=True).collect()
        assert frame["postcode"].to_list() == ["ab101aa"]
        assert "_ons_postcode" not in frame.columns

    def test_latest_and_geography_combine(self, tmp_path: Path) -> None:
        write_build(tmp_path, {2024: [("old", False), ("new", True)]})
        write_postcodes(tmp_path, {"AB10 1AA": ("S12000033", "Aberdeen City")})
        frame = scan(_SOURCE_KEY, tmp_path, latest=True, geography=True).collect()
        assert frame.select("certificate_number", "ons_matched").rows() == [
            ("new", True)
        ]

    def test_newest_postcode_directory_is_used(self, tmp_path: Path) -> None:
        write_build(tmp_path, {2024: [("a", True)]})
        write_postcodes(
            tmp_path, {"AB10 1AA": ("S1", "Old")}, snapshot="20260511T093000Z"
        )
        write_postcodes(
            tmp_path, {"AB10 1AA": ("S2", "New")}, snapshot="20261111T093000Z"
        )
        frame = scan(_SOURCE_KEY, tmp_path, geography=True).collect()
        assert frame["ons_lad_name"].to_list() == ["New"]

    def test_without_geography_no_areas_are_added(self, tmp_path: Path) -> None:
        write_build(tmp_path, {2024: [("a", True)]})
        write_postcodes(tmp_path, {"AB10 1AA": ("S12000033", "Aberdeen City")})
        columns = scan(_SOURCE_KEY, tmp_path).collect_schema().names()
        assert not any(name.startswith("ons_") for name in columns)

    def test_missing_postcode_directory_raises_file_not_found_error(
        self, tmp_path: Path
    ) -> None:
        write_build(tmp_path, {2024: [("a", True)]})
        with pytest.raises(FileNotFoundError, match="process_ons_postcode_directory"):
            scan(_SOURCE_KEY, tmp_path, geography=True)


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

    def test_geography_is_passed_on_to_scan(self, tmp_path: Path) -> None:
        write_build(tmp_path, {2024: [("a", True)]})
        write_postcodes(tmp_path, {"AB10 1AA": ("S12000033", "Aberdeen City")})
        frame = scan_epc("scotland", "domestic", tmp_path, geography=True)
        assert frame.collect()["ons_lad_name"].to_list() == ["Aberdeen City"]

    def test_unknown_segment_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown EPCSegment"):
            scan_epc("scotland", "homes", tmp_path)
