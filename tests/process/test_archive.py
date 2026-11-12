"""Tests for finding raw snapshots and reading their yearly files."""

import re
from collections.abc import Callable
from pathlib import Path

import pytest

from py_epc._manifest import MANIFEST_NAME
from py_epc._process._archive import find_snapshot, for_each_member

# Key of the source most tests use.
_SOURCE_KEY = "epc.scotland.domestic"

# Yearly members as each register names them, the pattern that picks them, and
# a member that must be skipped.
_ARCHIVES = [
    pytest.param(
        "7z",
        "epc.scotland.domestic",
        "{year}.csv",
        re.compile(r"(\d{4})\.csv"),
        "Publication Note.pdf",
        id="scotland-7z",
    ),
    pytest.param(
        "zip",
        "epc.england-and-wales.domestic",
        "certificates-{year}.csv",
        re.compile(r"certificates-(\d{4})\.csv"),
        "recommendations-2025.csv",
        id="england-and-wales-zip",
    ),
]


class TestFindSnapshot:
    """`find_snapshot` returns the folder of a completed download."""

    def test_single_snapshot_returns_its_folder(
        self, tmp_path: Path, make_snapshot: Callable[..., Path]
    ) -> None:
        folder = make_snapshot(_SOURCE_KEY, {"2025.csv": "a\n"})
        assert find_snapshot(tmp_path, _SOURCE_KEY) == folder

    def test_several_snapshots_return_newest(
        self, tmp_path: Path, make_snapshot: Callable[..., Path]
    ) -> None:
        make_snapshot(_SOURCE_KEY, {"2025.csv": "a\n"}, stamp="20261011T093000Z")
        newest = make_snapshot(
            _SOURCE_KEY, {"2025.csv": "a\n"}, stamp="20261111T093000Z"
        )
        assert find_snapshot(tmp_path, _SOURCE_KEY) == newest

    def test_named_snapshot_returns_that_folder(
        self, tmp_path: Path, make_snapshot: Callable[..., Path]
    ) -> None:
        older = make_snapshot(
            _SOURCE_KEY, {"2025.csv": "a\n"}, stamp="20261011T093000Z"
        )
        make_snapshot(_SOURCE_KEY, {"2025.csv": "a\n"}, stamp="20261111T093000Z")
        assert find_snapshot(tmp_path, _SOURCE_KEY, "20261011T093000Z") == older

    def test_newer_partial_folder_is_ignored(
        self, tmp_path: Path, make_snapshot: Callable[..., Path]
    ) -> None:
        folder = make_snapshot(_SOURCE_KEY, {"2025.csv": "a\n"})
        (folder.parent / ".20261211T093000Z.part").mkdir()
        assert find_snapshot(tmp_path, _SOURCE_KEY) == folder

    def test_newer_folder_without_manifest_is_ignored(
        self, tmp_path: Path, make_snapshot: Callable[..., Path]
    ) -> None:
        folder = make_snapshot(_SOURCE_KEY, {"2025.csv": "a\n"})
        (folder.parent / "20261211T093000Z").mkdir()
        assert find_snapshot(tmp_path, _SOURCE_KEY) == folder

    def test_root_starting_with_tilde_is_expanded_to_home(
        self,
        tmp_path: Path,
        make_snapshot: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        folder = make_snapshot(_SOURCE_KEY, {"2025.csv": "a\n"})
        assert find_snapshot("~", _SOURCE_KEY) == folder

    def test_no_snapshot_raises_file_not_found_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            find_snapshot(tmp_path, _SOURCE_KEY)

    def test_unknown_named_snapshot_raises_file_not_found_error(
        self, tmp_path: Path, make_snapshot: Callable[..., Path]
    ) -> None:
        make_snapshot(_SOURCE_KEY, {"2025.csv": "a\n"})
        with pytest.raises(FileNotFoundError):
            find_snapshot(tmp_path, _SOURCE_KEY, "20200101T000000Z")


@pytest.mark.parametrize(("archive", "key", "name", "pattern", "other"), _ARCHIVES)
class TestForEachMember:
    """`for_each_member` hands over each yearly file, then deletes it."""

    def test_matching_members_are_handled_once_with_their_year(
        self,
        tmp_path: Path,
        make_snapshot: Callable[..., Path],
        archive: str,
        key: str,
        name: str,
        pattern: re.Pattern[str],
        other: str,
    ) -> None:
        members = {name.format(year=y): f"year {y}\n" for y in (2024, 2025, 2026)}
        snapshot = make_snapshot(key, members, archive=archive)
        handled: list[tuple[int, str]] = []
        for_each_member(
            snapshot,
            pattern,
            lambda year, path: handled.append((year, path.read_text(encoding="utf-8"))),
            scratch=tmp_path,
        )
        assert sorted(handled) == [
            (2024, "year 2024\n"),
            (2025, "year 2025\n"),
            (2026, "year 2026\n"),
        ]

    def test_other_members_are_skipped(
        self,
        tmp_path: Path,
        make_snapshot: Callable[..., Path],
        archive: str,
        key: str,
        name: str,
        pattern: re.Pattern[str],
        other: str,
    ) -> None:
        members = {name.format(year=2025): "certificates\n", other: b"other"}
        snapshot = make_snapshot(key, members, archive=archive)
        years: list[int] = []
        for_each_member(
            snapshot, pattern, lambda year, _: years.append(year), scratch=tmp_path
        )
        assert years == [2025]

    def test_temporary_file_is_in_scratch_and_deleted_after_handling(
        self,
        tmp_path: Path,
        make_snapshot: Callable[..., Path],
        archive: str,
        key: str,
        name: str,
        pattern: re.Pattern[str],
        other: str,
    ) -> None:
        snapshot = make_snapshot(key, {name.format(year=2025): "a\n"}, archive=archive)
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        paths: list[Path] = []
        for_each_member(
            snapshot, pattern, lambda _, path: paths.append(path), scratch=scratch
        )
        assert [path.parent for path in paths] == [scratch]
        assert list(scratch.iterdir()) == []

    def test_failed_handle_raises_its_error_and_leaves_scratch_empty(
        self,
        tmp_path: Path,
        make_snapshot: Callable[..., Path],
        archive: str,
        key: str,
        name: str,
        pattern: re.Pattern[str],
        other: str,
    ) -> None:
        snapshot = make_snapshot(key, {name.format(year=2025): "a\n"}, archive=archive)
        scratch = tmp_path / "scratch"
        scratch.mkdir()

        def fail(year: int, path: Path) -> None:
            raise RuntimeError("handle failed")

        with pytest.raises(RuntimeError, match="handle failed"):
            for_each_member(snapshot, pattern, fail, scratch=scratch)
        assert list(scratch.iterdir()) == []

    def test_missing_manifest_raises_file_not_found_error(
        self,
        tmp_path: Path,
        make_snapshot: Callable[..., Path],
        archive: str,
        key: str,
        name: str,
        pattern: re.Pattern[str],
        other: str,
    ) -> None:
        snapshot = make_snapshot(key, {name.format(year=2025): "a\n"}, archive=archive)
        (snapshot / MANIFEST_NAME).unlink()
        with pytest.raises(FileNotFoundError):
            for_each_member(snapshot, pattern, lambda *_: None, scratch=tmp_path)

    def test_changed_archive_raises_value_error(
        self,
        tmp_path: Path,
        make_snapshot: Callable[..., Path],
        archive: str,
        key: str,
        name: str,
        pattern: re.Pattern[str],
        other: str,
    ) -> None:
        snapshot = make_snapshot(key, {name.format(year=2025): "a\n"}, archive=archive)
        file = next(path for path in snapshot.iterdir() if path.name != MANIFEST_NAME)
        file.write_bytes(file.read_bytes() + b"changed")
        with pytest.raises(ValueError, match="SHA-256"):
            for_each_member(snapshot, pattern, lambda *_: None, scratch=tmp_path)
