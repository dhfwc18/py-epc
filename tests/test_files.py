"""Tests for writing folders safely."""

from pathlib import Path

import pytest

from py_epc._files import atomic_folder, ensure_gitignore


class TestAtomicFolder:
    """`atomic_folder` creates a folder only when its block succeeds."""

    def test_completed_block_renames_partial_folder_to_folder(
        self, tmp_path: Path
    ) -> None:
        folder = tmp_path / "parent" / "result"
        with atomic_folder(folder) as partial:
            (partial / "file.txt").write_text("done", encoding="utf-8")
        assert (folder / "file.txt").read_text(encoding="utf-8") == "done"
        assert list(folder.parent.iterdir()) == [folder]

    def test_failed_block_leaves_nothing_and_reraises(self, tmp_path: Path) -> None:
        folder = tmp_path / "result"
        with (
            pytest.raises(RuntimeError, match="failed"),
            atomic_folder(folder) as partial,
        ):
            (partial / "file.txt").write_text("half", encoding="utf-8")
            raise RuntimeError("failed")
        assert list(tmp_path.iterdir()) == []

    def test_partial_folder_is_hidden_beside_folder(self, tmp_path: Path) -> None:
        with atomic_folder(tmp_path / "result") as partial:
            assert partial == tmp_path / ".result.part"

    def test_existing_partial_folder_raises_file_exists_error(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / ".result.part").mkdir()
        with pytest.raises(FileExistsError), atomic_folder(tmp_path / "result"):
            pass


class TestEnsureGitignore:
    """`ensure_gitignore` makes Git ignore a folder, keeping any own file."""

    def test_folder_without_gitignore_gets_ignore_all_pattern(
        self, tmp_path: Path
    ) -> None:
        ensure_gitignore(tmp_path)
        lines = (tmp_path / ".gitignore").read_text(encoding="utf-8").splitlines()
        assert "*" in lines

    def test_existing_gitignore_is_kept(self, tmp_path: Path) -> None:
        (tmp_path / ".gitignore").write_text("mine\n", encoding="utf-8")
        ensure_gitignore(tmp_path)
        assert (tmp_path / ".gitignore").read_text(encoding="utf-8") == "mine\n"
