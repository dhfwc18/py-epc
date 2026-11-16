"""Tests for running every step for an EPC origin and segment in one call."""

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from py_epc import _pipeline, run_epc_pipeline
from py_epc._manifest import MANIFEST_FILE

# Keys of the registers, and of the postcode directory.
_SCOTLAND_KEY = "epc.scotland.domestic"
_ENGLAND_AND_WALES_KEY = "epc.england-and-wales.domestic"
_POSTCODES_KEY = "ons.postcode-directory"


def complete(root: Path, stage: str, key: str, stamp: str) -> Path:
    """Write a completed folder of a step, as the step would leave it."""
    folder = root.joinpath(stage, *key.split("."), stamp)
    folder.mkdir(parents=True)
    (folder / MANIFEST_FILE).write_text("{}", encoding="utf-8")
    return folder


class Steps:
    """Fakes of every step, recording each call and writing its folder."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str | None]] = []
        self.scanned: dict[str, Any] = {}
        self.downloads = 0

    def download(self, key: str, root: Path, *, token: str | None) -> Path:
        self.downloads += 1
        stamp = f"2026010{self.downloads}T000000Z"
        self.calls.append(("download", key, token))
        return complete(root, "raw", key, stamp) / "file.zip"

    def process(self, key: str, root: Path, *, snapshot: str) -> Path:
        self.calls.append(("process", key, snapshot))
        return complete(root, "processed", key, snapshot)

    def build_dataset(
        self, key: str, root: Path, *, snapshot: str, strict: bool
    ) -> Path:
        self.calls.append(("build_dataset", key, snapshot))
        return complete(root, "datasets", key, snapshot)

    def download_postcodes(self, root: Path) -> Path:
        self.calls.append(("download", _POSTCODES_KEY, None))
        return complete(root, "raw", _POSTCODES_KEY, "20260101T000000Z") / "x.zip"

    def process_postcodes(self, root: Path, *, snapshot: str) -> Path:
        self.calls.append(("process", _POSTCODES_KEY, snapshot))
        return complete(root, "processed", _POSTCODES_KEY, snapshot)

    def scan_epc(self, origin: Any, segment: Any, root: Any, **options: Any) -> Any:
        self.scanned = {"origin": str(origin), **options}
        return pl.LazyFrame({"origin": [str(origin)]})


@pytest.fixture
def steps(monkeypatch: pytest.MonkeyPatch) -> Steps:
    """Replace every step of the pipeline with a recording fake."""
    fake = Steps()
    monkeypatch.setattr(_pipeline, "download", fake.download)
    monkeypatch.setattr(_pipeline, "process", fake.process)
    monkeypatch.setattr(_pipeline, "build_dataset", fake.build_dataset)
    monkeypatch.setattr(
        _pipeline, "download_ons_postcode_directory", fake.download_postcodes
    )
    monkeypatch.setattr(
        _pipeline, "process_ons_postcode_directory", fake.process_postcodes
    )
    monkeypatch.setattr(_pipeline, "scan_epc", fake.scan_epc)
    return fake


class TestRunEpcPipeline:
    """`run_epc_pipeline` makes what is missing, then reads it."""

    def test_empty_root_runs_every_step_and_reads_dataset(
        self, tmp_path: Path, steps: Steps
    ) -> None:
        frame = run_epc_pipeline("scotland", "domestic", tmp_path)
        assert steps.calls == [
            ("download", _SCOTLAND_KEY, None),
            ("process", _SCOTLAND_KEY, "20260101T000000Z"),
            ("build_dataset", _SCOTLAND_KEY, "20260101T000000Z"),
        ]
        assert steps.scanned == {
            "origin": "scotland",
            "snapshot": "20260101T000000Z",
            "stage": "dataset",
            "latest": False,
            "geography": False,
        }
        assert isinstance(frame, pl.LazyFrame)

    def test_steps_already_done_are_not_done_again(
        self, tmp_path: Path, steps: Steps
    ) -> None:
        for stage in ("raw", "processed", "datasets"):
            complete(tmp_path, stage, _SCOTLAND_KEY, "20260101T000000Z")
        run_epc_pipeline("scotland", "domestic", tmp_path)
        assert steps.calls == []

    def test_download_without_build_is_processed_and_built(
        self, tmp_path: Path, steps: Steps
    ) -> None:
        complete(tmp_path, "raw", _SCOTLAND_KEY, "20251201T000000Z")
        run_epc_pipeline("scotland", "domestic", tmp_path)
        assert steps.calls == [
            ("process", _SCOTLAND_KEY, "20251201T000000Z"),
            ("build_dataset", _SCOTLAND_KEY, "20251201T000000Z"),
        ]

    def test_processed_stage_builds_no_dataset(
        self, tmp_path: Path, steps: Steps
    ) -> None:
        run_epc_pipeline("scotland", "domestic", tmp_path, stage="processed")
        assert [call[0] for call in steps.calls] == ["download", "process"]
        assert steps.scanned["stage"] == "processed"

    def test_update_downloads_again_and_builds_from_new_snapshot(
        self, tmp_path: Path, steps: Steps
    ) -> None:
        for stage in ("raw", "processed", "datasets"):
            complete(tmp_path, stage, _SCOTLAND_KEY, "20251201T000000Z")
        run_epc_pipeline("scotland", "domestic", tmp_path, update=True)
        assert steps.calls == [
            ("download", _SCOTLAND_KEY, None),
            ("process", _SCOTLAND_KEY, "20260101T000000Z"),
            ("build_dataset", _SCOTLAND_KEY, "20260101T000000Z"),
        ]
        assert steps.scanned["snapshot"] == "20260101T000000Z"

    def test_token_is_given_to_download(self, tmp_path: Path, steps: Steps) -> None:
        run_epc_pipeline("england-and-wales", "domestic", tmp_path, token="t")
        assert steps.calls[0] == ("download", _ENGLAND_AND_WALES_KEY, "t")

    def test_gb_prepares_both_registers_and_postcodes(
        self, tmp_path: Path, steps: Steps
    ) -> None:
        run_epc_pipeline("gb", "domestic", tmp_path, token="t")
        assert [(call[0], call[1]) for call in steps.calls] == [
            ("download", _SCOTLAND_KEY),
            ("process", _SCOTLAND_KEY),
            ("build_dataset", _SCOTLAND_KEY),
            ("download", _ENGLAND_AND_WALES_KEY),
            ("process", _ENGLAND_AND_WALES_KEY),
            ("build_dataset", _ENGLAND_AND_WALES_KEY),
            ("download", _POSTCODES_KEY),
            ("process", _POSTCODES_KEY),
        ]
        assert (steps.scanned["snapshot"], steps.scanned["geography"]) == (None, True)

    def test_processed_postcodes_are_not_made_again(
        self, tmp_path: Path, steps: Steps
    ) -> None:
        complete(tmp_path, "processed", _POSTCODES_KEY, "20251201T000000Z")
        run_epc_pipeline("scotland", "domestic", tmp_path, geography=True)
        assert _POSTCODES_KEY not in [call[1] for call in steps.calls]

    def test_segment_without_definition_raises_before_downloading(
        self, tmp_path: Path, steps: Steps
    ) -> None:
        with pytest.raises(KeyError):
            run_epc_pipeline("scotland", "non-domestic", tmp_path)
        assert steps.calls == []

    def test_unknown_origin_raises_value_error(
        self, tmp_path: Path, steps: Steps
    ) -> None:
        with pytest.raises(ValueError, match="unknown EPCOrigin"):
            run_epc_pipeline("wales", "domestic", tmp_path)
