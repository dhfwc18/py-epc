"""Tests for running every step for an EPC origin and segment in one call."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from pydantic import BaseModel

from py_epc import PostcodeManifest, ProcessManifest, _pipeline, run_epc_pipeline
from py_epc._manifest import MANIFEST_FILE, DownloadManifest, write_manifest
from py_epc._pipeline import OutputManifest, output_manifest_path
from py_epc._process._postcodes import FieldSummary

# Keys of the registers, and of the postcode directory.
_SCOTLAND_KEY = "epc.scotland.domestic"
_ENGLAND_AND_WALES_KEY = "epc.england-and-wales.domestic"
_POSTCODES_KEY = "ons.postcode-directory"


# Certificates the fake scan returns.
_FRAME = pl.LazyFrame({"certificate_number": ["a", "b"]})

# When the fake steps ran.
_WHEN = datetime(2026, 1, 1, tzinfo=UTC)


def complete(
    root: Path, stage: str, key: str, stamp: str, manifest: BaseModel | None = None
) -> Path:
    """Write a completed folder of a step, as the step would leave it."""
    folder = root.joinpath(stage, *key.split("."), stamp)
    folder.mkdir(parents=True)
    if manifest is None:
        (folder / MANIFEST_FILE).write_text("{}", encoding="utf-8")
    else:
        write_manifest(folder, manifest)
    return folder


def raw_sha256(stamp: str) -> str:
    """Return the checksum the fakes give the download of `stamp`."""
    return stamp[:8].ljust(64, "0")


def downloaded(root: Path, key: str, stamp: str, sha256: str | None = None) -> Path:
    """Write a completed raw snapshot, with the manifest `download` writes."""
    manifest = DownloadManifest(
        source_key=key,
        source_url="https://example.org/file.zip",
        publisher="p",
        retrieved_at=_WHEN,
        file_name="file.zip",
        size_bytes=1,
        sha256=raw_sha256(stamp) if sha256 is None else sha256,
    )
    return complete(root, "raw", key, stamp, manifest)


class Steps:
    """Fakes of every step, recording each call and writing its folder."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str | None]] = []
        self.scanned: dict[str, Any] = {}
        self.downloads = 0
        self.sha256: str | None = None
        """Checksum of every download, if not one for each snapshot."""

    def download(self, key: str, root: Path, *, token: str | None) -> Path:
        self.downloads += 1
        stamp = f"2026010{self.downloads}T000000Z"
        self.calls.append(("download", key, token))
        return downloaded(root, key, stamp, self.sha256) / "file.zip"

    def process(self, key: str, root: Path, *, snapshot: str) -> Path:
        self.calls.append(("process", key, snapshot))
        manifest = ProcessManifest(
            source_key=key,
            snapshot=snapshot,
            raw_sha256=raw_sha256(snapshot),
            register_version=1,
            package_version="0",
            processed_at=_WHEN,
            years=(),
            latest=None,
        )
        return complete(root, "processed", key, snapshot, manifest)

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
        manifest = PostcodeManifest(
            source_key=_POSTCODES_KEY,
            snapshot=snapshot,
            raw_sha256=raw_sha256(snapshot),
            package_version="0",
            processed_at=_WHEN,
            release="AUG_2026",
            rows=1,
            excluded_rows=0,
            names=True,
            fields={
                "lad": FieldSummary(column="lad26cd", lookup_files=(), unmatched=0)
            },
        )
        return complete(root, "processed", _POSTCODES_KEY, snapshot, manifest)

    def scan_epc(self, origin: Any, segment: Any, root: Any, **options: Any) -> Any:
        self.scanned = {"origin": str(origin), **options}
        return _FRAME


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
        downloaded(tmp_path, _SCOTLAND_KEY, "20251201T000000Z")
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
        downloaded(tmp_path, _SCOTLAND_KEY, "20251201T000000Z")
        for stage in ("processed", "datasets"):
            complete(tmp_path, stage, _SCOTLAND_KEY, "20251201T000000Z")
        run_epc_pipeline("scotland", "domestic", tmp_path, update=True)
        assert steps.calls == [
            ("download", _SCOTLAND_KEY, None),
            ("process", _SCOTLAND_KEY, "20260101T000000Z"),
            ("build_dataset", _SCOTLAND_KEY, "20260101T000000Z"),
        ]
        assert steps.scanned["snapshot"] == "20260101T000000Z"

    def test_update_with_same_download_deletes_it_and_reuses_build(
        self, tmp_path: Path, steps: Steps
    ) -> None:
        downloaded(tmp_path, _SCOTLAND_KEY, "20251201T000000Z", "a" * 64)
        for stage in ("processed", "datasets"):
            complete(tmp_path, stage, _SCOTLAND_KEY, "20251201T000000Z")
        steps.sha256 = "a" * 64
        run_epc_pipeline("scotland", "domestic", tmp_path, update=True)
        assert steps.calls == [("download", _SCOTLAND_KEY, None)]
        assert steps.scanned["snapshot"] == "20251201T000000Z"
        raws = tmp_path.joinpath("raw", *_SCOTLAND_KEY.split("."))
        assert [folder.name for folder in raws.iterdir()] == ["20251201T000000Z"]

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


class TestRunEpcPipelineOutput:
    """`run_epc_pipeline` writes the certificates to a file, if asked."""

    def test_output_type_writes_default_file_and_returns_its_path(
        self, tmp_path: Path, steps: Steps
    ) -> None:
        path = run_epc_pipeline("scotland", "domestic", tmp_path, output_type="parquet")
        assert path == tmp_path / "outputs" / "epc-scotland-domestic-dataset.parquet"
        assert pl.read_parquet(path).equals(_FRAME.collect())

    def test_latest_is_named_in_default_file(
        self, tmp_path: Path, steps: Steps
    ) -> None:
        path = run_epc_pipeline(
            "scotland", "domestic", tmp_path, latest=True, output_type="sqlite"
        )
        assert path.name == "epc-scotland-domestic-dataset-latest.sqlite"

    def test_manifest_beside_output_names_every_download(
        self, tmp_path: Path, steps: Steps
    ) -> None:
        path = run_epc_pipeline(
            "gb", "domestic", tmp_path, token="t", output_type="parquet"
        )
        manifest = OutputManifest.model_validate_json(
            output_manifest_path(path).read_bytes()
        )
        assert (manifest.origin, manifest.stage, manifest.geography) == (
            "gb",
            "dataset",
            True,
        )
        assert manifest.file_name == path.name
        assert [
            (source.source_key, source.snapshot, source.raw_sha256)
            for source in manifest.sources
        ] == [
            (_SCOTLAND_KEY, "20260101T000000Z", raw_sha256("20260101T000000Z")),
            (
                _ENGLAND_AND_WALES_KEY,
                "20260102T000000Z",
                raw_sha256("20260102T000000Z"),
            ),
            (_POSTCODES_KEY, "20260101T000000Z", raw_sha256("20260101T000000Z")),
        ]

    def test_first_default_output_makes_git_ignore_outputs(
        self, tmp_path: Path, steps: Steps
    ) -> None:
        run_epc_pipeline("scotland", "domestic", tmp_path, output_type="parquet")
        assert (tmp_path / "outputs" / ".gitignore").is_file()

    def test_output_path_writes_there_with_manifest_beside(
        self, tmp_path: Path, steps: Steps
    ) -> None:
        target = tmp_path / "elsewhere" / "homes.parquet"
        path = run_epc_pipeline(
            "scotland",
            "domestic",
            tmp_path / "data",
            output_type="parquet",
            output_path=target,
        )
        assert path == target
        assert (tmp_path / "elsewhere" / "homes.manifest.json").is_file()

    def test_existing_output_raises_file_exists_error_before_any_step(
        self, tmp_path: Path, steps: Steps
    ) -> None:
        run_epc_pipeline("scotland", "domestic", tmp_path, output_type="parquet")
        steps.calls.clear()
        with pytest.raises(FileExistsError, match="overwrite=True"):
            run_epc_pipeline("scotland", "domestic", tmp_path, output_type="parquet")
        assert steps.calls == []

    def test_existing_output_with_overwrite_is_replaced(
        self, tmp_path: Path, steps: Steps
    ) -> None:
        path = run_epc_pipeline("scotland", "domestic", tmp_path, output_type="parquet")
        path.write_bytes(b"old")
        run_epc_pipeline(
            "scotland", "domestic", tmp_path, output_type="parquet", overwrite=True
        )
        assert pl.read_parquet(path).equals(_FRAME.collect())
        manifest = json.loads(output_manifest_path(path).read_text(encoding="utf-8"))
        assert manifest["output_type"] == "parquet"

    def test_unknown_output_type_raises_value_error(
        self, tmp_path: Path, steps: Steps
    ) -> None:
        with pytest.raises(ValueError, match="unknown OutputType"):
            run_epc_pipeline("scotland", "domestic", tmp_path, output_type="csv")
        assert steps.calls == []


class TestRunEpcPipelineCleanUp:
    """`run_epc_pipeline` deletes what an output was made from, if asked."""

    def test_clean_up_deletes_snapshots_read_and_keeps_output(
        self, tmp_path: Path, steps: Steps
    ) -> None:
        path = run_epc_pipeline(
            "scotland", "domestic", tmp_path, output_type="parquet", clean_up=True
        )
        snapshot = "20260101T000000Z"
        assert [
            tmp_path.joinpath(stage, *_SCOTLAND_KEY.split("."), snapshot).exists()
            for stage in ("raw", "processed", "datasets")
        ] == [False, False, False]
        assert path.is_file()
        assert output_manifest_path(path).is_file()

    def test_clean_up_keeps_postcodes_and_other_snapshots(
        self, tmp_path: Path, steps: Steps
    ) -> None:
        older = downloaded(tmp_path, _SCOTLAND_KEY, "20250101T000000Z")
        run_epc_pipeline(
            "gb",
            "domestic",
            tmp_path,
            token="t",
            update=True,
            output_type="parquet",
            clean_up=True,
        )
        assert older.is_dir()
        assert tmp_path.joinpath("processed", *_POSTCODES_KEY.split(".")).is_dir()
        assert not tmp_path.joinpath(
            "raw", *_ENGLAND_AND_WALES_KEY.split("."), "20260102T000000Z"
        ).exists()

    def test_without_clean_up_every_snapshot_is_kept(
        self, tmp_path: Path, steps: Steps
    ) -> None:
        run_epc_pipeline("scotland", "domestic", tmp_path, output_type="parquet")
        assert tmp_path.joinpath(
            "datasets", *_SCOTLAND_KEY.split("."), "20260101T000000Z"
        ).is_dir()

    def test_clean_up_without_output_type_raises_value_error(
        self, tmp_path: Path, steps: Steps
    ) -> None:
        with pytest.raises(ValueError, match="clean_up needs output_type"):
            run_epc_pipeline("scotland", "domestic", tmp_path, clean_up=True)
        assert steps.calls == []

    def test_output_inside_a_cleaned_folder_raises_value_error(
        self, tmp_path: Path, steps: Steps
    ) -> None:
        with pytest.raises(ValueError, match="which clean_up deletes from"):
            run_epc_pipeline(
                "scotland",
                "domestic",
                tmp_path,
                output_type="parquet",
                output_path=tmp_path / "datasets" / "homes.parquet",
                clean_up=True,
            )
        assert steps.calls == []
