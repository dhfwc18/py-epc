"""Tests for building a dataset from a processed build."""

import json
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from py_epc import (
    DatasetManifest,
    EPCOrigin,
    EPCSegment,
    ProcessManifest,
    build_dataset,
    build_epc_dataset,
)
from py_epc._dataset import _build
from py_epc._dataset._definition import DatasetDefinition
from py_epc._dataset._parts import MisalignedParts
from py_epc._manifest import read_manifest, write_manifest
from py_epc._process._build import YearSummary

# Key of the source the tests build.
_SOURCE_KEY = "epc.scotland.domestic"

# Name of the processed build the tests write by default.
_SNAPSHOT = "20261111T093000Z"

# Lighting phrases, as published.
_HALF = "Low energy lighting in 50% of fixed outlets"
_ALL = "Low energy lighting in 100% of fixed outlets"

# A small definition, splitting and parsing lighting.
_DEFINITION = DatasetDefinition.model_validate_json(
    json.dumps(
        {
            "version": 1,
            "source_key": _SOURCE_KEY,
            "building_elements": [
                {
                    "name": "lighting",
                    "description": "lighting_description",
                    "energy_eff": "lighting_energy_eff",
                    "env_eff": "lighting_env_eff",
                }
            ],
            "parsers": [{"column": "lighting_description", "parser": "lighting"}],
        }
    )
)

# Type of the `make_build` fixture.
type MakeBuild = Callable[..., Path]


@pytest.fixture(autouse=True)
def small_definition(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build with the small definition."""
    monkeypatch.setattr(_build, "load_dataset_definition", lambda key: _DEFINITION)


@pytest.fixture
def make_build(tmp_path: Path) -> MakeBuild:
    """
    Return a function that writes a tiny processed build, as `process` would.

    The function takes the lighting of each year's certificates, as a
    mapping of year to descriptions, and writes them with their ratings and
    the build's manifest to ``<tmp_path>/processed/<key parts>/<stamp>/``.
    The build's name can be set with ``stamp=``.
    """

    def make(lighting: dict[int, list[str | None]], *, stamp: str = _SNAPSHOT) -> Path:
        build = tmp_path.joinpath("processed", *_SOURCE_KEY.split("."), stamp)
        for year, descriptions in lighting.items():
            folder = build / "certificates" / f"lodgement_year={year}"
            folder.mkdir(parents=True)
            pl.DataFrame(
                {
                    "lodgement_date": [date(year, 5, 1)] * len(descriptions),
                    "lighting_description": descriptions,
                    "lighting_energy_eff": ["Good"] * len(descriptions),
                    "lighting_env_eff": ["Good"] * len(descriptions),
                }
            ).write_parquet(folder / "data.parquet")
        write_manifest(
            build,
            ProcessManifest(
                source_key=_SOURCE_KEY,
                snapshot=stamp,
                raw_sha256="0" * 64,
                register_version=1,
                package_version="0",
                processed_at=datetime(2026, 11, 11, tzinfo=UTC),
                years=tuple(
                    YearSummary(
                        year=year, rows=len(rows), invalid={}, year_mismatches=0
                    )
                    for year, rows in lighting.items()
                ),
                latest=None,
            ),
        )
        return build

    return make


def year_file(dataset: Path, year: int) -> Path:
    """Return the file of one year of a dataset."""
    return dataset / "certificates" / f"lodgement_year={year}" / "data.parquet"


class TestBuildDataset:
    """`build_dataset` writes a processed build's years, prepared."""

    def test_each_year_is_written_prepared(
        self, tmp_path: Path, make_build: MakeBuild
    ) -> None:
        make_build({2014: [_HALF, None], 2015: [_ALL]})
        dataset = build_dataset(_SOURCE_KEY, tmp_path)
        assert dataset == tmp_path.joinpath(
            "datasets", *_SOURCE_KEY.split("."), _SNAPSHOT
        )
        written = {
            year: pl.read_parquet(year_file(dataset, year)) for year in (2014, 2015)
        }
        assert written[2014]["lighting_low_energy_percent"].to_list() == [50.0, None]
        assert written[2015]["lighting_description_kind"].to_list() == ["described"]

    def test_manifest_records_build_and_counts_of_each_year(
        self, tmp_path: Path, make_build: MakeBuild
    ) -> None:
        make_build({2015: [_ALL], 2014: [_HALF, "???", None]})
        manifest = read_manifest(build_dataset(_SOURCE_KEY, tmp_path), DatasetManifest)
        assert (manifest.source_key, manifest.snapshot, manifest.raw_sha256) == (
            _SOURCE_KEY,
            _SNAPSHOT,
            "0" * 64,
        )
        assert [(summary.year, summary.rows) for summary in manifest.years] == [
            (2014, 3),
            (2015, 1),
        ]
        counts = manifest.years[0].kinds["lighting_description"]
        assert (counts.described, counts.not_a_description, counts.missing) == (
            1,
            1,
            1,
        )
        assert manifest.years[0].misaligned_parts == {
            "lighting": MisalignedParts(different_counts=0)
        }

    def test_first_build_makes_git_ignore_datasets(
        self, tmp_path: Path, make_build: MakeBuild
    ) -> None:
        make_build({2014: [_HALF]})
        build_dataset(_SOURCE_KEY, tmp_path)
        assert (tmp_path / "datasets" / ".gitignore").is_file()

    def test_named_snapshot_builds_that_build(
        self, tmp_path: Path, make_build: MakeBuild
    ) -> None:
        make_build({2014: [_HALF]}, stamp="20260101T000000Z")
        make_build({2014: [_ALL]})
        dataset = build_dataset(_SOURCE_KEY, tmp_path, snapshot="20260101T000000Z")
        assert dataset.name == "20260101T000000Z"
        written = pl.read_parquet(year_file(dataset, 2014))
        assert written["lighting_low_energy_percent"].to_list() == [50.0]

    def test_no_processed_build_raises_file_not_found_error(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError):
            build_dataset(_SOURCE_KEY, tmp_path)

    def test_existing_dataset_raises_file_exists_error(
        self, tmp_path: Path, make_build: MakeBuild
    ) -> None:
        make_build({2014: [_HALF]})
        build_dataset(_SOURCE_KEY, tmp_path)
        with pytest.raises(FileExistsError, match="overwrite=True"):
            build_dataset(_SOURCE_KEY, tmp_path)

    def test_existing_dataset_with_overwrite_is_replaced(
        self, tmp_path: Path, make_build: MakeBuild
    ) -> None:
        build = make_build({2014: [_HALF]})
        build_dataset(_SOURCE_KEY, tmp_path)
        pl.DataFrame(
            {
                "lodgement_date": [date(2014, 5, 1)],
                "lighting_description": [_ALL],
                "lighting_energy_eff": ["Good"],
                "lighting_env_eff": ["Good"],
            }
        ).write_parquet(year_file(build, 2014))
        dataset = build_dataset(_SOURCE_KEY, tmp_path, overwrite=True)
        written = pl.read_parquet(year_file(dataset, 2014))
        assert written["lighting_low_energy_percent"].to_list() == [100.0]

    def test_unknown_phrase_raises_value_error_and_writes_nothing(
        self, tmp_path: Path, make_build: MakeBuild
    ) -> None:
        make_build({2014: [_HALF], 2015: ["Candles"]})
        with pytest.raises(ValueError, match="'Candles'"):
            build_dataset(_SOURCE_KEY, tmp_path)
        assert list((tmp_path / "datasets").rglob("*.parquet")) == []


class TestBuildEpcDataset:
    """`build_epc_dataset` builds the source of an origin and segment."""

    @pytest.mark.parametrize(
        "names",
        [
            pytest.param(("scotland", "domestic"), id="strings"),
            pytest.param((EPCOrigin.SCOTLAND, EPCSegment.DOMESTIC), id="members"),
        ],
    )
    def test_origin_and_segment_build_their_source(
        self, tmp_path: Path, make_build: MakeBuild, names: tuple[Any, Any]
    ) -> None:
        make_build({2014: [_HALF]})
        dataset = build_epc_dataset(*names, tmp_path)
        assert dataset == tmp_path.joinpath(
            "datasets", *_SOURCE_KEY.split("."), _SNAPSHOT
        )

    def test_unknown_origin_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            build_epc_dataset("mars", "domestic", tmp_path)
