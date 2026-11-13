"""Tests for processing a raw snapshot into typed Parquet."""

import codecs
import json
import warnings
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path

import polars as pl
import pytest

from py_epc import (
    EPCOrigin,
    EPCSegment,
    ProcessManifest,
    ProcessWarning,
    process,
    process_epc,
)
from py_epc._manifest import read_manifest
from py_epc._process._register import Register

# Key of the source the tests process.
_SOURCE_KEY = "epc.scotland.domestic"

# Header of each shipped register's files, as published.
_HEADERS = Path(__file__).parent / "headers"

# Header of the small register's files.
_HEADER = "LODGEMENT_DATE,TOTAL_FLOOR_AREA,CONSTITUENCY"

# A small register like Scotland's: a label row, and three columns.
_REGISTER = Register.model_validate_json(
    json.dumps(
        {
            "version": 1,
            "source_key": _SOURCE_KEY,
            "member_pattern": r"(\d{4})\.csv",
            "label_rows": 1,
            "year_column": "lodgement_date",
            "missing_values": ["", "N/A"],
            "columns": [
                {
                    "source": "LODGEMENT_DATE",
                    "name": "lodgement_date",
                    "type": "date",
                    "description": "d",
                    "formats": ["%Y-%m-%d"],
                },
                {
                    "source": "TOTAL_FLOOR_AREA",
                    "name": "total_floor_area",
                    "type": "number",
                    "description": "d",
                },
                {
                    "source": "CONSTITUENCY",
                    "name": "ward",
                    "type": "text",
                    "description": "d",
                },
            ],
        }
    )
)


def yearly_csv(rows: list[str], header: str = _HEADER) -> bytes:
    """Return a yearly file as Scotland publishes it: BOM, labels and CRLF."""
    labels = ",".join(f"label {name}" for name in header.split(","))
    lines = "\r\n".join([header, labels, *rows]) + "\r\n"
    return codecs.BOM_UTF8 + lines.encode()


@pytest.fixture(autouse=True)
def small_register(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Process with the small register, unless the test class says not to."""
    if not getattr(request.cls, "uses_shipped_register", False):
        monkeypatch.setattr(
            "py_epc._process._build.load_register", lambda key: _REGISTER
        )


def certificates(build: Path) -> pl.DataFrame:
    """Return every certificate in a build, with its lodgement year."""
    return (
        pl.scan_parquet(build / "certificates", hive_partitioning=True)
        .collect()
        .sort("lodgement_date")
    )


class TestProcess:
    """`process` turns a raw snapshot into typed Parquet, one file per year."""

    def test_snapshot_writes_one_parquet_file_per_year(
        self, tmp_path: Path, make_snapshot: Callable[..., Path]
    ) -> None:
        make_snapshot(
            _SOURCE_KEY,
            {
                "2025.csv": yearly_csv(["2025-03-26,72.5,S13002516 "]),
                "2026.csv": yearly_csv(["2026-01-02,80,"]),
                "Note.pdf": b"%PDF",
            },
        )
        build = process(_SOURCE_KEY, tmp_path)
        files = sorted(build.rglob("*.parquet"))
        assert [path.relative_to(build).as_posix() for path in files] == [
            "certificates/lodgement_year=2025/data.parquet",
            "certificates/lodgement_year=2026/data.parquet",
        ]

    def test_build_is_named_after_snapshot_under_processed(
        self, tmp_path: Path, make_snapshot: Callable[..., Path]
    ) -> None:
        raw = make_snapshot(_SOURCE_KEY, {"2025.csv": yearly_csv(["2025-03-26,1,"])})
        build = process(_SOURCE_KEY, tmp_path)
        processed = tmp_path / "processed" / "epc" / "scotland" / "domestic"
        assert build == processed / raw.name

    def test_values_are_parsed_and_columns_renamed(
        self, tmp_path: Path, make_snapshot: Callable[..., Path]
    ) -> None:
        make_snapshot(
            _SOURCE_KEY,
            {"2025.csv": yearly_csv(["2025-03-26,72.5,S13002516 ", "2025-04-01,N/A,"])},
        )
        frame = certificates(process(_SOURCE_KEY, tmp_path))
        assert frame.to_dicts() == [
            {
                "lodgement_date": date(2025, 3, 26),
                "total_floor_area": 72.5,
                "ward": "S13002516",
                "lodgement_year": 2025,
            },
            {
                "lodgement_date": date(2025, 4, 1),
                "total_floor_area": None,
                "ward": None,
                "lodgement_year": 2025,
            },
        ]

    def test_manifest_records_source_snapshot_and_rows(
        self, tmp_path: Path, make_snapshot: Callable[..., Path]
    ) -> None:
        raw = make_snapshot(
            _SOURCE_KEY, {"2025.csv": yearly_csv(["2025-03-26,1,", "2025-03-27,2,"])}
        )
        manifest = read_manifest(process(_SOURCE_KEY, tmp_path), ProcessManifest)
        assert (manifest.source_key, manifest.snapshot) == (_SOURCE_KEY, raw.name)
        assert [(year.year, year.rows) for year in manifest.years] == [(2025, 2)]

    def test_invalid_values_are_counted_with_examples_and_warned(
        self, tmp_path: Path, make_snapshot: Callable[..., Path]
    ) -> None:
        make_snapshot(
            _SOURCE_KEY,
            {
                "2025.csv": yearly_csv(
                    ["2025-03-26,big,", "2025-03-27,huge ,", "2025-03-28,big,"]
                )
            },
        )
        with pytest.warns(ProcessWarning, match=r"2025: total_floor_area \(3\)"):
            build = process(_SOURCE_KEY, tmp_path)
        (summary,) = read_manifest(build, ProcessManifest).years
        assert summary.invalid["total_floor_area"].count == 3
        assert summary.invalid["total_floor_area"].examples == ("big", "huge")
        assert certificates(build)["total_floor_area"].to_list() == [None] * 3

    def test_row_in_wrong_year_is_counted_and_warned(
        self, tmp_path: Path, make_snapshot: Callable[..., Path]
    ) -> None:
        make_snapshot(
            _SOURCE_KEY,
            {"2025.csv": yearly_csv(["2025-03-26,1,", "2024-12-31,1,"])},
        )
        with pytest.warns(ProcessWarning, match=r"year mismatches \(1\)"):
            build = process(_SOURCE_KEY, tmp_path)
        assert read_manifest(build, ProcessManifest).years[0].year_mismatches == 1

    def test_clean_snapshot_warns_nothing(
        self, tmp_path: Path, make_snapshot: Callable[..., Path]
    ) -> None:
        make_snapshot(_SOURCE_KEY, {"2025.csv": yearly_csv(["2025-03-26,1,"])})
        with warnings.catch_warnings():
            warnings.simplefilter("error", ProcessWarning)
            process(_SOURCE_KEY, tmp_path)

    def test_mismatched_header_raises_value_error_and_leaves_nothing(
        self, tmp_path: Path, make_snapshot: Callable[..., Path]
    ) -> None:
        make_snapshot(
            _SOURCE_KEY,
            {"2025.csv": yearly_csv(["2025-03-26,1"], header="LODGEMENT_DATE,NEW")},
        )
        with pytest.raises(ValueError, match="header does not match"):
            process(_SOURCE_KEY, tmp_path)
        builds = tmp_path / "processed" / "epc"
        assert not any(path.is_file() for path in builds.rglob("*"))

    def test_existing_build_raises_file_exists_error(
        self, tmp_path: Path, make_snapshot: Callable[..., Path]
    ) -> None:
        make_snapshot(_SOURCE_KEY, {"2025.csv": yearly_csv(["2025-03-26,1,"])})
        process(_SOURCE_KEY, tmp_path)
        with pytest.raises(FileExistsError, match="overwrite=True"):
            process(_SOURCE_KEY, tmp_path)

    def test_overwrite_replaces_existing_build(
        self, tmp_path: Path, make_snapshot: Callable[..., Path]
    ) -> None:
        make_snapshot(_SOURCE_KEY, {"2025.csv": yearly_csv(["2025-03-26,1,"])})
        build = process(_SOURCE_KEY, tmp_path)
        (build / "stale.txt").write_text("old", encoding="utf-8")
        assert process(_SOURCE_KEY, tmp_path, overwrite=True) == build
        assert not (build / "stale.txt").exists()
        assert [path.name for path in build.parent.iterdir()] == [build.name]

    def test_named_snapshot_is_processed(
        self, tmp_path: Path, make_snapshot: Callable[..., Path]
    ) -> None:
        older = make_snapshot(
            _SOURCE_KEY,
            {"2025.csv": yearly_csv(["2025-03-26,1,"])},
            stamp="20261011T093000Z",
        )
        make_snapshot(
            _SOURCE_KEY,
            {"2025.csv": yearly_csv(["2025-03-26,1,"])},
            stamp="20261111T093000Z",
        )
        assert process(_SOURCE_KEY, tmp_path, snapshot=older.name).name == older.name

    def test_first_build_makes_git_ignore_processed_folder(
        self, tmp_path: Path, make_snapshot: Callable[..., Path]
    ) -> None:
        make_snapshot(_SOURCE_KEY, {"2025.csv": yearly_csv(["2025-03-26,1,"])})
        process(_SOURCE_KEY, tmp_path)
        lines = (tmp_path / "processed" / ".gitignore").read_text(encoding="utf-8")
        assert "*" in lines.splitlines()

    def test_build_holds_only_certificates_and_manifest(
        self, tmp_path: Path, make_snapshot: Callable[..., Path]
    ) -> None:
        make_snapshot(_SOURCE_KEY, {"2025.csv": yearly_csv(["2025-03-26,1,"])})
        build = process(_SOURCE_KEY, tmp_path)
        assert sorted(path.name for path in build.iterdir()) == [
            "certificates",
            "manifest.json",
        ]


class TestProcessWithShippedRegister:
    """`process` reads a published header with the shipped definition."""

    # Tells the `small_register` fixture to leave the shipped one in place.
    uses_shipped_register = True

    def test_scotland_domestic_file_is_parsed_with_every_column(
        self, tmp_path: Path, make_snapshot: Callable[..., Path]
    ) -> None:
        header = (_HEADERS / "epc-scotland-domestic.csv").read_text().strip()
        names = header.split(",")
        row = dict.fromkeys(names, "")
        row |= {
            "LODGEMENT_DATE": "2025-03-26",
            "CREATED_AT": "26/03/2025 14:37:12",
            "OSG_REFERENCE_NUMBER": "906700000001",
            "HISTORIC": "False",
            "TOTAL_FLOOR_AREA": "72.50 ",
        }
        make_snapshot(
            _SOURCE_KEY,
            {"2025.csv": yearly_csv([",".join(row[name] for name in names)], header)},
        )
        frame = certificates(process(_SOURCE_KEY, tmp_path))
        assert frame.select(
            "lodgement_date", "uprn", "historic", "total_floor_area"
        ).to_dicts() == [
            {
                "lodgement_date": date(2025, 3, 26),
                "uprn": 906700000001,
                "historic": False,
                "total_floor_area": 72.5,
            }
        ]
        assert frame.width == len(names) + 2
        assert frame["is_latest"].to_list() == [True]

    def test_england_and_wales_domestic_zip_is_parsed_with_every_column(
        self, tmp_path: Path, make_snapshot: Callable[..., Path]
    ) -> None:
        key = "epc.england-and-wales.domestic"
        header = (_HEADERS / "epc-england-and-wales-domestic.csv").read_text().strip()
        names = header.split(",")
        row = dict.fromkeys(names, "")
        row |= {
            "lodgement_date": "2025-03-26",
            "lodgement_datetime": "2025-03-26 14:37:12",
            "uprn": "100030317077",
            "mains_gas_flag": "Y",
            "current_energy_rating": "INVALID!",
            "total_floor_area": "72.5",
        }
        lines = [header, ",".join(row[name] for name in names), ""]
        make_snapshot(
            key,
            {
                "certificates-2025.csv": "\n".join(lines),
                "recommendations-2025.csv": "certificate_number,improvement\n",
            },
            archive="zip",
        )
        frame = certificates(process(key, tmp_path))
        assert frame.select(
            "lodgement_datetime", "uprn", "mains_gas_flag", "current_energy_rating"
        ).to_dicts() == [
            {
                "lodgement_datetime": datetime(2025, 3, 26, 14, 37, 12),
                "uprn": 100030317077,
                "mains_gas_flag": True,
                "current_energy_rating": None,
            }
        ]
        assert frame.width == len(names) + 2
        assert frame["is_latest"].to_list() == [True]


class TestProcessEpc:
    """`process_epc` processes the EPC source for an origin and segment."""

    @pytest.mark.parametrize(
        ("origin", "segment"),
        [
            pytest.param("scotland", "domestic", id="strings"),
            pytest.param(EPCOrigin.SCOTLAND, EPCSegment.DOMESTIC, id="enum-members"),
        ],
    )
    def test_origin_and_segment_process_their_source(
        self,
        tmp_path: Path,
        make_snapshot: Callable[..., Path],
        origin: EPCOrigin | str,
        segment: EPCSegment | str,
    ) -> None:
        make_snapshot(_SOURCE_KEY, {"2025.csv": yearly_csv(["2025-03-26,1,"])})
        build = process_epc(origin, segment, tmp_path)
        assert read_manifest(build, ProcessManifest).source_key == _SOURCE_KEY

    def test_unknown_origin_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown EPCOrigin"):
            process_epc("scottland", "domestic", tmp_path)
