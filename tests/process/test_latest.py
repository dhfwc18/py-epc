"""Tests for marking each property's latest certificate."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from py_epc._process._latest import mark_latest, property_key
from py_epc._process._register import LatestRule

# A rule like Scotland's: UPRN first, then the register's own property id.
_RULE = LatestRule.model_validate_json(
    json.dumps(
        {
            "property_columns": ["uprn", "building_reference_number"],
            "order_columns": ["lodgement_datetime", "certificate_number"],
            "certificate_column": "certificate_number",
            "superseded_column": "historic",
        }
    )
)

# Schema of the certificates in the tests.
_SCHEMA = {
    "certificate_number": pl.String,
    "uprn": pl.Int64,
    "building_reference_number": pl.String,
    "lodgement_datetime": pl.Datetime,
    "historic": pl.Boolean,
}


def certificate(number: str | None, lodged: str, **fields: Any) -> dict[str, Any]:
    """Return one certificate, with no property unless given."""
    return {
        "certificate_number": number,
        "uprn": None,
        "building_reference_number": None,
        "lodgement_datetime": datetime.fromisoformat(lodged),
        "historic": None,
        **fields,
    }


def write_build(certificates: Path, years: dict[int, list[dict[str, Any]]]) -> None:
    """Write each year's certificates as a build's yearly file."""
    for year, rows in years.items():
        folder = certificates / f"lodgement_year={year}"
        folder.mkdir(parents=True)
        pl.DataFrame(rows, schema=_SCHEMA).write_parquet(folder / "data.parquet")


def latest(certificates: Path) -> dict[str | None, bool]:
    """Return `is_latest` for each certificate number in a build."""
    frame = pl.read_parquet(certificates / "*" / "data.parquet")
    return dict(zip(frame["certificate_number"], frame["is_latest"], strict=True))


class TestPropertyKey:
    """`property_key` names a certificate's property by its first column."""

    @pytest.mark.parametrize(
        ("uprn", "reference", "expected"),
        [
            pytest.param(7, "B1", "uprn:7", id="uprn-first"),
            pytest.param(None, "B1", "building_reference_number:B1", id="fallback"),
            pytest.param(None, None, None, id="none"),
        ],
    )
    def test_first_column_with_value_names_property(
        self, uprn: int | None, reference: str | None, expected: str | None
    ) -> None:
        frame = pl.DataFrame(
            {"uprn": [uprn], "building_reference_number": [reference]},
            schema={"uprn": pl.Int64, "building_reference_number": pl.String},
        )
        assert frame.select(property_key(_RULE)).item() == expected


class TestMarkLatest:
    """`mark_latest` flags each property's latest certificate."""

    def test_newer_certificate_in_later_year_is_latest(self, tmp_path: Path) -> None:
        write_build(
            tmp_path,
            {
                2020: [certificate("old", "2020-05-01T10:00", uprn=1)],
                2024: [certificate("new", "2024-05-01T10:00", uprn=1)],
            },
        )
        mark_latest(tmp_path, _RULE)
        assert latest(tmp_path) == {"old": False, "new": True}

    def test_same_time_is_decided_by_certificate_number(self, tmp_path: Path) -> None:
        write_build(
            tmp_path,
            {
                2024: [
                    certificate("0002", "2024-05-01T10:00", uprn=1),
                    certificate("0001", "2024-05-01T10:00", uprn=1),
                ]
            },
        )
        mark_latest(tmp_path, _RULE)
        assert latest(tmp_path) == {"0002": True, "0001": False}

    def test_fallback_column_groups_certificates_without_uprn(
        self, tmp_path: Path
    ) -> None:
        write_build(
            tmp_path,
            {
                2024: [
                    certificate("a", "2024-01-01T10:00", building_reference_number="7"),
                    certificate("b", "2024-02-01T10:00", building_reference_number="7"),
                    certificate("c", "2024-01-01T10:00", uprn=7),
                ]
            },
        )
        mark_latest(tmp_path, _RULE)
        assert latest(tmp_path) == {"a": False, "b": True, "c": True}

    def test_certificate_without_property_is_its_own_latest(
        self, tmp_path: Path
    ) -> None:
        write_build(
            tmp_path,
            {
                2024: [
                    certificate("a", "2024-01-01T10:00"),
                    certificate("b", "2024-02-01T10:00"),
                ]
            },
        )
        mark_latest(tmp_path, _RULE)
        assert latest(tmp_path) == {"a": True, "b": True}

    def test_flag_is_added_last_and_rows_keep_their_order(self, tmp_path: Path) -> None:
        rows = [
            certificate(f"{n:04}", f"2024-01-{n:02}T10:00", uprn=n % 3)
            for n in range(1, 21)
        ]
        write_build(tmp_path, {2024: rows})
        mark_latest(tmp_path, _RULE)
        frame = pl.read_parquet(tmp_path / "lodgement_year=2024" / "data.parquet")
        assert frame.columns == [*_SCHEMA, "is_latest"]
        assert frame["certificate_number"].to_list() == [
            row["certificate_number"] for row in rows
        ]

    def test_build_holds_no_working_files_afterwards(self, tmp_path: Path) -> None:
        write_build(tmp_path, {2024: [certificate("a", "2024-01-01T10:00")]})
        mark_latest(tmp_path, _RULE)
        assert [path.name for path in tmp_path.rglob("*") if path.is_file()] == [
            "data.parquet"
        ]


class TestMarkLatestSummary:
    """`mark_latest` counts properties, gaps and disagreements."""

    def test_summary_counts_properties_and_certificate_numbers(
        self, tmp_path: Path
    ) -> None:
        write_build(
            tmp_path,
            {
                2023: [
                    certificate("a", "2023-01-01T10:00", uprn=1),
                    certificate("a", "2023-02-01T10:00", uprn=2),
                ],
                2024: [
                    certificate("b", "2024-01-01T10:00", uprn=1),
                    certificate(None, "2024-02-01T10:00"),
                ],
            },
        )
        summary = mark_latest(tmp_path, _RULE)
        assert (
            summary.properties,
            summary.without_property,
            summary.blank_certificate_numbers,
            summary.duplicate_certificate_numbers,
        ) == (3, 1, 1, 1)

    def test_disagreement_with_publisher_flag_is_counted(self, tmp_path: Path) -> None:
        write_build(
            tmp_path,
            {
                2024: [
                    certificate("a", "2024-01-01T10:00", uprn=1, historic=True),
                    certificate("b", "2024-02-01T10:00", uprn=1, historic=False),
                    certificate("c", "2024-01-01T10:00", uprn=2, historic=True),
                ]
            },
        )
        assert mark_latest(tmp_path, _RULE).superseded_disagreements == 1

    def test_rule_without_publisher_flag_counts_no_disagreements(
        self, tmp_path: Path
    ) -> None:
        rule = _RULE.model_copy(update={"superseded_column": None})
        write_build(tmp_path, {2024: [certificate("a", "2024-01-01T10:00")]})
        assert mark_latest(tmp_path, rule).superseded_disagreements is None
