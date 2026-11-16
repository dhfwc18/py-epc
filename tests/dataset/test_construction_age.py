"""Tests for parsing the years a dwelling was built."""

from datetime import date

import polars as pl
import pytest

from py_epc._dataset._construction_age import (
    ParsedAgeBand,
    construction_age_parser,
    null_construction_age_after_lodgement,
    parse_age_band,
)
from py_epc._dataset._phrases import parse_column

# A lodgement date for rows whose date does not matter.
_LODGED: date = date(2020, 6, 1)


def construction_age(bands: list[str | None], lodged: date = _LODGED) -> pl.DataFrame:
    """Return the years of each band, lodged on `lodged`, after the check."""
    frame = pl.LazyFrame(
        {"construction_age_band": bands, "lodgement_date": [lodged] * len(bands)}
    )
    frame = parse_column(frame, construction_age_parser())
    return null_construction_age_after_lodgement(frame).collect()


class TestParseAgeBand:
    """`parse_age_band` reads the years an age band covers."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            pytest.param(
                "England and Wales: 1900-1929",
                ParsedAgeBand("described", 1900, 1929),
                id="register-prefix",
            ),
            pytest.param(
                "1950-1964", ParsedAgeBand("described", 1950, 1964), id="range"
            ),
            pytest.param(
                "England and Wales: before 1900",
                ParsedAgeBand("described", None, 1899),
                id="before",
            ),
            pytest.param("Pre-1900", ParsedAgeBand("described", None, 1899), id="pre"),
            pytest.param(
                "2008 onwards", ParsedAgeBand("described", 2008, None), id="onwards"
            ),
            pytest.param(
                "Post-2006", ParsedAgeBand("described", 2007, None), id="post"
            ),
            pytest.param(
                "2015", ParsedAgeBand("described", 2015, 2015), id="exact-year"
            ),
            pytest.param("1276", ParsedAgeBand("described", 1276, 1276), id="old-year"),
            pytest.param(
                "1012", ParsedAgeBand("invalid", 1012, 1012), id="year-too-early"
            ),
            pytest.param("0", ParsedAgeBand("invalid", 0, 0), id="zero"),
            pytest.param(
                "20013", ParsedAgeBand("invalid", 20013, 20013), id="five-digits"
            ),
            pytest.param(
                "1975-1967",
                ParsedAgeBand("invalid", 1975, 1967),
                id="reversed-range",
            ),
            pytest.param(
                "Not applicable",
                ParsedAgeBand("not_a_description"),
                id="not-applicable",
            ),
            pytest.param(None, ParsedAgeBand("not_a_description"), id="none"),
            pytest.param("Victorian", ParsedAgeBand("unknown"), id="unknown"),
        ],
    )
    def test_phrase_returns_parsed_age_band(
        self, text: str | None, expected: ParsedAgeBand
    ) -> None:
        assert parse_age_band(text) == expected


class TestConstructionAgeParser:
    """`construction_age_parser` gives the years through `parse_column`."""

    def test_years_are_added_for_each_row(self) -> None:
        result = construction_age(["before 1919", "2015", None, "0"])
        assert result.schema == pl.Schema(
            {
                "construction_age_band": pl.String(),
                "lodgement_date": pl.Date(),
                "construction_age_from": pl.Int64(),
                "construction_age_to": pl.Int64(),
            }
        )
        assert result.drop("construction_age_band", "lodgement_date").rows() == [
            (None, 1918),
            (2015, 2015),
            (None, None),
            (None, None),
        ]

    def test_unknown_band_raises_value_error_listing_it(self) -> None:
        with pytest.raises(ValueError, match=r"1 unknown phrases.*'Victorian'"):
            construction_age(["Victorian"])


class TestNullConstructionAgeAfterLodgement:
    """`null_construction_age_after_lodgement` drops impossible years."""

    @pytest.mark.parametrize(
        ("band", "expected"),
        [
            pytest.param("2021", (2021, 2021), id="year-after-lodgement"),
            pytest.param("2022", (None, None), id="two-years-after-lodgement"),
            pytest.param("2022 onwards", (None, None), id="band-after-lodgement"),
            pytest.param("before 2030", (None, 2029), id="open-start"),
        ],
    )
    def test_band_starting_after_lodgement_gives_null(
        self, band: str, expected: tuple[int | None, int | None]
    ) -> None:
        result = construction_age([band])
        assert result.row(0)[2:] == expected
