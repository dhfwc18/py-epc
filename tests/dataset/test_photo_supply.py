"""Tests for parsing Scotland's photovoltaic arrays into totals."""

import polars as pl
import pytest

from py_epc._dataset._photo_supply import (
    ParsedPhotoSupply,
    parse_photo_supply,
    photo_supply_parser,
)
from py_epc._dataset._phrases import parse_column

# Connections as published.
_CONNECTED = "Connection: connected to dwelling's electricity meter;"
_NOT_CONNECTED = "Connection: not connected to dwelling's electricity meter;"

# An array given by its peak power, as published.
_PEAK_ARRAY = (
    "Array: Peak Power: {kw}; Orientation: South; Pitch: 30 degrees;"
    " Overshading: none or very little; {connection} |"
)


def peak_array(kw: float, connection: str = _CONNECTED) -> str:
    """Return an array of peak power `kw`, with its connection."""
    return _PEAK_ARRAY.format(kw=kw, connection=connection)


class TestParsePhotoSupply:
    """`parse_photo_supply` gives the totals of a dwelling's arrays."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            pytest.param(
                f"Array: Roof Area: 40%; {_CONNECTED} |",
                ParsedPhotoSupply("described", 40.0, None, 1, "connected"),
                id="roof-area",
            ),
            pytest.param(
                "Array: Roof Area: 0%; Connection: not applicable (FGHRS or no PV); |",
                ParsedPhotoSupply("described", 0.0, None, 0),
                id="no-panels",
            ),
            pytest.param(
                "Array: Roof Area: 0%; |",
                ParsedPhotoSupply("described", 0.0, None, 0),
                id="no-panels-without-connection",
            ),
            pytest.param(
                "Array: Roof Area: 0%; Connection: not recorded; |",
                ParsedPhotoSupply("described", 0.0, None, 0),
                id="no-panels-connection-not-read",
            ),
            pytest.param(
                "Array: Supply: 0; |",
                ParsedPhotoSupply("described", 0.0, None, 0),
                id="older-supply",
            ),
            pytest.param(
                "Array: Supply: 25; |",
                ParsedPhotoSupply("described", 25.0, None, 1),
                id="older-supply-as-roof-area",
            ),
            pytest.param(
                peak_array(2.7),
                ParsedPhotoSupply("described", None, 2.7, 1, "connected"),
                id="peak-power",
            ),
            pytest.param(
                peak_array(1.5) + " " + peak_array(2.5),
                ParsedPhotoSupply("described", None, 4.0, 2, "connected"),
                id="peak-powers-summed",
            ),
            pytest.param(
                peak_array(1.5) + " " + peak_array(2.5, _NOT_CONNECTED),
                ParsedPhotoSupply("described", None, 4.0, 2, "partly_connected"),
                id="partly-connected",
            ),
            pytest.param(
                "Array: Peak Power: 2; Orientation: South; Pitch: horizontal;"
                " Overshading: modest; Connection: not recorded; |",
                ParsedPhotoSupply("described", None, 2.0, 1, "not_recorded"),
                id="not-recorded",
            ),
            pytest.param(
                "Array: Roof Area: %; |",
                ParsedPhotoSupply("not_a_description"),
                id="empty-share",
            ),
            pytest.param(
                "Array: Roof Area: 140%; |",
                ParsedPhotoSupply("invalid", 140.0, None, 1),
                id="share-above-100",
            ),
            pytest.param(None, ParsedPhotoSupply("not_a_description"), id="none"),
        ],
    )
    def test_phrase_returns_totals(
        self, text: str | None, expected: ParsedPhotoSupply
    ) -> None:
        assert parse_photo_supply(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("Array: Tilt: 30 degrees; |", id="unknown-field"),
            pytest.param("Array: Roof Area: some; |", id="unknown-share"),
            pytest.param("Array: Supply: 2.5 kW; |", id="unknown-number"),
            pytest.param(
                "Array: Roof Area: 40%; Connection: by cable; |",
                id="unknown-connection",
            ),
            pytest.param(
                "Array: Roof Area: 40%; Peak Power: 2; |", id="share-and-power"
            ),
            pytest.param("Solar panels", id="no-fields"),
        ],
    )
    def test_array_of_no_known_form_is_unknown(self, text: str) -> None:
        assert parse_photo_supply(text).kind == "unknown"


class TestPhotoSupplyParser:
    """`photo_supply_parser` gives the totals through `parse_column`."""

    def test_totals_are_added_for_each_row(self) -> None:
        frame = pl.LazyFrame(
            {
                "photo_supply_description": [
                    f"Array: Roof Area: 40%; {_CONNECTED} |",
                    peak_array(1.5) + " " + peak_array(2.5),
                    "Array: Roof Area: %; |",
                    None,
                ]
            }
        )
        result = parse_column(frame, photo_supply_parser()).collect()
        assert result.schema == pl.Schema(
            {
                "photo_supply_description": pl.String(),
                "photo_supply_roof_area_percent": pl.Float64(),
                "photo_supply_peak_kw": pl.Float64(),
                "photo_supply_array_count": pl.Int64(),
                "photo_supply_connection": pl.Enum(
                    ["connected", "partly_connected", "not_connected", "not_recorded"]
                ),
            }
        )
        assert result.drop("photo_supply_description").rows() == [
            (40.0, None, 1, "connected"),
            (None, 4.0, 2, "connected"),
            (None, None, None, None),
            (None, None, None, None),
        ]

    def test_unknown_phrase_raises_value_error_listing_it(self) -> None:
        frame = pl.LazyFrame({"photo_supply_description": ["Solar panels"]})
        with pytest.raises(ValueError, match=r"1 unknown phrases.*'Solar panels'"):
            parse_column(frame, photo_supply_parser())
