"""Tests for preparing one year of a processed build."""

import datetime as dt
import json
from typing import Any

import polars as pl
import pytest

from py_epc._dataset._definition import DatasetDefinition
from py_epc._dataset._prepare import prepare_certificates

# Lighting phrases, as published.
_HALF = "Low energy lighting in 50% of fixed outlets"
_ALL = "Low energy lighting in 100% of fixed outlets"
_WELSH_HALF = "Goleuadau ynni-isel mewn 50% o'r mannau gosod"

# Fields of a definition parsing lighting and construction age.
_DEFINITION: dict[str, Any] = {
    "version": 1,
    "source_key": "epc.scotland.domestic",
    "parsers": [
        {"column": "lighting_description", "parser": "lighting"},
        {"column": "construction_age_band", "parser": "construction_age"},
    ],
}


def definition(**fields: Any) -> DatasetDefinition:
    """Return the definition of `_DEFINITION` and `fields`, read from JSON."""
    return DatasetDefinition.model_validate_json(json.dumps({**_DEFINITION, **fields}))


def certificates(lighting: list[str | None], **columns: list[Any]) -> pl.LazyFrame:
    """Return rows of lighting, built in 1950-1964 and lodged in 2014."""
    rows = len(lighting)
    return pl.LazyFrame(
        {
            "lighting_description": lighting,
            "construction_age_band": ["1950-1964"] * rows,
            "lodgement_date": [dt.date(2014, 5, 1)] * rows,
            **columns,
        }
    )


# A building element of lighting, as Scotland publishes it.
_LIGHTING_ELEMENT = {
    "name": "lighting",
    "description": "lighting_description",
    "energy_eff": "lighting_energy_eff",
    "env_eff": "lighting_env_eff",
}


class TestPrepareCertificates:
    """`prepare_certificates` runs a definition's steps on one year."""

    def test_parsed_columns_are_added_with_their_kinds(self) -> None:
        frame = certificates([_HALF, "Candles", None])
        with pytest.warns(UserWarning, match="'Candles', given null"):
            prepared = prepare_certificates(frame, definition(), strict=False)
        result = prepared.collect()
        assert result.select(
            "lighting_low_energy_percent",
            "lighting_description_kind",
            "construction_age_from",
            "construction_age_band_kind",
        ).rows() == [
            (50.0, "described", 1950, "described"),
            (None, "unknown", 1950, "described"),
            (None, None, 1950, "described"),
        ]

    def test_every_text_column_is_cleaned(self) -> None:
        frame = certificates([f"  {_HALF}"], address1=["1  High   Street "])
        result = prepare_certificates(frame, definition()).collect()
        assert result.select("lighting_description", "address1").row(0) == (
            _HALF,
            "1 High Street",
        )

    def test_bilingual_column_is_parsed_in_english(self) -> None:
        frame = certificates([f"{_HALF}|{_WELSH_HALF}"])
        prepared = prepare_certificates(
            frame, definition(bilingual_columns=["lighting_description"])
        ).collect()
        assert prepared.select(
            "lighting_description", "lighting_low_energy_percent"
        ).row(0) == (_HALF, 50.0)

    def test_building_element_is_parsed_from_main_part_with_parts_cleaned(
        self,
    ) -> None:
        frame = certificates(
            [f"{_HALF} | {_ALL.replace(' ', '  ')}"],
            lighting_energy_eff=["Good | Very Good"],
            lighting_env_eff=["Good | Very Good"],
        )
        prepared = prepare_certificates(
            frame, definition(building_elements=[_LIGHTING_ELEMENT])
        ).collect()
        assert prepared.select(
            "lighting_low_energy_percent",
            "lighting_energy_eff",
            "lighting_description_parts",
            "lighting_part_count",
        ).row(0) == (50.0, "Good", [_HALF, _ALL], 2)

    def test_construction_age_after_lodgement_is_null(self) -> None:
        frame = certificates([_HALF]).with_columns(
            construction_age_band=pl.lit("2020 onwards")
        )
        result = prepare_certificates(frame, definition()).collect()
        assert result.select("construction_age_from", "construction_age_to").row(0) == (
            None,
            None,
        )

    def test_renamed_column_is_renamed_after_parsing(self) -> None:
        frame = certificates([_HALF], photo_supply=[25.0])
        prepared = prepare_certificates(
            frame,
            definition(
                renamed_columns={
                    "photo_supply": "photo_supply_roof_area_percent",
                    "lighting_description": "lighting_text",
                }
            ),
        ).collect()
        assert prepared.select(
            "photo_supply_roof_area_percent",
            "lighting_text",
            "lighting_low_energy_percent",
        ).row(0) == (25.0, _HALF, 50.0)
        assert "photo_supply" not in prepared.columns

    def test_rows_keep_their_order(self) -> None:
        frame = certificates([_ALL, _HALF, _ALL, None], row=[1, 2, 3, 4])
        result = prepare_certificates(frame, definition()).collect()
        assert result["row"].to_list() == [1, 2, 3, 4]
        assert result["lighting_low_energy_percent"].to_list() == [
            100.0,
            50.0,
            100.0,
            None,
        ]

    def test_unknown_phrase_raises_value_error(self) -> None:
        frame = certificates(["Candles"])
        with pytest.raises(ValueError, match=r"1 unknown phrases.*'Candles'"):
            prepare_certificates(frame, definition())
