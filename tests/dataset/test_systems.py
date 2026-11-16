"""Tests for parsing the phrases of a dwelling's systems."""

import json
from collections.abc import Mapping
from typing import Any

import polars as pl
import pytest
from pydantic import ValidationError

from py_epc._dataset._phrases import parse_column
from py_epc._dataset._systems import (
    SystemVocabulary,
    parse_systems,
    system_parser,
)
from py_epc._dataset._vocabulary import load_vocabulary

# A small main heating vocabulary, laid out as the shipped one is.
_MAINHEAT_VOCABULARY: dict[str, Any] = {
    "version": 1,
    "format": "system",
    "column": "mainheat_description",
    "prefix": "mainheat",
    "misplaced": ["Single glazed"],
    "not_a_description": ["SAP05:Main-Heating", "with fan-assisted flue"],
    "fields": ["system", "emitter", "emitter_detail", "fuel"],
    "max_systems": 2,
    "labels": {
        "system": {
            "boiler": "Boiler",
            "heat_pump": "Heat pump",
            "storage_heaters": "Storage heaters",
            "warm_air": "Warm air",
            "none": "No system",
        },
        "emitter": {"radiators": "Radiators", "warm_air": "Warm air"},
        "emitter_detail": {"screed": "Pipes in screed"},
        "fuel": {"mains_gas": "Mains gas", "electricity": "Electricity"},
    },
    "parts": [
        {
            "phrases": ["Boiler and radiators", "Bwyler a rheiddiaduron"],
            "values": {"system": "boiler", "emitter": "radiators"},
        },
        {"phrases": ["Heat pump"], "values": {"system": "heat_pump"}},
        {
            "phrases": ["Electric storage heaters"],
            "values": {"system": "storage_heaters", "fuel": "electricity"},
        },
        {
            "phrases": ["Warm air"],
            "values": {"system": "warm_air"},
            "otherwise": {"emitter": "warm_air"},
        },
        {
            "phrases": ["No system present: electric heaters assumed"],
            "values": {"system": "none", "fuel": "electricity"},
            "assumed": True,
        },
        {"phrases": ["pipes in screed"], "values": {"emitter_detail": "screed"}},
        {
            "phrases": ["mains gas", "nwy prif gyflenwad"],
            "values": {"fuel": "mains_gas"},
        },
        {"phrases": ["electric"], "values": {"fuel": "electricity"}},
    ],
}


def vocabulary(**changes: Any) -> SystemVocabulary:
    """Return the small main heating vocabulary, with fields changed."""
    return SystemVocabulary.model_validate_json(
        json.dumps({**_MAINHEAT_VOCABULARY, **changes})
    )


def systems(text: str | None) -> tuple[Mapping[str, str], ...]:
    """Return the systems the small vocabulary reads in `text`."""
    return parse_systems(text, vocabulary()).systems


class TestSystemVocabulary:
    """`SystemVocabulary` checks a vocabulary when it is made."""

    @pytest.mark.parametrize("element", ["mainheat", "secondheat", "hotwater"])
    def test_shipped_vocabulary_is_valid(self, element: str) -> None:
        vocabulary = load_vocabulary(element, SystemVocabulary)
        assert (vocabulary.column, vocabulary.prefix) == (
            f"{element}_description",
            element,
        )

    def test_field_without_labels_raises_validation_error(self) -> None:
        labels = {**_MAINHEAT_VOCABULARY["labels"]}
        del labels["emitter_detail"]
        with pytest.raises(ValidationError, match="not every field"):
            vocabulary(labels=labels)

    def test_value_without_label_raises_validation_error(self) -> None:
        parts = [{"phrases": ["Oil"], "values": {"fuel": "oil"}}]
        with pytest.raises(ValidationError, match="'oil' has no label"):
            vocabulary(parts=parts)

    def test_unknown_field_raises_validation_error(self) -> None:
        parts = [{"phrases": ["x"], "values": {"tariff": "off_peak"}}]
        with pytest.raises(ValidationError, match="sets 'tariff'"):
            vocabulary(parts=parts)

    def test_phrase_in_two_parts_raises_validation_error(self) -> None:
        parts = [
            {"phrases": ["electric"], "values": {"fuel": "electricity"}},
            {"phrases": ["ELECTRIC"], "values": {"fuel": "electricity"}},
        ]
        with pytest.raises(ValidationError, match="in two parts"):
            vocabulary(parts=parts)


class TestParseSystems:
    """`parse_systems` reads the systems a phrase describes."""

    def test_parts_fill_one_system(self) -> None:
        assert systems("Boiler and radiators, mains gas") == (
            {"system": "boiler", "emitter": "radiators", "fuel": "mains_gas"},
        )

    def test_welsh_parts_fill_one_system(self) -> None:
        assert systems("Bwyler a rheiddiaduron, nwy prif gyflenwad") == (
            {"system": "boiler", "emitter": "radiators", "fuel": "mains_gas"},
        )

    def test_system_named_again_starts_second_system(self) -> None:
        text = "Boiler and radiators, mains gas, Boiler and radiators, mains gas"
        assert len(systems(text)) == 2

    def test_changed_value_starts_second_system(self) -> None:
        assert systems("Boiler and radiators, mains gas, Electric storage heaters") == (
            {"system": "boiler", "emitter": "radiators", "fuel": "mains_gas"},
            {"system": "storage_heaters", "fuel": "electricity"},
        )

    def test_repeated_value_is_absorbed(self) -> None:
        assert systems("Electric storage heaters, electric") == (
            {"system": "storage_heaters", "fuel": "electricity"},
        )

    def test_otherwise_values_fit_when_system_is_set(self) -> None:
        assert systems("Heat pump, Warm air, electric") == (
            {"system": "heat_pump", "emitter": "warm_air", "fuel": "electricity"},
        )

    def test_part_without_otherwise_starts_the_system_first(self) -> None:
        assert systems("Warm air, mains gas") == (
            {"system": "warm_air", "fuel": "mains_gas"},
        )

    def test_emitter_detail_is_its_own_field(self) -> None:
        assert systems("Heat pump, pipes in screed, electric") == (
            {"system": "heat_pump", "emitter_detail": "screed", "fuel": "electricity"},
        )

    def test_assumed_part_sets_assumed(self) -> None:
        result = parse_systems(
            "No system present: electric heaters assumed", vocabulary()
        )
        assert result.assumed is True

    def test_empty_otherwise_adds_nothing_when_its_field_is_set(self) -> None:
        parts = [
            *_MAINHEAT_VOCABULARY["parts"],
            {"phrases": ["gas"], "values": {"fuel": "mains_gas"}, "otherwise": {}},
        ]
        result = parse_systems("Electric storage heaters, gas", vocabulary(parts=parts))
        assert result.systems == ({"system": "storage_heaters", "fuel": "electricity"},)

    def test_ignored_part_is_skipped(self) -> None:
        assert systems("Boiler and radiators, with fan-assisted flue, mains gas") == (
            {"system": "boiler", "emitter": "radiators", "fuel": "mains_gas"},
        )

    def test_lost_separator_is_read_as_comma(self) -> None:
        assert systems("Boiler and radiators??????????mains gas")[0]["fuel"] == (
            "mains_gas"
        )

    @pytest.mark.parametrize(
        ("text", "kind"),
        [
            pytest.param("SAP05:Main-Heating", "not_a_description", id="placeholder"),
            pytest.param("Single glazed", "misplaced", id="misplaced"),
            pytest.param(
                "Boiler and radiators, Single glazed", "misplaced", id="misplaced-part"
            ),
            pytest.param("Boiler and radiators, peat", "unknown", id="unknown-part"),
            pytest.param(
                "Heat pump, Electric storage heaters, Boiler and radiators",
                "unknown",
                id="more-systems-than-allowed",
            ),
            pytest.param("????", "not_a_description", id="question-marks"),
            pytest.param(None, "not_a_description", id="none"),
        ],
    )
    def test_phrase_returns_kind(self, text: str | None, kind: str) -> None:
        assert parse_systems(text, vocabulary()).kind == kind


class TestSystemParser:
    """`system_parser` gives the fields of up to two systems."""

    def test_columns_are_added_for_each_row(self) -> None:
        frame = pl.LazyFrame(
            {
                "mainheat_description": [
                    "Boiler and radiators, mains gas",
                    "Boiler and radiators, mains gas, Electric storage heaters",
                    None,
                ]
            }
        )
        result = parse_column(frame, system_parser(vocabulary())).collect()
        assert result.columns == [
            "mainheat_description",
            "mainheat_system",
            "mainheat_emitter",
            "mainheat_emitter_detail",
            "mainheat_fuel",
            "mainheat_second_system",
            "mainheat_second_emitter",
            "mainheat_second_emitter_detail",
            "mainheat_second_fuel",
            "mainheat_system_count",
            "mainheat_assumed",
        ]
        assert result.select(
            "mainheat_system",
            "mainheat_fuel",
            "mainheat_second_system",
            "mainheat_system_count",
        ).rows() == [
            ("boiler", "mains_gas", None, 1),
            ("boiler", "mains_gas", "storage_heaters", 2),
            (None, None, None, None),
        ]

    def test_field_types_are_enums_of_labelled_values(self) -> None:
        frame = pl.LazyFrame({"mainheat_description": ["Heat pump"]})
        schema = parse_column(frame, system_parser(vocabulary())).collect_schema()
        assert (
            schema["mainheat_fuel"],
            schema["mainheat_second_emitter"],
            schema["mainheat_system_count"],
            schema["mainheat_assumed"],
        ) == (
            pl.Enum(["mains_gas", "electricity"]),
            pl.Enum(["radiators", "warm_air"]),
            pl.Int64(),
            pl.Boolean(),
        )

    def test_unknown_phrase_raises_value_error(self) -> None:
        frame = pl.LazyFrame({"mainheat_description": ["Boiler and radiators, peat"]})
        with pytest.raises(ValueError, match="1 unknown phrases"):
            parse_column(frame, system_parser(vocabulary()))

    def test_one_system_vocabulary_adds_no_second_columns(self) -> None:
        frame = pl.LazyFrame({"mainheat_description": ["Electric storage heaters"]})
        result = parse_column(frame, system_parser(vocabulary(max_systems=1))).collect()
        assert result.columns == [
            "mainheat_description",
            "mainheat_system",
            "mainheat_emitter",
            "mainheat_emitter_detail",
            "mainheat_fuel",
            "mainheat_assumed",
        ]
