"""Tests for summarising one prepared year of a dataset."""

import datetime as dt
import json
from typing import get_args

import polars as pl

from py_epc._dataset._definition import DatasetDefinition
from py_epc._dataset._parts import MisalignedParts
from py_epc._dataset._phrases import PHRASE_KIND_DTYPE, PhraseKind
from py_epc._dataset._prepare import prepare_certificates
from py_epc._dataset._summary import (
    PhraseKindCounts,
    count_kinds,
    summarise_year,
)

# A definition parsing lighting and construction age.
_DEFINITION = DatasetDefinition.model_validate_json(
    json.dumps(
        {
            "version": 1,
            "source_key": "epc.scotland.domestic",
            "parsers": [
                {"column": "lighting_description", "parser": "lighting"},
                {"column": "construction_age_band", "parser": "construction_age"},
            ],
        }
    )
)


class TestPhraseKindCounts:
    """`PhraseKindCounts` counts every kind of phrase, and missing ones."""

    def test_fields_are_each_phrase_kind_and_missing(self) -> None:
        assert list(PhraseKindCounts.model_fields) == [
            *get_args(PhraseKind.__value__),
            "missing",
        ]


class TestSummariseYear:
    """`summarise_year` summarises the counts of `count_kinds`."""

    def test_prepared_year_gives_rows_by_kind_of_each_column(self) -> None:
        frame = pl.LazyFrame(
            {
                "lighting_description": [
                    "Low energy lighting in 50% of fixed outlets",
                    "Low energy lighting in 150% of fixed outlets",
                    "???",
                    None,
                ],
                "construction_age_band": ["1950-1964"] * 4,
                "lodgement_date": [dt.date(2014, 5, 1)] * 4,
            }
        )
        counts = count_kinds(prepare_certificates(frame, _DEFINITION), _DEFINITION)
        summary = summarise_year(2014, counts.collect(), {})
        assert summary.rows == 4
        assert summary.kinds == {
            "lighting_description": PhraseKindCounts(
                described=1,
                not_a_description=1,
                misplaced=0,
                invalid=1,
                unknown=0,
                missing=1,
            ),
            "construction_age_band": PhraseKindCounts(
                described=4,
                not_a_description=0,
                misplaced=0,
                invalid=0,
                unknown=0,
                missing=0,
            ),
        }

    def test_misaligned_parts_are_kept_by_element(self) -> None:
        frame = pl.LazyFrame(
            {
                "lighting_description_kind": pl.Series([None], dtype=PHRASE_KIND_DTYPE),
                "construction_age_band_kind": pl.Series(
                    ["described"], dtype=PHRASE_KIND_DTYPE
                ),
            }
        )
        misaligned = MisalignedParts(different_counts=2)
        summary = summarise_year(
            2014, count_kinds(frame, _DEFINITION).collect(), {"walls": misaligned}
        )
        assert summary.misaligned_parts == {"walls": misaligned}
