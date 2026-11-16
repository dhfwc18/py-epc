"""Tests for keeping the English of values published in English and Welsh."""

import polars as pl
import pytest

from py_epc._dataset._bilingual import english_text, keep_english_text


class TestEnglishText:
    """`english_text` keeps the English parts of a bilingual value."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param("Single glazed", "Single glazed", id="english-only"),
            pytest.param(
                "Single glazed|Gwydrau sengl", "Single glazed", id="english-and-welsh"
            ),
            pytest.param(
                "Boiler and radiators, |Bwyler a rheiddiaduron, |mains gas|nwy",
                "Boiler and radiators, mains gas",
                id="alternating-parts",
            ),
            pytest.param(" |Gwydrau sengl", None, id="no-english"),
            pytest.param(None, None, id="none"),
        ],
    )
    def test_value_returns_english(self, value: str | None, expected: str | None):
        frame = pl.DataFrame({"text": [value]}, schema={"text": pl.String})
        assert frame.select(english_text(pl.col("text"))).item() == expected


class TestKeepEnglishText:
    """`keep_english_text` puts columns in English, row by row."""

    def test_columns_listed_are_in_english_in_row_order(self) -> None:
        frame = pl.LazyFrame(
            {
                "windows_description": ["Single glazed|Gwydrau sengl", None, "a"],
                "other": ["x|y", "x|y", "x|y"],
            }
        )
        result = keep_english_text(frame, ["windows_description"]).collect()
        assert result.to_dict(as_series=False) == {
            "windows_description": ["Single glazed", None, "a"],
            "other": ["x|y", "x|y", "x|y"],
        }

    def test_odd_number_of_parts_raises_value_error_listing_it(self) -> None:
        frame = pl.LazyFrame({"walls_description": ["a|b|c", "a|b"]})
        with pytest.raises(
            ValueError, match=r"walls_description: 1 such as \['a\|b\|c'\]"
        ):
            keep_english_text(frame, ["walls_description"])

    def test_odd_number_of_parts_without_strict_gives_null_and_warns(self) -> None:
        frame = pl.LazyFrame({"walls_description": ["a|b|c", "a|b"]})
        with pytest.warns(UserWarning, match="odd number of parts.*given null"):
            result = keep_english_text(frame, ["walls_description"], strict=False)
        assert result.collect()["walls_description"].to_list() == [None, "a"]
