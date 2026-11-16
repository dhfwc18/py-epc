"""Tests for converting a column one distinct value at a time."""

import polars as pl
import pytest

from py_epc._dataset._distinct import join_distinct, map_distinct


def upper(values: list[str]) -> list[str | None]:
    """Return each value in capitals, with "skip" as None."""
    return [None if value == "skip" else value.upper() for value in values]


def lengths(values: list[str]) -> pl.DataFrame:
    """Return the length of each value, as a frame of one column."""
    return pl.DataFrame({"length": [len(value) for value in values]})


class TestMapDistinct:
    """`map_distinct` converts each distinct value of each column once."""

    def test_values_are_converted_in_row_order(self) -> None:
        frame = pl.LazyFrame({"word": ["b", "a", "b", None], "row": [1, 2, 3, 4]})
        result = map_distinct(frame, {"word": upper}).collect()
        assert result.to_dict(as_series=False) == {
            "word": ["B", "A", "B", None],
            "row": [1, 2, 3, 4],
        }

    def test_column_keeps_its_place(self) -> None:
        frame = pl.LazyFrame({"row": [1], "word": ["a"], "other": [2]})
        assert map_distinct(frame, {"word": upper}).collect().columns == [
            "row",
            "word",
            "other",
        ]

    def test_several_columns_are_each_converted_by_their_own(self) -> None:
        frame = pl.LazyFrame({"word": ["a", "b"], "name": ["c", "d"]})

        def first(values: list[str]) -> list[str | None]:
            return [f"first {value}" for value in values]

        result = map_distinct(frame, {"word": upper, "name": first}).collect()
        assert result.to_dict(as_series=False) == {
            "word": ["A", "B"],
            "name": ["first c", "first d"],
        }

    def test_unchanged_values_are_kept_and_never_converted(self) -> None:
        seen: list[list[str]] = []

        def record(values: list[str]) -> list[str | None]:
            seen.append(sorted(values))
            return upper(values)

        frame = pl.LazyFrame({"word": ["a", "kept", "skip", None]})
        result = map_distinct(
            frame,
            {"word": record},
            is_unchanged=lambda value: value.str.starts_with("k"),
        ).collect()
        assert result["word"].to_list() == ["A", "kept", None, None]
        assert seen == [["a", "skip"]]

    def test_no_converters_leaves_frame_unchanged(self) -> None:
        frame = pl.LazyFrame({"word": ["a"]})
        assert map_distinct(frame, {}).collect().equals(frame.collect())

    def test_converter_returning_none_gives_null(self) -> None:
        frame = pl.LazyFrame({"word": ["skip", "a"]})
        result = map_distinct(frame, {"word": upper}).collect()
        assert result["word"].to_list() == [None, "A"]

    def test_each_distinct_value_is_converted_once(self) -> None:
        seen: list[list[str]] = []

        def record(values: list[str]) -> list[str | None]:
            seen.append(sorted(values))
            return list(values)

        frame = pl.LazyFrame({"word": ["a", "b", "a", "a", None]})
        map_distinct(frame, {"word": record}).collect()
        assert seen == [["a", "b"]]

    def test_converter_of_wrong_length_raises_value_error(self) -> None:
        frame = pl.LazyFrame({"word": ["a", "b"]})
        with pytest.raises(
            ValueError, match="converter of 'word' returned 1 values for 2"
        ):
            map_distinct(frame, {"word": lambda values: values[:1]})


class TestJoinDistinct:
    """`join_distinct` builds columns from each distinct value once."""

    def test_columns_are_added_in_row_order(self) -> None:
        frame = pl.LazyFrame({"word": ["ccc", "a", "ccc"], "row": [1, 2, 3]})
        result = join_distinct(frame, {"word": lengths}).collect()
        assert result.to_dict(as_series=False) == {
            "word": ["ccc", "a", "ccc"],
            "row": [1, 2, 3],
            "length": [3, 1, 3],
        }

    def test_several_columns_add_their_columns_in_order(self) -> None:
        frame = pl.LazyFrame({"word": ["a", "bb"], "name": ["ccc", None]})

        def initials(values: list[str]) -> pl.DataFrame:
            return pl.DataFrame({"initial": [value[0] for value in values]})

        result = join_distinct(frame, {"word": lengths, "name": initials}).collect()
        assert result.to_dict(as_series=False) == {
            "word": ["a", "bb"],
            "name": ["ccc", None],
            "length": [1, 2],
            "initial": ["c", None],
        }

    def test_null_value_gives_null_in_added_columns(self) -> None:
        frame = pl.LazyFrame({"word": [None, "ab"]}, schema={"word": pl.String})
        result = join_distinct(frame, {"word": lengths}).collect()
        assert result["length"].to_list() == [None, 2]

    def test_builder_of_wrong_height_raises_value_error(self) -> None:
        frame = pl.LazyFrame({"word": ["a", "b"]})
        with pytest.raises(ValueError, match="builder of 'word' returned 1 rows for 2"):
            join_distinct(frame, {"word": lambda values: lengths(values[:1])})

    def test_builder_returning_existing_column_raises_value_error(self) -> None:
        frame = pl.LazyFrame({"word": ["a"], "length": [9]})
        with pytest.raises(ValueError, match=r"columns the frame has: \['length'\]"):
            join_distinct(frame, {"word": lengths})

    def test_builders_returning_same_column_raise_value_error(self) -> None:
        frame = pl.LazyFrame({"word": ["a"], "name": ["b"]})
        with pytest.raises(ValueError, match=r"builder of 'name' returned columns"):
            join_distinct(frame, {"word": lengths, "name": lengths})
