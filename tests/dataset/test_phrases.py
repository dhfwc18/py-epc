"""Tests for parsing a column of phrases, each distinct phrase once."""

from dataclasses import dataclass

import polars as pl
import pytest

from py_epc._dataset._phrases import (
    PHRASE_KIND_DTYPE,
    ColumnParser,
    OutputColumn,
    ParsedPhrase,
    parse_column,
    prepare_phrase,
)


@dataclass(frozen=True)
class ParsedNumber(ParsedPhrase):
    """A number word, parsed."""

    value: int | None = None
    """The number."""


# Phrases described as a number, keyed to it.
_NUMBER_PHRASES: dict[str, int] = {"one": 1, "two": 2}

# Phrases of every other kind but unknown. The invalid one carries a value,
# which must never reach an output.
_UNDESCRIBED_PHRASES: dict[str, ParsedNumber] = {
    "none": ParsedNumber("not_a_description"),
    "window": ParsedNumber("misplaced"),
    "lots": ParsedNumber("invalid", 99),
}


def parse_number(text: str | None) -> ParsedNumber:
    """Return a number word as a number."""
    phrase = prepare_phrase(text)
    if phrase is None:
        return ParsedNumber("not_a_description")
    if phrase in _NUMBER_PHRASES:
        return ParsedNumber("described", _NUMBER_PHRASES[phrase])
    return _UNDESCRIBED_PHRASES.get(phrase, ParsedNumber("unknown"))


def number_parser(*, replaces_column: bool = False) -> ColumnParser[ParsedNumber]:
    """Return a parser of number words, adding or replacing the column."""
    return ColumnParser(
        column="count",
        parse_phrase=parse_number,
        outputs=(
            OutputColumn(
                "count" if replaces_column else "count_value",
                lambda n: n.value,
                pl.Int64(),
            ),
            OutputColumn(
                "count_is_even", lambda n: (n.value or 0) % 2 == 0, pl.Boolean()
            ),
        ),
        replaces_column=replaces_column,
    )


class TestPreparePhrase:
    """`prepare_phrase` readies a cleaned phrase for parsing."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            pytest.param("Cavity wall", "Cavity wall", id="plain"),
            pytest.param(
                "Description: Single glazing", "Single glazing", id="description-prefix"
            ),
            pytest.param(
                "description:Single glazing", "Single glazing", id="prefix-any-case"
            ),
            pytest.param(
                "Pitched??????????250 mm loft insulation",
                "Pitched, 250 mm loft insulation",
                id="lost-separator",
            ),
            pytest.param("Unknown ?? text", "Unknown ?? text", id="two-marks-kept"),
            pytest.param("Description:", None, id="prefix-only"),
            pytest.param("????????", None, id="question-marks"),
            pytest.param("", None, id="empty"),
            pytest.param(None, None, id="none"),
        ],
    )
    def test_text_returns_phrase(self, text: str | None, expected: str | None) -> None:
        assert prepare_phrase(text) == expected


class TestColumnParser:
    """`ColumnParser` checks its outputs when it is made."""

    def test_no_outputs_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="has no outputs"):
            ColumnParser(column="count", parse_phrase=parse_number, outputs=())

    def test_repeated_output_raises_value_error(self) -> None:
        output = OutputColumn[ParsedNumber]("value", lambda n: n.value, pl.Int64())
        with pytest.raises(ValueError, match="repeats an output"):
            ColumnParser(
                column="count", parse_phrase=parse_number, outputs=(output, output)
            )

    def test_replacement_named_otherwise_raises_value_error(self) -> None:
        output = OutputColumn[ParsedNumber]("value", lambda n: n.value, pl.Int64())
        with pytest.raises(ValueError, match="must name its first output so"):
            ColumnParser(
                column="count",
                parse_phrase=parse_number,
                outputs=(output,),
                replaces_column=True,
            )


class TestParseColumn:
    """`parse_column` adds the outputs of a parser for each row."""

    def test_outputs_are_added_for_each_row(self) -> None:
        frame = pl.LazyFrame(
            {"count": ["two", "one", None, "two"], "row": [1, 2, 3, 4]}
        )
        result = parse_column(frame, number_parser()).collect()
        assert result.to_dict(as_series=False) == {
            "count": ["two", "one", None, "two"],
            "row": [1, 2, 3, 4],
            "count_value": [2, 1, None, 2],
            "count_is_even": [True, False, None, True],
        }

    @pytest.mark.parametrize(
        "phrase",
        [
            pytest.param("none", id="not-a-description"),
            pytest.param("window", id="misplaced"),
            pytest.param("lots", id="invalid"),
        ],
    )
    def test_phrase_not_described_gives_null_in_every_output(self, phrase: str) -> None:
        frame = pl.LazyFrame({"count": [phrase]})
        result = parse_column(frame, number_parser()).collect()
        assert result.row(0) == (phrase, None, None)

    def test_kind_column_gives_each_rows_kind(self) -> None:
        frame = pl.LazyFrame({"count": ["two", "none", "window", "lots", None]})
        result = parse_column(frame, number_parser(), kind_column="count_kind")
        assert result.collect_schema()["count_kind"] == PHRASE_KIND_DTYPE
        assert result.collect()["count_kind"].to_list() == [
            "described",
            "not_a_description",
            "misplaced",
            "invalid",
            None,
        ]

    def test_kind_column_follows_outputs_when_replacing(self) -> None:
        frame = pl.LazyFrame({"count": ["one"], "row": [1]})
        parser = number_parser(replaces_column=True)
        result = parse_column(frame, parser, kind_column="count_kind").collect()
        assert result.columns == ["count", "row", "count_is_even", "count_kind"]

    def test_kind_column_named_as_output_raises_value_error(self) -> None:
        frame = pl.LazyFrame({"count": ["one"]})
        with pytest.raises(ValueError, match="'count_value' is an output"):
            parse_column(frame, number_parser(), kind_column="count_value")

    def test_unknown_phrase_without_strict_has_unknown_kind(self) -> None:
        frame = pl.LazyFrame({"count": ["many"]})
        with pytest.warns(UserWarning, match="given null"):
            result = parse_column(
                frame, number_parser(), strict=False, kind_column="count_kind"
            )
        assert result.collect()["count_kind"].to_list() == ["unknown"]

    def test_replacing_puts_first_output_in_place_of_column(self) -> None:
        frame = pl.LazyFrame({"row": [1, 2], "count": ["one", "two"], "last": [0, 0]})
        result = parse_column(frame, number_parser(replaces_column=True)).collect()
        assert result.columns == ["row", "count", "last", "count_is_even"]
        assert result["count"].to_list() == [1, 2]

    def test_column_that_is_not_text_is_read_as_text(self) -> None:
        def parse_digit(text: str | None) -> ParsedNumber:
            if text is None:
                return ParsedNumber("not_a_description")
            return ParsedNumber("described", int(text) * 10)

        parser = ColumnParser(
            column="code",
            parse_phrase=parse_digit,
            outputs=(OutputColumn("code", lambda n: n.value, pl.Int64()),),
            replaces_column=True,
        )
        result = parse_column(pl.LazyFrame({"code": [1, None, 3]}), parser).collect()
        assert result["code"].to_list() == [10, None, 30]

    def test_each_distinct_phrase_is_parsed_once(self) -> None:
        parsed: list[str | None] = []

        def record(text: str | None) -> ParsedNumber:
            parsed.append(text)
            return parse_number(text)

        parser = ColumnParser(
            column="count", parse_phrase=record, outputs=number_parser().outputs
        )
        frame = pl.LazyFrame({"count": ["one", "two", "one", "one"]})
        parse_column(frame, parser).collect()
        assert sorted(parsed, key=str) == ["one", "two"]

    def test_unknown_phrase_raises_value_error_listing_it(self) -> None:
        frame = pl.LazyFrame({"count": ["seven", "one", "eight"]})
        with pytest.raises(
            ValueError, match=r"2 unknown phrases in 'count': 'eight'; 'seven'"
        ):
            parse_column(frame, number_parser())

    def test_unknown_phrase_without_strict_gives_null_and_warns(self) -> None:
        frame = pl.LazyFrame({"count": ["seven", "one"]})
        with pytest.warns(UserWarning, match=r"1 unknown phrases.*given null"):
            result = parse_column(frame, number_parser(), strict=False)
        assert result.collect()["count_value"].to_list() == [None, 1]
