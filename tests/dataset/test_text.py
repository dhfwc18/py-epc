"""Tests for cleaning published text into ASCII."""

import polars as pl
import pytest

from py_epc._dataset import _text as text_module
from py_epc._dataset._text import clean_text, normalise_text_columns, unknown_characters

# A U-value's unit, W/m2K, whose superscript two was decoded wrongly once.
_UNIT_SQUARE_DECODED_WRONGLY_ONCE = (
    "W/m\N{LATIN CAPITAL LETTER A WITH CIRCUMFLEX}\N{SUPERSCRIPT TWO}K"
)

# "Programmer's controls", whose apostrophe was decoded wrongly twice.
_APOSTROPHE_DECODED_WRONGLY_TWICE = (
    "Programmer\N{LATIN CAPITAL LETTER A WITH TILDE}\N{CENT SIGN}"
    "\N{LATIN SMALL LETTER A WITH CIRCUMFLEX}\N{SINGLE LOW-9 QUOTATION MARK}"
    "\N{NOT SIGN}\N{LATIN SMALL LETTER A WITH CIRCUMFLEX}"
    "\N{DOUBLE LOW-9 QUOTATION MARK}\N{CENT SIGN}s controls"
)


class TestCleanText:
    """`clean_text` turns published text into ASCII, keeping the pound sign."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param("Cavity wall, filled", "Cavity wall, filled", id="ascii"),
            pytest.param("0.30 W/m&#0178;K", "0.30 W/m2K", id="html-entity"),
            pytest.param("Tom &amp; Jerry", "Tom & Jerry", id="html-ampersand"),
            pytest.param(
                _UNIT_SQUARE_DECODED_WRONGLY_ONCE, "W/m2K", id="decoded-wrongly-once"
            ),
            pytest.param(
                _APOSTROPHE_DECODED_WRONGLY_TWICE,
                "Programmer's controls",
                id="decoded-wrongly-twice",
            ),
            pytest.param("0.30 W/m\N{SUPERSCRIPT TWO}K", "0.30 W/m2K", id="square"),
            pytest.param("5 m\N{SUPERSCRIPT THREE}/h", "5 m3/h", id="cube"),
            pytest.param(
                "owner\N{RIGHT SINGLE QUOTATION MARK}s", "owner's", id="quote"
            ),
            pytest.param("1900\N{EN DASH}1929", "1900-1929", id="dash"),
            pytest.param("a\N{NO-BREAK SPACE}b", "a b", id="no-break-space"),
            pytest.param("Note\N{FULLWIDTH COLON} x", "Note: x", id="fullwidth"),
            pytest.param(
                "Ynys M\N{LATIN SMALL LETTER O WITH CIRCUMFLEX}n",
                "Ynys Mon",
                id="accent",
            ),
            pytest.param(
                "dw\N{LATIN SMALL LETTER W WITH CIRCUMFLEX}r", "dwwr", id="welsh-w"
            ),
            pytest.param(
                "Stra\N{LATIN SMALL LETTER SHARP S}e", "Strasse", id="sharp-s"
            ),
            pytest.param(
                "Save \N{POUND SIGN}120 a year",
                "Save \N{POUND SIGN}120 a year",
                id="pound",
            ),
            pytest.param(
                "  Solid   brick,\tas built ", "Solid brick, as built", id="whitespace"
            ),
            pytest.param("Mid-Terrace", "Mid-Terrace", id="case-kept"),
            pytest.param(
                "o\N{ACUTE ACCENT}r mannau", "o'r mannau", id="acute-as-apostrophe"
            ),
            pytest.param("0.28 W/m??K", "0.28 W/m2K", id="lost-square"),
            pytest.param("0.28 W/m?K", "0.28 W/m2K", id="lost-square-one-mark"),
            pytest.param(
                "40% o\N{LATIN SMALL LETTER A WITH CIRCUMFLEX}??r mannau",
                "40% o'r mannau",
                id="lost-quote",
            ),
            pytest.param("Unknown ??? text", "Unknown ??? text", id="other-marks-kept"),
            pytest.param(
                "St Andrew\N{INVERTED QUESTION MARK}s",
                "St Andrew's",
                id="lost-apostrophe",
            ),
            pytest.param("Flat 1\N{FRACTION SLASH}2", "Flat 1/2", id="fraction-slash"),
            pytest.param("Flat \N{VULGAR FRACTION ONE HALF}", "Flat 1/2", id="half"),
            pytest.param("\N{BULLET} Main Street", "Main Street", id="bullet"),
            pytest.param("\N{EURO SIGN}120", "EUR120", id="euro"),
            pytest.param("Hall\N{ZERO WIDTH SPACE}", "Hall", id="zero-width-space"),
        ],
    )
    def test_published_text_returns_ascii(self, value: str, expected: str) -> None:
        assert clean_text(value) == expected

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param("", id="empty"),
            pytest.param(" \t ", id="spaces"),
            pytest.param(None, id="none"),
        ],
    )
    def test_empty_text_returns_none(self, value: str | None) -> None:
        assert clean_text(value) is None

    def test_unknown_character_is_left_for_reporting(self) -> None:
        assert (
            clean_text("north \N{RIGHTWARDS ARROW} south")
            == "north \N{RIGHTWARDS ARROW} south"
        )


class TestUnknownCharacters:
    """`unknown_characters` names what is neither ASCII nor a pound sign."""

    def test_ascii_and_pound_have_none(self) -> None:
        assert unknown_characters("Save \N{POUND SIGN}120") == set()

    def test_other_character_is_named(self) -> None:
        assert unknown_characters("a \N{RIGHTWARDS ARROW} b") == {
            "\N{RIGHTWARDS ARROW}"
        }


class TestNormaliseTextColumns:
    """`normalise_text_columns` cleans chosen columns of a frame."""

    def test_chosen_columns_are_cleaned_and_others_kept(self) -> None:
        frame = pl.LazyFrame(
            {
                "walls": ["0.30 W/m&#0178;K", None, "Cavity  wall"],
                "postcode": ["AB1 0AA", "AB1 0AB", "AB1 0AD"],
            }
        )
        result = normalise_text_columns(frame, ["walls"]).collect()
        assert result.to_dict(as_series=False) == {
            "walls": ["0.30 W/m2K", None, "Cavity wall"],
            "postcode": ["AB1 0AA", "AB1 0AB", "AB1 0AD"],
        }

    def test_rows_keep_their_order_and_column_its_place(self) -> None:
        values = [f"value {n % 7}  " for n in range(50)]
        frame = pl.LazyFrame({"id": list(range(50)), "text": values, "last": [1] * 50})
        result = normalise_text_columns(frame, ["text"]).collect()
        assert result.columns == ["id", "text", "last"]
        assert result["id"].to_list() == list(range(50))
        assert result["text"].to_list() == [value.strip() for value in values]

    def test_unknown_character_raises_value_error_naming_it(self) -> None:
        frame = pl.LazyFrame({"walls": ["a \N{RIGHTWARDS ARROW} b"]})
        with pytest.raises(ValueError, match="walls: \\['\N{RIGHTWARDS ARROW}'\\]"):
            normalise_text_columns(frame, ["walls"])

    def test_unknown_character_without_strict_is_removed_and_warned(self) -> None:
        frame = pl.LazyFrame({"walls": ["a \N{RIGHTWARDS ARROW} b"]})
        with pytest.warns(UserWarning, match="removed characters"):
            result = normalise_text_columns(frame, ["walls"], strict=False)
        assert result.collect()["walls"].to_list() == ["a b"]

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param("a", id="one-character"),
            pytest.param("Save \N{POUND SIGN}5", id="pound"),
            pytest.param(" a", id="leading-space"),
            pytest.param("a ", id="trailing-space"),
            pytest.param("a  b", id="two-spaces"),
            pytest.param("a\tb", id="tab"),
            pytest.param("a &amp; b", id="html-entity"),
            pytest.param("0.3 W/m?K", id="lost-character"),
            pytest.param("Caf\N{LATIN SMALL LETTER E WITH ACUTE}", id="accent"),
            pytest.param("", id="empty"),
        ],
    )
    def test_any_value_gives_clean_text(self, value: str) -> None:
        frame = pl.LazyFrame({"text": [value]})
        result = normalise_text_columns(frame, ["text"]).collect()
        assert result["text"].to_list() == [clean_text(value)]

    def test_values_cleaning_cannot_change_are_not_cleaned(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cleaned: list[str | None] = []

        def record(value: str | None) -> str | None:
            cleaned.append(value)
            return clean_text(value)

        monkeypatch.setattr(text_module, "clean_text", record)
        frame = pl.LazyFrame({"text": ["Flat 1, High Street", "Flat 2 ", None]})
        normalise_text_columns(frame, ["text"]).collect()
        assert cleaned == ["Flat 2 "]
