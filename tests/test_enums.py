"""Tests for the names used in config keys."""

import pytest

from py_epc import EPCOrigin
from py_epc._enums import parse_member


class TestParseMember:
    """`parse_member` turns a member or its value into the member."""

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param("scotland", id="string"),
            pytest.param(EPCOrigin.SCOTLAND, id="member"),
        ],
    )
    def test_known_value_returns_member(self, value: EPCOrigin | str) -> None:
        assert parse_member(EPCOrigin, value) is EPCOrigin.SCOTLAND

    def test_unknown_value_raises_value_error_naming_choices(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"unknown EPCOrigin 'scottland'; expected one of 'scotland', ",
        ):
            parse_member(EPCOrigin, "scottland")
