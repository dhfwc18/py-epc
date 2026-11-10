"""Tests for the package root."""

import py_epc


class TestPackage:
    """The `py_epc` package can be imported."""

    def test_import_resolves_to_py_epc(self) -> None:
        assert py_epc.__name__ == "py_epc"
