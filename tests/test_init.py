"""Tests for the package root."""

import py_epc


class TestImport:
    """Importing the package."""

    def test_import_package_resolves_to_py_epc(self) -> None:
        assert py_epc.__name__ == "py_epc"
