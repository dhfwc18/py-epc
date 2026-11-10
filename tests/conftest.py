"""Fixtures shared by the tests."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def fresh_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test from the defaults, and undo its changes afterwards."""
    monkeypatch.setattr("py_epc._config._current_config", None)
    monkeypatch.delenv("PY_EPC_CONFIG", raising=False)


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    """Return where the test's saved config is written."""
    return tmp_path / "mine.json"


@pytest.fixture
def write_config(config_path: Path) -> Callable[[object], Path]:
    """Return a function that writes a JSON document as the saved config."""

    def write(document: object) -> Path:
        config_path.write_text(json.dumps(document), encoding="utf-8")
        return config_path

    return write


@pytest.fixture
def replace_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[[object], None]:
    """Return a function that ships other defaults, like a later release."""

    def replace(document: object) -> None:
        path = tmp_path / "defaults.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        monkeypatch.setattr("py_epc._config._DEFAULT_CONFIG_PATH", path)

    return replace
