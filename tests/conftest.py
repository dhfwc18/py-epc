"""Fixtures shared by the tests."""

import traceback
from collections.abc import Callable

import pytest


@pytest.fixture(autouse=True)
def fresh_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test from the defaults, and undo its changes afterwards."""
    monkeypatch.setattr("py_epc._config._current_config", None)
    monkeypatch.delenv("PY_EPC_CONFIG", raising=False)


@pytest.fixture
def error_text() -> Callable[[BaseException], str]:
    """
    Return a function giving every message a traceback shows for an error.

    The text holds the messages and notes of the error and of every error
    chained to it, so a test can check that no secret appears anywhere a user
    or a log would see it.

    The source lines are left out, because a test's own source may hold the
    secret it checks for.
    """

    def text(error: BaseException) -> str:
        shown: traceback.TracebackException | None
        shown = traceback.TracebackException.from_exception(error)
        messages: list[str] = []
        while shown is not None:
            messages.extend(shown.format_exception_only())
            shown = shown.__cause__ or (
                None if shown.__suppress_context__ else shown.__context__
            )
        return "".join(messages)

    return text
