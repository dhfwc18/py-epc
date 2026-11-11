"""Tests for the package config."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from py_epc import Config, Dataset, EPCOrigin, EPCSegment, ONSProduct

# Fields of a minimal valid dataset.
_DATASET: dict[str, Any] = {
    "name": "n",
    "url": "https://example.invalid/file",
    "publisher": "p",
}

# Key of the setting most tests change.
_URL_KEY = "epc.scotland.domestic.url"


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


def read_json(path: Path) -> object:
    """Return the JSON document in a file."""
    return json.loads(path.read_text(encoding="utf-8"))


class TestDefaultsJson:
    """The shipped ``defaults.json`` holds the expected datasets."""

    def test_epc_datasets_cover_expected_origin_segment_pairs(self) -> None:
        pairs = {
            (origin, segment)
            for origin, segments in Config.state()["epc"].items()
            for segment in segments
        }
        # Every origin-segment pair except Scotland display should be covered.
        expected = {
            (origin, segment) for origin in EPCOrigin for segment in EPCSegment
        } - {(EPCOrigin.SCOTLAND, EPCSegment.DISPLAY)}
        assert pairs == expected

    def test_all_epc_dataset_names_are_unique(self) -> None:
        names = [
            dataset["name"]
            for segments in Config.state()["epc"].values()
            for dataset in segments.values()
        ]
        assert len(names) == len(set(names))

    def test_only_england_and_wales_epc_datasets_require_token(self) -> None:
        epc = Config.state()["epc"]
        assert all(d["requires_token"] for d in epc["england-and-wales"].values())
        assert not any(d["requires_token"] for d in epc["scotland"].values())

    def test_ons_datasets_cover_every_product(self) -> None:
        assert set(Config.state()["ons"]) == set(ONSProduct)

    def test_ons_postcode_directory_needs_no_token(self) -> None:
        assert not Config.get("ons.postcode-directory.requires_token")


class TestConfigGet:
    """`Config.get` reads a setting by its dotted key."""

    def test_field_key_returns_value(self) -> None:
        assert Config.get("epc.scotland.domestic.publisher") == "Scottish Government"

    def test_dataset_key_returns_dataset(self) -> None:
        assert isinstance(Config.get("epc.scotland.domestic"), Dataset)

    def test_key_built_from_enums_returns_same_value_as_string_key(self) -> None:
        key = f"epc.{EPCOrigin.SCOTLAND}.{EPCSegment.DOMESTIC}"
        assert Config.get(key) == Config.get("epc.scotland.domestic")

    def test_changed_result_leaves_global_config_unchanged(self) -> None:
        Config.get("epc.scotland").clear()
        assert Config.get("epc.scotland")

    @pytest.mark.parametrize(
        "key",
        [
            pytest.param("epc.scottland", id="unknown-origin"),
            pytest.param("epc.scotland.display", id="missing-dataset"),
            pytest.param("epc.scotland.domestic.token", id="unknown-field"),
            pytest.param("postcode", id="unknown-section"),
            pytest.param("version", id="version"),
            pytest.param("epc..domestic", id="empty-part"),
        ],
    )
    def test_unknown_key_raises_key_error(self, key: str) -> None:
        with pytest.raises(KeyError, match="No config setting named"):
            Config.get(key)


class TestConfigSet:
    """`Config.set` changes a setting, checking it as it is set."""

    def test_field_key_changes_value(self) -> None:
        Config.set(_URL_KEY, "https://x.invalid")
        assert Config.get(_URL_KEY) == "https://x.invalid"

    def test_field_key_leaves_other_settings_unchanged(self) -> None:
        before = Config.get("epc.scotland.non-domestic")
        Config.set(_URL_KEY, "https://x.invalid")
        assert Config.get("epc.scotland.non-domestic") == before

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(Dataset(**_DATASET), id="dataset"),
            pytest.param(_DATASET, id="dict"),
        ],
    )
    def test_new_dataset_key_adds_dataset(self, value: object) -> None:
        Config.set("epc.scotland.display", value)
        assert Config.get("epc.scotland.display") == Dataset(**_DATASET)

    @pytest.mark.parametrize(
        ("key", "value", "match"),
        [
            pytest.param(
                _URL_KEY, 1, r"url\n  Input should be a valid string", id="wrong-type"
            ),
            pytest.param(
                "epc.scotland.domestic.requires_token",
                "no",
                r"requires_token\n  Input should be a valid boolean",
                id="string-for-bool",
            ),
            pytest.param(
                "epc.scotland.domestic.token",
                "t",
                r"token\n  Extra inputs are not permitted",
                id="unknown-field",
            ),
            pytest.param(
                "epc.scottland",
                {},
                r"epc\.scottland\.\[key\]\n  Input should be",
                id="unknown-origin",
            ),
            pytest.param(
                "postcode", {}, r"postcode\n  Extra inputs", id="unknown-section"
            ),
        ],
    )
    def test_invalid_value_raises_validation_error_and_changes_nothing(
        self, key: str, value: object, match: str
    ) -> None:
        before = Config.state()
        with pytest.raises(ValidationError, match=match):
            Config.set(key, value)
        assert Config.state() == before

    def test_missing_parent_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="No config setting named"):
            Config.set("epc.scotland.display.url", "https://x.invalid")

    def test_invalid_value_is_not_shown_in_error(
        self, error_text: Callable[[BaseException], str]
    ) -> None:
        with pytest.raises(ValidationError) as caught:
            Config.set("epc.scotland.domestic.token", "not-for-logs")
        assert "not-for-logs" not in error_text(caught.value)


class TestConfigReset:
    """`Config.reset` puts settings back to the defaults."""

    def test_changed_key_restores_default(self) -> None:
        default = Config.get(_URL_KEY)
        Config.set(_URL_KEY, "https://x.invalid")
        Config.reset(_URL_KEY)
        assert Config.get(_URL_KEY) == default

    def test_added_dataset_key_removes_dataset(self) -> None:
        Config.set("epc.scotland.display", _DATASET)
        Config.reset("epc.scotland.display")
        with pytest.raises(KeyError):
            Config.get("epc.scotland.display")

    def test_no_key_restores_all_defaults(self) -> None:
        default = Config.state()
        Config.set(_URL_KEY, "https://x.invalid")
        Config.set("epc.scotland.display", _DATASET)
        Config.reset()
        assert Config.state() == default

    def test_unknown_key_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="No config setting named"):
            Config.reset("epc.scotland.display")

    def test_no_key_with_env_var_restores_shipped_defaults(
        self,
        write_config: Callable[[object], Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        default = Config.get(_URL_KEY)
        monkeypatch.setattr("py_epc._config._current_config", None)
        changed = {"scotland": {"domestic": {"url": "https://x.invalid"}}}
        path = write_config({"version": 1, "epc": changed})
        monkeypatch.setenv("PY_EPC_CONFIG", str(path))
        Config.reset()
        assert Config.get(_URL_KEY) == default


class TestConfigSave:
    """`Config.save` writes only what differs from the defaults."""

    def test_unchanged_config_writes_only_version(self, config_path: Path) -> None:
        Config.save(config_path)
        assert read_json(config_path) == {"version": 1}

    def test_changed_field_writes_only_that_field(self, config_path: Path) -> None:
        Config.set(_URL_KEY, "https://x.invalid")
        Config.save(config_path)
        assert read_json(config_path) == {
            "version": 1,
            "epc": {"scotland": {"domestic": {"url": "https://x.invalid"}}},
        }

    def test_added_dataset_writes_whole_dataset(self, config_path: Path) -> None:
        Config.set("epc.scotland.display", _DATASET)
        Config.save(config_path)
        assert read_json(config_path) == {
            "version": 1,
            "epc": {"scotland": {"display": Dataset(**_DATASET).model_dump()}},
        }

    def test_changed_config_loads_back_equal(self, config_path: Path) -> None:
        Config.set(_URL_KEY, "https://x.invalid")
        Config.set("epc.scotland.display", _DATASET)
        changed = Config.state()
        Config.save(config_path)
        Config.reset()
        Config.load(config_path)
        assert Config.state() == changed

    def test_removed_default_raises_value_error_and_writes_nothing(
        self, config_path: Path
    ) -> None:
        Config.set("epc.scotland", {})
        with pytest.raises(ValueError, match=r"cannot save without epc\.scotland\."):
            Config.save(config_path)
        assert not config_path.exists()

    def test_existing_file_raises_file_exists_error_and_keeps_file(
        self, config_path: Path
    ) -> None:
        config_path.write_text("keep", encoding="utf-8")
        with pytest.raises(FileExistsError, match="overwrite=True"):
            Config.save(config_path)
        assert config_path.read_text(encoding="utf-8") == "keep"

    def test_existing_file_with_overwrite_replaces_file(
        self, config_path: Path
    ) -> None:
        config_path.write_text("old", encoding="utf-8")
        Config.save(config_path, overwrite=True)
        assert read_json(config_path) == {"version": 1}

    def test_successful_write_leaves_no_temporary_file(self, config_path: Path) -> None:
        Config.save(config_path)
        assert list(config_path.parent.iterdir()) == [config_path]


class TestConfigLoad:
    """`Config.load` lays a saved config over the defaults."""

    def test_saved_field_replaces_only_that_default(
        self, write_config: Callable[[object], Path]
    ) -> None:
        publisher = Config.get("epc.scotland.domestic.publisher")
        changed = {"scotland": {"domestic": {"url": "https://x.invalid"}}}
        Config.load(write_config({"version": 1, "epc": changed}))
        assert Config.get(_URL_KEY) == "https://x.invalid"
        assert Config.get("epc.scotland.domestic.publisher") == publisher

    def test_unsaved_change_is_discarded(
        self, write_config: Callable[[object], Path]
    ) -> None:
        default = Config.get(_URL_KEY)
        Config.set(_URL_KEY, "https://x.invalid")
        Config.load(write_config({"version": 1}))
        assert Config.get(_URL_KEY) == default

    def test_newer_defaults_apply_to_settings_not_saved(
        self, config_path: Path, replace_defaults: Callable[[object], None]
    ) -> None:
        Config.set(_URL_KEY, "https://mine")
        Config.save(config_path)

        Config.reset()
        release: dict[str, Any] = {"version": 1, **Config.state()}
        release["epc"]["scotland"]["non-domestic"]["url"] = "https://fixed"
        replace_defaults(release)

        Config.load(config_path)
        assert Config.get(_URL_KEY) == "https://mine"
        assert Config.get("epc.scotland.non-domestic.url") == "https://fixed"

    @pytest.mark.parametrize(
        ("document", "error", "match"),
        [
            pytest.param({"epc": {}}, ValueError, '"version"', id="missing-version"),
            pytest.param([], ValueError, '"version"', id="not-an-object"),
            pytest.param(
                {"version": 2},
                ValidationError,
                r"version\n  Input should be 1",
                id="wrong-version",
            ),
            pytest.param(
                {"version": 1, "epcs": {}},
                ValidationError,
                r"epcs\n  Extra inputs are not permitted",
                id="unknown-section",
            ),
            pytest.param(
                {"version": 1, "epc": {"scottland": {"domestic": _DATASET}}},
                ValidationError,
                r"epc\.scottland\.\[key\]\n  Input should be",
                id="unknown-origin",
            ),
            pytest.param(
                {"version": 1, "epc": {"scotland": []}},
                ValidationError,
                r"epc\.scotland\n  Input should be an object",
                id="origin-not-an-object",
            ),
            pytest.param(
                {"version": 1, "epc": {"scotland": {"display": {"url": "u"}}}},
                ValidationError,
                r"epc\.scotland\.display\.name\n  Field required",
                id="new-dataset-missing-field",
            ),
            pytest.param(
                {"version": 1, "epc": {"scotland": {"domestic": {"token": "t"}}}},
                ValidationError,
                r"epc\.scotland\.domestic\.token\n  Extra inputs are not permitted",
                id="unknown-field",
            ),
        ],
    )
    def test_invalid_file_raises_value_error_and_changes_nothing(
        self,
        write_config: Callable[[object], Path],
        document: object,
        error: type[Exception],
        match: str,
    ) -> None:
        Config.set(_URL_KEY, "https://x.invalid")
        before = Config.state()
        with pytest.raises(error, match=match):
            Config.load(write_config(document))
        assert Config.state() == before

    def test_invalid_json_raises_value_error(self, config_path: Path) -> None:
        config_path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError, match="Expecting property name"):
            Config.load(config_path)

    def test_missing_file_raises_file_not_found_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            Config.load(tmp_path / "missing.json")

    def test_invalid_file_error_names_the_file(
        self, write_config: Callable[[object], Path]
    ) -> None:
        path = write_config({"version": 2})
        with pytest.raises(ValidationError) as caught:
            Config.load(path)
        assert caught.value.__notes__ == [f"while loading {path}"]

    def test_invalid_file_values_are_not_shown_in_error(
        self,
        write_config: Callable[[object], Path],
        error_text: Callable[[BaseException], str],
    ) -> None:
        values = {"token": "not-for-logs", "requires_token": "no-log"}
        path = write_config({"version": 1, "epc": {"scotland": {"domestic": values}}})
        with pytest.raises(ValidationError) as caught:
            Config.load(path)
        shown = error_text(caught.value)
        assert "not-for-logs" not in shown
        assert "no-log" not in shown


class TestConfigFirstUse:
    """The first use of `Config` loads the file named by ``PY_EPC_CONFIG``."""

    def test_env_var_set_loads_named_config(
        self,
        write_config: Callable[[object], Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        changed = {"scotland": {"domestic": {"url": "https://x.invalid"}}}
        path = write_config({"version": 1, "epc": changed})
        monkeypatch.setenv("PY_EPC_CONFIG", str(path))
        assert Config.get(_URL_KEY) == "https://x.invalid"


class TestConfigContext:
    """`Config`, as a context manager or decorator, undoes changes."""

    def test_block_exit_undoes_changes(self) -> None:
        default = Config.get(_URL_KEY)
        with Config():
            Config.set(_URL_KEY, "https://x.invalid")
            assert Config.get(_URL_KEY) == "https://x.invalid"
        assert Config.get(_URL_KEY) == default

    def test_block_exit_by_error_undoes_changes(self) -> None:
        default = Config.get(_URL_KEY)
        with pytest.raises(RuntimeError), Config():
            Config.set(_URL_KEY, "https://x.invalid")
            raise RuntimeError
        assert Config.get(_URL_KEY) == default

    def test_nested_blocks_undo_innermost_first(self) -> None:
        default = Config.get(_URL_KEY)
        with Config():
            Config.set(_URL_KEY, "https://outer")
            with Config():
                Config.set(_URL_KEY, "https://inner")
            assert Config.get(_URL_KEY) == "https://outer"
        assert Config.get(_URL_KEY) == default

    def test_decorated_function_return_undoes_changes(self) -> None:
        default = Config.get(_URL_KEY)

        @Config()
        def change() -> str:
            Config.set(_URL_KEY, "https://x.invalid")
            return Config.get(_URL_KEY)

        assert change() == "https://x.invalid"
        assert Config.get(_URL_KEY) == default
