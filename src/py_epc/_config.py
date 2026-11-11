"""Configurations for the Py-EPC library."""

import copy
import json
import os
from contextlib import ContextDecorator
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field

from py_epc._dataset import Dataset
from py_epc._enums import EPCOrigin, EPCSegment, ONSProduct, OSProduct

_DEFAULT_CONFIG_PATH: Final[Traversable] = files("py_epc") / "defaults.json"

# Environment variable naming a saved config to load on first use.
_CONFIG_ENV_VAR: Final = "PY_EPC_CONFIG"


class _ConfigContent(BaseModel):
    """The checked contents of the config."""

    model_config = ConfigDict(
        title="Config",
        frozen=True,
        extra="forbid",
        strict=True,
        hide_input_in_errors=True,
        use_attribute_docstrings=True,
    )
    version: Literal[1]
    """Version of the config format."""

    epc: dict[EPCOrigin, dict[EPCSegment, Dataset]] = Field(default_factory=dict)
    """EPC datasets, keyed by origin, then by segment."""

    ons: dict[ONSProduct, Dataset] = Field(default_factory=dict)
    """Datasets of the Office for National Statistics, keyed by product."""

    os: dict[OSProduct, Dataset] = Field(default_factory=dict)
    """Datasets of Ordnance Survey, keyed by product."""


_current_config: _ConfigContent | None = None


class Config(ContextDecorator):
    """
    The configuration settings for the Py-EPC library.

    The defaults are shipped with the package. Change them with the `set`,
    `reset` and `load` methods. Every change is checked against the config's
    schema, and a rejected change leaves the config as it was. Errors never
    show the values found, so private tokens used to fetch data cannot leak.

    `save` writes only the settings that differ from the defaults, so fixes to
    the defaults in later releases still apply to everything you have not
    changed. If the ``PY_EPC_CONFIG`` environment variable names a saved
    config, it is loaded on first use.

    When used as a context manager or a decorator, `Config` undoes every change
    made inside it on exit. The config is shared by every thread.

    Examples
    --------
    >>> from py_epc import Config
    >>> Config.get("epc.scotland.domestic.publisher")
    'Scottish Government'

    Change a setting:

    >>> Config.set("epc.scotland.domestic.url", "https://example.org/a.csv")

    Save the change, and load it in a later session:

    >>> Config.save("my_config.json")
    >>> Config.reset()
    >>> Config.load("my_config.json")
    >>> Config.get("epc.scotland.domestic.url")
    'https://example.org/a.csv'

    Change settings only inside a block:

    >>> with Config():
    ...     Config.set("epc.scotland.domestic.url", "https://example.org/b.csv")
    >>> Config.get("epc.scotland.domestic.url")
    'https://example.org/a.csv'
    """

    def __init__(self) -> None:
        # Configs saved on entry.
        self._saved: list[_ConfigContent] = []

    def __enter__(self) -> Self:
        """Save the config, so changes inside the block can be undone."""
        self._saved.append(_config())
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Restore the config saved on entry."""
        _replace(self._saved.pop())

    @classmethod
    def get(cls, key: str) -> Any:
        """
        Return the value of a setting.

        Parameters
        ----------
        key : str
            Dotted key of the setting, such as ``"epc.scotland.domestic.url"``.

        Returns
        -------
        Any
            A copy of the value, so changing it does not change the config.

        Raises
        ------
        KeyError
            If there is no setting with that key.
        """
        node: Any = _config()
        for part in _parts(key):
            if isinstance(node, BaseModel) and part in type(node).model_fields:
                node = getattr(node, part)
            elif isinstance(node, dict) and part in node:
                node = node[part]
            else:
                raise KeyError(f"No config setting named {key!r}.")

        return copy.deepcopy(node)

    @classmethod
    def set(cls, key: str, value: object) -> None:
        """
        Change a setting, or add an entry, such as a dataset.

        Parameters
        ----------
        key : str
            Dotted key of the setting. New entries, such as a dataset, may be
            entred to parts of the config that accepts them.
        value : object
            The new value, which must be JSON-compatible. An entry may be
            given as a model, such as a `Dataset`, or as a dict.

        Raises
        ------
        KeyError
            If the setting the key belongs to does not exist.
        TypeError
            If the value cannot be written as JSON.
        pydantic.ValidationError
            If the config would not be valid with the new entry added.
        """
        *parents, last = _parts(key)
        data = _config().model_dump(mode="json")
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json")
        _node(data, parents, key)[last] = value
        _replace(_validate(data))

    @classmethod
    def reset(cls, key: str | None = None) -> None:
        """
        Put a setting, or the whole config, back to the defaults.

        Parameters
        ----------
        key : str, optional
            Dotted key of the setting. An entry that is not in the defaults,
            such as an added dataset, is removed. Without a key, the whole
            config is reset.

        Raises
        ------
        KeyError
            If there is no setting with that key.
        pydantic.ValidationError
            If the reset would leave an added entry incomplete, such as
            resetting a field of an added dataset.
        """
        defaults = _defaults()
        if key is None:
            _replace(defaults)
            return
        *parents, last = _parts(key)
        data = _config().model_dump(mode="json")
        node = _node(data, parents, key)
        if last not in node:
            raise KeyError(f"No config setting named {key!r}.")
        try:
            default = _node(defaults.model_dump(mode="json"), parents, key)
        except KeyError:
            default = {}
        if last in default:
            node[last] = default[last]
        else:
            del node[last]
        _replace(_validate(data))

    @classmethod
    def state(cls) -> dict[str, Any]:
        """Return every setting as a dict, as it would be saved in JSON."""
        return _config().model_dump(mode="json", exclude={"version"})

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> None:
        """
        Use the defaults with a saved config laid over them.

        This replaces the config in use, so changes made with `set` are lost.
        If loading fails, the config in use is left as it was.

        Parameters
        ----------
        path : str | os.PathLike[str]
            A saved config, holding only what differs from the defaults.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file is not valid JSON, or not a JSON object with a
            ``"version"``.
        pydantic.ValidationError
            If the file does not match the config.
        """
        source = Path(path)
        try:
            saved = json.loads(source.read_bytes())
            if not isinstance(saved, dict) or "version" not in saved:
                raise ValueError('a saved config is a JSON object with "version"')
            config = _validate(_merge(_defaults().model_dump(mode="json"), saved))
        except ValueError as error:
            error.add_note(f"while loading {source}")
            raise
        _replace(config)

    @classmethod
    def save(cls, path: str | os.PathLike[str], *, overwrite: bool = False) -> None:
        """
        Save the settings that differ from the defaults, as JSON.

        The file is written under a temporary name and then renamed, so a
        failed write never leaves a partial file.

        Parameters
        ----------
        path : str | os.PathLike[str]
            File to write.
        overwrite : bool, default False
            Whether to replace the file if it already exists.

        Raises
        ------
        FileExistsError
            If the file exists and `overwrite` is false.
        ValueError
            If a default entry has been removed, which a saved config,
            holding only changes, cannot record.
        """
        target = Path(path)
        if target.exists() and not overwrite:
            raise FileExistsError(f"{target} exists; pass overwrite=True to replace it")
        config = _config()
        changes = _diff(
            _defaults().model_dump(mode="json"), config.model_dump(mode="json")
        )
        text = json.dumps({"version": config.version, **changes}, indent=2)
        temporary = target.with_name(f".{target.name}.tmp")
        try:
            temporary.write_text(text + "\n", encoding="utf-8")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


def _config() -> _ConfigContent:
    """Return the config in use, loading it on first use."""
    if _current_config is None:
        _replace(_defaults())
        if path := os.environ.get(_CONFIG_ENV_VAR):
            Config.load(path)
    assert _current_config is not None
    return _current_config


def _replace(config: _ConfigContent) -> None:
    """Make `config` the config in use."""
    global _current_config
    _current_config = config


def _defaults() -> _ConfigContent:
    """Read the defaults shipped with the package."""
    return _ConfigContent.model_validate_json(_DEFAULT_CONFIG_PATH.read_bytes())


def _validate(data: object) -> _ConfigContent:
    """Check JSON-shaped data against the config."""
    return _ConfigContent.model_validate_json(json.dumps(data))


def _parts(key: str) -> list[str]:
    """Split a dotted key, rejecting keys that name no setting."""
    parts = key.split(".")
    if "" in parts or parts[0] == "version":
        raise KeyError(f"No config setting named {key!r}.")
    return parts


def _node(data: dict[str, Any], parts: list[str], key: str) -> dict[str, Any]:
    """Return the dict that `parts` lead to in `data`."""
    for part in parts:
        if not isinstance(data, dict) or part not in data:
            raise KeyError(f"No config setting named {key!r}.")
        data = data[part]
    return data


def _merge(default: object, override: object) -> object:
    """Lay `override` over `default`, merging nested dicts key by key."""
    if not (isinstance(default, dict) and isinstance(override, dict)):
        return override
    merged = dict(default)
    for key, value in override.items():
        merged[key] = _merge(merged[key], value) if key in merged else value
    return merged


def _diff(
    default: dict[str, Any], current: dict[str, Any], location: str = ""
) -> dict[str, Any]:
    """Return the part of `current` that differs from `default`."""
    prefix = f"{location}." if location else ""
    if removed := default.keys() - current.keys():
        names = ", ".join(sorted(f"{prefix}{key}" for key in removed))
        raise ValueError(f"cannot save without {names}: a saved config holds changes")
    changes = {}
    for key, value in current.items():
        if key not in default:
            changes[key] = value
        elif isinstance(value, dict) and isinstance(default[key], dict):
            if nested := _diff(default[key], value, f"{prefix}{key}"):
                changes[key] = nested
        elif value != default[key]:
            changes[key] = value
    return changes
