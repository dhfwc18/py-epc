"""Register definitions (how to read and name each register's columns)."""

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from importlib.resources import files
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Type a column is parsed into.
type ColumnType = Literal["text", "integer", "number", "date", "datetime", "flag"]

# Shared config across models in this module.
_MODEL_CONFIG = ConfigDict(
    frozen=True,
    extra="forbid",
    strict=True,
    hide_input_in_errors=True,
    use_attribute_docstrings=True,
)


class Column(BaseModel):
    """One published column: its header names, new name, type and meaning."""

    model_config = _MODEL_CONFIG

    source: str
    """Name in the current header, such as ``"LODGEMENT_DATE"``."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    """Name after processing, in snake case, such as ``"lodgement_date"``."""

    type: ColumnType
    """Type the values are parsed into."""

    description: str
    """What each value records, such as the date the certificate was lodged."""

    aliases: tuple[str, ...] = ()
    """Names the same column had in the header of earlier releases."""

    formats: tuple[str, ...] = ()
    """Accepted ``strftime`` formats, tried in order. Dates and times only."""

    unit: str | None = None
    """Unit of the values, such as ``"m2"`` or ``"kWh/year"``."""

    restricted: bool = False
    """Whether the column is address data, which has a restricted licence."""

    @model_validator(mode="after")
    def _check_formats(self) -> Self:
        timed = self.type in {"date", "datetime"}
        if timed and not self.formats:
            raise ValueError(f"{self.type} column {self.source!r} needs formats")
        if not timed and self.formats:
            raise ValueError(f"{self.type} column {self.source!r} takes no formats")
        return self


class Register(BaseModel):
    """The columns of one register's published files, and how to read them."""

    model_config = _MODEL_CONFIG

    version: Literal[1]
    """Version of the definition format."""

    source_key: str
    """Config key of the source, such as ``"epc.scotland.domestic"``."""

    member_pattern: str
    """Pattern a yearly member's whole name matches, capturing the year."""

    label_rows: int = Field(default=0, ge=0)
    """Rows of labels below the header, which are skipped."""

    missing_values: tuple[str, ...]
    """Values that mean a value is missing, in every column."""

    true_values: tuple[str, ...] = ()
    """Values that mean true, in every flag column."""

    false_values: tuple[str, ...] = ()
    """Values that mean false, in every flag column."""

    year_column: str | None = None
    """Name of the date column whose year should match the member's year."""

    columns: tuple[Column, ...]
    """Every published column, in header order."""

    @property
    def member_regex(self) -> re.Pattern[str]:
        """Return `member_pattern`, compiled."""
        return re.compile(self.member_pattern)

    @model_validator(mode="after")
    def _check(self) -> Self:
        try:
            groups = re.compile(self.member_pattern).groups
        except re.error as error:
            raise ValueError(
                f"member_pattern is not a valid pattern: {error}"
            ) from None
        if groups != 1:
            raise ValueError("member_pattern must have one group, for the year")
        _check_unique("column name", (column.name for column in self.columns))
        _check_unique(
            "header name",
            (
                header
                for column in self.columns
                for header in (column.source, *column.aliases)
            ),
        )
        if self.year_column is not None and not any(
            column.name == self.year_column and column.type in {"date", "datetime"}
            for column in self.columns
        ):
            raise ValueError(
                f"year_column {self.year_column!r} is not a date or datetime column"
            )
        flags = any(column.type == "flag" for column in self.columns)
        if flags and not (self.true_values and self.false_values):
            raise ValueError("flag columns need true_values and false_values")
        return self

    def match_header(self, header: Sequence[str]) -> dict[str, Column]:
        """
        Return the column for each name in a file's header.

        A name matches a column by its `Column.source` or one of its
        `Column.aliases`.

        Parameters
        ----------
        header : Sequence[str]
            Column names in the file's header.

        Returns
        -------
        dict[str, Column]
            Column for each header name, in header order.

        Raises
        ------
        ValueError
            If the header has a name no column matches, lacks a column, or
            names one column twice.
        """
        by_header = {
            name: column
            for column in self.columns
            for name in (column.source, *column.aliases)
        }
        problems: list[str] = []
        unknown = [name for name in header if name not in by_header]
        if unknown:
            problems.append(f"unknown columns {unknown}")
        matched = [by_header[name] for name in header if name in by_header]
        repeated = sorted(
            column.source for column, count in Counter(matched).items() if count > 1
        )
        if repeated:
            problems.append(f"columns given more than once {repeated}")
        missing = [column.source for column in self.columns if column not in matched]
        if missing:
            problems.append(f"missing columns {missing}")
        if problems:
            raise ValueError(
                f"header does not match {self.source_key!r}: " + "; ".join(problems)
            )
        return {name: by_header[name] for name in header}


def load_register(key: str) -> Register:
    """
    Return the shipped register definition for a source.

    Parameters
    ----------
    key : str
        Config key of the source, such as ``"epc.scotland.domestic"``.

    Returns
    -------
    Register
        The definition, checked when loaded.

    Raises
    ------
    KeyError
        If no definition is shipped for the source.
    pydantic.ValidationError
        If the shipped definition is not valid.
    ValueError
        If the shipped definition names a different source.
    """
    path = files("py_epc._process").joinpath(
        "_registers", key.replace(".", "-") + ".json"
    )
    if not path.is_file():
        raise KeyError(f"No register definition for {key!r}.")
    register = Register.model_validate_json(path.read_bytes())
    if register.source_key != key:
        raise ValueError(
            f"register definition for {key!r} names {register.source_key!r}"
        )
    return register


def _check_unique(what: str, names: Iterable[str]) -> None:
    """Raise ValueError naming every value of `names` that repeats."""
    repeated = sorted(name for name, count in Counter(names).items() if count > 1)
    if repeated:
        raise ValueError(f"{what}s must be unique; repeated: {repeated}")
