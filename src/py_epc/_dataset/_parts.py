"""Split Scotland's values into the parts of a building."""

from typing import Final

import polars as pl
from pydantic import BaseModel

from py_epc._dataset._models import MODEL_CONFIG

# Suffixes of the columns added for each element's parts.
_PARTS_SUFFIX: Final = "_parts"
_PART_COUNT_SUFFIX: Final = "_part_count"


class BuildingElement(BaseModel):
    """One element of a building, with its description and two ratings."""

    model_config = MODEL_CONFIG

    name: str
    """Name of the element, such as ``"walls"``, for the columns added."""

    description: str
    """Column describing the element, such as ``"walls_description"``."""

    energy_eff: str
    """Column rating its energy efficiency, such as ``"walls_energy_eff"``."""

    env_eff: str
    """Column rating its environmental impact, such as ``"walls_env_eff"``."""


class MisalignedParts(BaseModel):
    """Counts of certificates whose parts do not line up, for one element."""

    model_config = MODEL_CONFIG

    different_counts: int
    """Certificates whose description and ratings have different numbers of
    parts, so the ratings may belong to other parts."""


def split_building_parts(frame: pl.LazyFrame, element: BuildingElement) -> pl.LazyFrame:
    """
    Return `frame` with an element's columns reduced to its main part.

    Scotland describes each part of a building, such as an extension,
    separated by ``" | "``, with each rating split the same way. The main
    part is the first part with a description, which matches England and
    Wales, where only the first part is published. Its description and
    ratings replace the published values.

    Every part is kept, in three lists of the same order,
    ``<description>_parts``, ``<energy_eff>_parts`` and ``<env_eff>_parts``,
    with ``<name>_part_count`` the number of parts. Empty parts are null.

    Parameters
    ----------
    frame : polars.LazyFrame
        Rows of a Scotland register.
    element : BuildingElement
        The element whose columns to split.

    Returns
    -------
    polars.LazyFrame
        The rows, with the element's main part and its parts added.
    """
    columns = (element.description, element.energy_eff, element.env_eff)
    parts = {column: _parts(pl.col(column)) for column in columns}
    main = (
        parts[element.description].list.eval(pl.element().is_not_null()).list.arg_max()
    )
    return frame.with_columns(
        *(part.alias(f"{column}{_PARTS_SUFFIX}") for column, part in parts.items()),
        parts[element.description]
        .list.len()
        .alias(f"{element.name}{_PART_COUNT_SUFFIX}"),
    ).with_columns(
        pl.col(f"{column}{_PARTS_SUFFIX}")
        .list.get(main, null_on_oob=True)
        .alias(column)
        for column in columns
    )


def misaligned_parts(frame: pl.LazyFrame, element: BuildingElement) -> MisalignedParts:
    """
    Return counts of certificates whose parts do not line up.

    Parameters
    ----------
    frame : polars.LazyFrame
        Rows of a Scotland register, before `split_building_parts`.
    element : BuildingElement
        The element to check.

    Returns
    -------
    MisalignedParts
        The counts.
    """
    description = _parts(pl.col(element.description))
    counts = [description.list.len()]
    counts += [
        _parts(pl.col(column)).list.len()
        for column in (element.energy_eff, element.env_eff)
    ]
    different = pl.any_horizontal(
        counts[0].is_not_null() & count.is_not_null() & (count != counts[0])
        for count in counts[1:]
    )
    result = frame.select(different.sum()).collect(engine="streaming")
    return MisalignedParts(different_counts=int(result.item()))


def _parts(column: pl.Expr) -> pl.Expr:
    """Return a column's parts, trimmed, with empty parts as null."""
    return column.str.split("|").list.eval(
        pl.when(pl.element().str.strip_chars() != "").then(
            pl.element().str.strip_chars()
        )
    )
