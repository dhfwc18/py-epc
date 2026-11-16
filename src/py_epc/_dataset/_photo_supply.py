"""Parse Scotland's photovoltaic arrays into totals for each dwelling."""

import re
from dataclasses import dataclass
from typing import Final

import polars as pl

from py_epc._dataset._phrases import (
    ColumnParser,
    OutputColumn,
    ParsedPhrase,
    prepare_phrase,
)

# The word each array phrase starts with.
_ARRAY_PREFIX_PATTERN: Final = re.compile(r"^array:\s*", re.IGNORECASE)

# A share of roof area, such as "40%". An empty share, "%", gives none.
_ROOF_AREA_PATTERN: Final = re.compile(r"(?P<percent>[0-9]+(?:\.[0-9]+)?)?\s*%")

# A number, such as a peak power of "2.7" kW or an older supply of "00".
_NUMBER_PATTERN: Final = re.compile(r"[0-9]+(?:\.[0-9]+)?")

# Fields of an array that give how much of it there is. "Supply", an older
# form, is read as a share of roof area, as England and Wales defines it.
_ROOF_AREA_FIELD: Final = "roof area"
_SUPPLY_FIELD: Final = "supply"
_PEAK_POWER_FIELD: Final = "peak power"
_CONNECTION_FIELD: Final = "connection"

# Fields read but not given, as only totals are.
_IGNORED_FIELDS: Final = frozenset({"orientation", "pitch", "overshading"})

# Connections of an array, keyed by phrase. "Not applicable" says there is
# no array to connect.
_PV_CONNECTION_PHRASES: Final[dict[str, str | None]] = {
    "connected to dwelling's electricity meter": "connected",
    "not connected to dwelling's electricity meter": "not_connected",
    "not recorded": "not_recorded",
    "not applicable (fghrs or no pv)": None,
}

# Connections of a dwelling's arrays, "partly_connected" if some are
# connected and some not.
_PV_CONNECTIONS: Final = (
    "connected",
    "partly_connected",
    "not_connected",
    "not_recorded",
)


@dataclass(frozen=True)
class ParsedPhotoSupply(ParsedPhrase):
    """Totals of one phrase's photovoltaic arrays."""

    roof_area_percent: float | None = None
    """Share of roof area the arrays cover, if given as a share."""

    peak_kw: float | None = None
    """Peak power of the arrays, in kW, if given as a power."""

    array_count: int | None = None
    """Number of arrays with panels, 0 if there are none."""

    connection: str | None = None
    """Whether the arrays with panels are connected to the dwelling's
    meter."""


@dataclass(frozen=True)
class _Array:
    """What one array of a phrase gives."""

    roof_area_percent: float | None = None
    peak_kw: float | None = None
    connection: str | None = None


def parse_photo_supply(text: str | None) -> ParsedPhotoSupply:
    """
    Return the totals of a cleaned phrase of photovoltaic arrays.

    Arrays are separated by ``"|"``, and each gives either a share of roof
    area, such as ``"Array: Roof Area: 40%; Connection: connected to
    dwelling's electricity meter;"``, or a peak power with its orientation,
    pitch and overshading. An older ``"Supply"`` is read as a share of roof
    area. Shares and powers are summed. An array has panels if its share or
    power is above 0, and only the connections of arrays with panels are
    read. Orientation, pitch and overshading are not given.

    Parameters
    ----------
    text : str or None
        A phrase, already cleaned by `clean_text`.

    Returns
    -------
    ParsedPhotoSupply
        The totals. Its kind is ``"unknown"`` if an array has a field or
        value of no known form, and ``"invalid"`` if a share is above 100.
    """
    phrase = prepare_phrase(text)
    if phrase is None:
        return ParsedPhotoSupply("not_a_description")
    arrays: list[_Array] = []
    for piece in phrase.split("|"):
        piece = _ARRAY_PREFIX_PATTERN.sub("", piece.strip())
        if not piece:
            continue
        array = _parse_array(piece)
        if array is None:
            return ParsedPhotoSupply("unknown")
        if array.roof_area_percent is not None or array.peak_kw is not None:
            arrays.append(array)
    if not arrays:
        return ParsedPhotoSupply("not_a_description")
    shares = [a.roof_area_percent for a in arrays if a.roof_area_percent is not None]
    powers = [a.peak_kw for a in arrays if a.peak_kw is not None]
    with_panels = [
        a for a in arrays if (a.roof_area_percent or 0) > 0 or (a.peak_kw or 0) > 0
    ]
    roof_area_percent = sum(shares) if shares else None
    return ParsedPhotoSupply(
        (
            "invalid"
            if roof_area_percent is not None and roof_area_percent > 100
            else "described"
        ),
        roof_area_percent=roof_area_percent,
        peak_kw=sum(powers) if powers else None,
        array_count=len(with_panels),
        connection=_dwelling_connection(with_panels),
    )


def photo_supply_parser() -> ColumnParser[ParsedPhotoSupply]:
    """
    Return a parser giving the totals of each dwelling's photovoltaic arrays.

    The columns given are ``photo_supply_roof_area_percent``,
    ``photo_supply_peak_kw``, ``photo_supply_array_count`` and
    ``photo_supply_connection``, a `polars.Enum`.

    Returns
    -------
    ColumnParser
        A parser of ``photo_supply_description`` for `parse_column`.
    """
    return ColumnParser(
        column="photo_supply_description",
        parse_phrase=parse_photo_supply,
        outputs=(
            OutputColumn(
                "photo_supply_roof_area_percent",
                lambda parsed: parsed.roof_area_percent,
                pl.Float64(),
            ),
            OutputColumn(
                "photo_supply_peak_kw", lambda parsed: parsed.peak_kw, pl.Float64()
            ),
            OutputColumn(
                "photo_supply_array_count",
                lambda parsed: parsed.array_count,
                pl.Int64(),
            ),
            OutputColumn(
                "photo_supply_connection",
                lambda parsed: parsed.connection,
                pl.Enum(list(_PV_CONNECTIONS)),
            ),
        ),
    )


def _parse_array(piece: str) -> _Array | None:
    """Return what one array gives, or None if it has an unknown form."""
    fields: dict[str, str] = {}
    for field in piece.split(";"):
        if not field.strip():
            continue
        name, colon, value = field.partition(":")
        if not colon:
            return None
        fields[name.strip().casefold()] = value.strip()
    connection = None
    if _CONNECTION_FIELD in fields:
        phrase = fields.pop(_CONNECTION_FIELD).casefold()
        if phrase not in _PV_CONNECTION_PHRASES:
            return None
        connection = _PV_CONNECTION_PHRASES[phrase]
    for name in _IGNORED_FIELDS & fields.keys():
        del fields[name]
    if len(fields) != 1:
        return None
    ((name, value),) = fields.items()
    if name == _ROOF_AREA_FIELD:
        if not (share := _ROOF_AREA_PATTERN.fullmatch(value)):
            return None
        percent = share["percent"]
        return _Array(float(percent) if percent else None, None, connection)
    if not _NUMBER_PATTERN.fullmatch(value):
        return None
    if name == _SUPPLY_FIELD:
        return _Array(float(value), None, connection)
    if name == _PEAK_POWER_FIELD:
        return _Array(None, float(value), connection)
    return None


def _dwelling_connection(arrays: list[_Array]) -> str | None:
    """Return how a dwelling's arrays are connected, from each array's."""
    known = {a.connection for a in arrays} - {None, "not_recorded"}
    if {"connected", "not_connected"} <= known:
        return "partly_connected"
    if known:
        return known.pop()
    if any(a.connection == "not_recorded" for a in arrays):
        return "not_recorded"
    return None
