"""Names used in config keys: EPC registers and types, ONS and OS products."""

from enum import StrEnum
from typing import Final


class EPCOrigin(StrEnum):
    """National register that publishes the certificates."""

    SCOTLAND = "scotland"
    """The Scottish EPC register."""

    ENGLAND_AND_WALES = "england-and-wales"
    """The England and Wales register. API requires a bearer token."""

    GB = "gb"
    """
    Both registers, read together for Great Britain by `scan_epc`.
    Nothing is downloaded, processed or built for it.
    """


# Origins that are registers, each publishing its own certificates.
EPC_REGISTERS: Final = (EPCOrigin.SCOTLAND, EPCOrigin.ENGLAND_AND_WALES)


class EPCSegment(StrEnum):
    """Type of energy performance certificate."""

    DOMESTIC = "domestic"
    """Energy Performance Certificates for homes."""

    NON_DOMESTIC = "non-domestic"
    """Energy Performance Certificates for other buildings."""

    DISPLAY = "display"
    """Display Energy Certificates for public buildings."""


class ONSProduct(StrEnum):
    """Source published by the Office for National Statistics."""

    POSTCODE_DIRECTORY = "postcode-directory"
    """The ONS Postcode Directory (ONSPD)."""


class OSProduct(StrEnum):
    """Source published by Ordnance Survey."""

    OPEN_UPRN = "open-uprn"
    """OS Open UPRN, the coordinates of every UPRN in Great Britain."""


def parse_member[E: StrEnum](enum: type[E], value: E | str) -> E:
    """
    Return the member of `enum` for `value`.

    Parameters
    ----------
    enum : type[StrEnum]
        Enum to look the value up in, such as `EPCOrigin`.
    value : StrEnum | str
        A member, or its value, such as ``"scotland"``.

    Returns
    -------
    StrEnum
        The member of `enum`.

    Raises
    ------
    ValueError
        If `value` is not a member's value. The message names the choices.
    """
    try:
        return enum(value)
    except ValueError:
        choices = ", ".join(repr(member.value) for member in enum)
        raise ValueError(
            f"unknown {enum.__name__} {value!r}; expected one of {choices}"
        ) from None


def parse_register(value: EPCOrigin | str) -> EPCOrigin:
    """
    Return the register for an origin, which must not be GB.

    Parameters
    ----------
    value : EPCOrigin | str
        An origin, or its value, such as ``"scotland"``.

    Returns
    -------
    EPCOrigin
        The register.

    Raises
    ------
    ValueError
        If `value` is not an origin, or is ``"gb"``, which is only read.
    """
    origin = parse_member(EPCOrigin, value)
    if origin not in EPC_REGISTERS:
        raise ValueError(
            f"{origin.value!r} combines both registers, so it is only read, by"
            " scan_epc; download, process and build each register instead"
        )
    return origin
