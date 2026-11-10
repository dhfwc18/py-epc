"""Names of the EPC registers and certificate types."""

from enum import StrEnum


class EPCOrigin(StrEnum):
    """National register that publishes the certificates."""

    SCOTLAND = "scotland"
    """The Scottish EPC register."""

    ENGLAND_AND_WALES = "england-and-wales"
    """The England and Wales register. API requires a bearer token."""


class EPCSegment(StrEnum):
    """Type of energy performance certificate."""

    DOMESTIC = "domestic"
    """Energy Performance Certificates for homes."""

    NON_DOMESTIC = "non-domestic"
    """Energy Performance Certificates for other buildings."""

    DISPLAY = "display"
    """Display Energy Certificates for public buildings."""
