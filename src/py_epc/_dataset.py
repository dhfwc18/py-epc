"""Describe a source dataset to be downloaded."""

from pydantic import BaseModel, ConfigDict


class Dataset(BaseModel):
    """A published dataset and its metadata."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        hide_input_in_errors=True,
        use_attribute_docstrings=True,
    )

    name: str
    """Unique identifier, used as filename."""

    url: str
    """Source url to download the file."""

    publisher: str
    """Organisation that publishes the file."""

    expected_bytes: int | None = None
    """Size of the published file in bytes."""

    requires_token: bool = False
    """Whether downloading needs a token, passed to the download function."""
