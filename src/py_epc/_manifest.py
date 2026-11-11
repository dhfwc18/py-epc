"""Manifests written beside each completed result."""

from pathlib import Path
from typing import Final, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict

# Name of the manifest file in each folder.
MANIFEST_NAME: Final = "manifest.json"


class DownloadManifest(BaseModel):
    """The record of one download, written once the file is complete."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        hide_input_in_errors=True,
        use_attribute_docstrings=True,
    )

    version: Literal[1] = 1
    """Version of the manifest format."""

    dataset_key: str
    """Config key of the source, such as ``"epc.scotland.domestic"``."""

    source_url: str
    """The source's stable URL. A pre-signed URL is never stored."""

    publisher: str
    """Organisation that publishes the file."""

    retrieved_at: AwareDatetime
    """When the download started, in UTC."""

    file_name: str
    """Name of the downloaded file, in the manifest's folder."""

    size_bytes: int
    """Size of the downloaded file, in bytes."""

    sha256: str
    """SHA-256 checksum of the downloaded file, as lowercase hex."""


def write_manifest(folder: Path, manifest: BaseModel) -> None:
    """Write `manifest` to the manifest file in `folder`, as JSON."""
    (folder / MANIFEST_NAME).write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )


def read_manifest[M: BaseModel](folder: Path, model: type[M]) -> M:
    """
    Return the manifest in `folder`, checked against `model`.

    Parameters
    ----------
    folder : Path
        Folder holding the manifest file.
    model : type[pydantic.BaseModel]
        Manifest model to check it against, such as `DownloadManifest`.

    Returns
    -------
    pydantic.BaseModel
        The manifest, as an instance of `model`.

    Raises
    ------
    FileNotFoundError
        If `folder` has no manifest, as when a step did not complete.
    pydantic.ValidationError
        If the manifest does not match `model`.
    """
    return model.model_validate_json((folder / MANIFEST_NAME).read_bytes())
