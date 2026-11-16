"""Py-EPC library to download & process Energy Performance of Building data."""

from py_epc._config import Config
from py_epc._dataset._build import (
    DatasetManifest,
    build_dataset,
    build_epc_dataset,
)
from py_epc._download import (
    DownloadError,
    download,
    download_epc,
    download_ons_postcode_directory,
    download_os_open_uprn,
)
from py_epc._enums import EPCOrigin, EPCSegment, ONSProduct, OSProduct
from py_epc._manifest import DownloadManifest
from py_epc._output import OutputType
from py_epc._pipeline import OutputManifest, run_epc_pipeline
from py_epc._process._build import (
    ProcessManifest,
    ProcessWarning,
    process,
    process_epc,
)
from py_epc._process._postcodes import (
    PostcodeManifest,
    process_ons_postcode_directory,
)
from py_epc._scan import scan, scan_epc
from py_epc._source import Source

__all__ = [
    "Config",
    "DatasetManifest",
    "DownloadError",
    "DownloadManifest",
    "EPCOrigin",
    "EPCSegment",
    "ONSProduct",
    "OSProduct",
    "OutputManifest",
    "OutputType",
    "PostcodeManifest",
    "ProcessManifest",
    "ProcessWarning",
    "Source",
    "build_dataset",
    "build_epc_dataset",
    "download",
    "download_epc",
    "download_ons_postcode_directory",
    "download_os_open_uprn",
    "process",
    "process_epc",
    "process_ons_postcode_directory",
    "run_epc_pipeline",
    "scan",
    "scan_epc",
]
