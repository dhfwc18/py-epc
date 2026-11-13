"""Tests for processing the ONS Postcode Directory."""

import codecs
from collections.abc import Callable
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from py_epc import Config, PostcodeManifest, process_ons_postcode_directory
from py_epc._manifest import read_manifest
from py_epc._process._postcodes import load_postcode_directory

# Key of the source the tests process.
_SOURCE_KEY = "ons.postcode-directory"

# Header of the directory's data file, with this release's year suffixes.
_HEADER = "pcds,lad26cd,pcon24cd,itl25cd,rgn26cd,lsoa21cd,imd25ind,doterm,wd26cd"

# Postcodes in England, Scotland and Northern Ireland, and one with no codes.
_ROWS = [
    "E1 6AN,E09000030,E14001172,E06000045,E12000007,E01000001,26525,,E05009317",
    "AB1 0AA,S12000033,S14000061,S30000026,S99999999,S01013482,0,199006,",
    "BT1 1AA,N09000003,N05000003,N10000001,N99999999,N21000001,0,,",
    "ZZ9 9ZZ,,,,,,,,",
]

# Lookups of codes to names, as published in the archive's Documents folder.
_DOCUMENTS = {
    "Documents/LAD Local Authority District names and codes UK as at 04_25.csv": (
        codecs.BOM_UTF8
        + b"LAD25CD,LAD25NM,LAD25NMW\r\n"
        + b"E09000030,Tower Hamlets,\r\nS12000033,Aberdeen City,\r\n"
    ),
    "Documents/PCON Westminster Parliamentary Constituency names and codes UK"
    " as at 12_24.csv": (
        "PCON24CD,PCON24NM,PCON24NMW\r\n" "E14001172,Bethnal Green and Stepney,\r\n"
    ),
    "Documents/ITL125_ITL225_ITL325_LAU125_LAD24_UK_LU.csv": (
        "ITL125CD,ITL125NM,ITL225CD,ITL225NM,ITL325CD,ITL325NM,LAU125CD,LAU125NM,,\r\n"
        "TLI,London,TLI4,Inner London - East,TLI41,Hackney and Newham,"
        "E06000045,Tower Hamlets,,\r\n"
        "TLM,Scotland,TLM5,North Eastern Scotland,TLM50,Aberdeen City,"
        "S30000026,Aberdeen City,,\r\n"
    ),
    "Documents/LSOA (2021) names and codes EW as at 12_21.csv": (
        "LSOA21CD,LSOA21NM\r\nE01000001,City of London 001A\r\n"
    ),
    "Documents/LSOA (2021) DZ names and codes SC as at 12_24.csv": (
        "DZ22CD,DZ22NM\r\nS01013482,Culter - 01\r\n"
    ),
    "Documents/RGN Region names and codes EN as at 05_25.csv": (
        "RGN25CD,RGN25NM,RGN25NMW\r\nE12000007,London,Llundain\r\n"
    ),
}


@pytest.fixture
def directory(make_snapshot: Callable[..., Path]) -> Path:
    """Write a tiny raw postcode directory, as ONS publishes it."""
    return make_snapshot(
        _SOURCE_KEY,
        {
            "Data/ONSPD_AUG_2026_UK.csv": "\r\n".join([_HEADER, *_ROWS]) + "\r\n",
            "Data/multi_csv/ONSPD_AUG_2026_UK_AB.csv": "ignored\r\n",
            "User Guide/ONSPD User Guide Aug 2026.pdf": b"%PDF",
            **_DOCUMENTS,
        },
        archive="zip",
    )


def postcodes(build: Path) -> pl.DataFrame:
    """Return the processed postcodes of a build."""
    return pl.read_parquet(build / "postcodes.parquet")


class TestLoadPostcodeDirectory:
    """`load_postcode_directory` returns the shipped field catalogue."""

    def test_default_fields_are_in_catalogue(self) -> None:
        fields = load_postcode_directory().fields
        assert all(name in fields for name in Config.get("geography.fields"))

    def test_default_fields_are_lad_pcon_and_itl1_with_names(self) -> None:
        assert tuple(Config.get("geography.fields")) == ("lad", "pcon", "itl1")
        assert Config.get("geography.names") is True


class TestProcessOnsPostcodeDirectory:
    """`process_ons_postcode_directory` keeps the chosen areas of postcodes."""

    def test_default_fields_give_codes_and_names(
        self, tmp_path: Path, directory: Path
    ) -> None:
        frame = postcodes(process_ons_postcode_directory(tmp_path))
        assert frame.columns == [
            "postcode",
            "lad_code",
            "lad_name",
            "pcon_code",
            "pcon_name",
            "itl1_code",
            "itl1_name",
        ]
        assert frame.filter(pl.col("postcode") == "E1 6AN").row(0) == (
            "E1 6AN",
            "E09000030",
            "Tower Hamlets",
            "E14001172",
            "Bethnal Green and Stepney",
            "TLI",
            "London",
        )

    def test_itl1_gives_region_for_scotland_too(
        self, tmp_path: Path, directory: Path
    ) -> None:
        frame = postcodes(process_ons_postcode_directory(tmp_path, fields=["itl1"]))
        scotland = frame.filter(pl.col("postcode") == "AB1 0AA")
        assert scotland.select("itl1_code", "itl1_name").row(0) == ("TLM", "Scotland")

    def test_northern_ireland_postcodes_are_left_out_and_counted(
        self, tmp_path: Path, directory: Path
    ) -> None:
        build = process_ons_postcode_directory(tmp_path)
        assert "BT1 1AA" not in postcodes(build)["postcode"].to_list()
        manifest = read_manifest(build, PostcodeManifest)
        assert (manifest.rows, manifest.excluded_rows) == (3, 1)

    def test_postcodes_are_sorted(self, tmp_path: Path, directory: Path) -> None:
        frame = postcodes(process_ons_postcode_directory(tmp_path))
        assert frame["postcode"].to_list() == ["AB1 0AA", "E1 6AN", "ZZ9 9ZZ"]

    def test_pseudo_codes_and_blanks_become_null(
        self, tmp_path: Path, directory: Path
    ) -> None:
        frame = postcodes(process_ons_postcode_directory(tmp_path, fields=["rgn"]))
        regions = dict(zip(frame["postcode"], frame["rgn_code"], strict=True))
        assert regions == {"AB1 0AA": None, "E1 6AN": "E12000007", "ZZ9 9ZZ": None}

    def test_lookup_files_are_combined(self, tmp_path: Path, directory: Path) -> None:
        frame = postcodes(process_ons_postcode_directory(tmp_path, fields=["lsoa21"]))
        names = dict(zip(frame["postcode"], frame["lsoa21_name"], strict=True))
        assert names == {
            "AB1 0AA": "Culter - 01",
            "E1 6AN": "City of London 001A",
            "ZZ9 9ZZ": None,
        }

    def test_fields_without_names_give_one_typed_column(
        self, tmp_path: Path, directory: Path
    ) -> None:
        build = process_ons_postcode_directory(tmp_path, fields=["imd25", "doterm"])
        frame = postcodes(build).sort("postcode")
        assert frame.columns == ["postcode", "imd25", "doterm"]
        assert frame.rows() == [
            ("AB1 0AA", None, date(1990, 6, 1)),
            ("E1 6AN", 26525, None),
            ("ZZ9 9ZZ", None, None),
        ]

    def test_only_chosen_fields_are_kept(self, tmp_path: Path, directory: Path) -> None:
        frame = postcodes(process_ons_postcode_directory(tmp_path, fields=["lad"]))
        assert frame.columns == ["postcode", "lad_code", "lad_name"]

    def test_config_setting_chooses_fields(
        self, tmp_path: Path, directory: Path
    ) -> None:
        Config.set("geography.fields", ["pcon"])
        frame = postcodes(process_ons_postcode_directory(tmp_path))
        assert frame.columns == ["postcode", "pcon_code", "pcon_name"]

    def test_manifest_records_release_columns_and_unnamed_codes(
        self, tmp_path: Path, directory: Path
    ) -> None:
        build = process_ons_postcode_directory(tmp_path, fields=["lad", "pcon"])
        manifest = read_manifest(build, PostcodeManifest)
        assert manifest.release == "AUG_2026"
        assert manifest.fields["lad"].column == "lad26cd"
        assert manifest.fields["pcon"].lookup_files == (
            "PCON Westminster Parliamentary Constituency names and codes UK"
            " as at 12_24.csv",
        )
        # The tiny lookup has no name for Aberdeen's constituency.
        assert manifest.fields["pcon"].unmatched == 1
        assert manifest.fields["lad"].unmatched == 0
        assert manifest.names

    def test_without_names_fields_give_codes_only_and_read_no_lookup(
        self, tmp_path: Path, directory: Path
    ) -> None:
        build = process_ons_postcode_directory(
            tmp_path, fields=["lad", "pcon"], names=False
        )
        frame = postcodes(build)
        assert frame.columns == ["postcode", "lad_code", "pcon_code"]
        assert frame.filter(pl.col("postcode") == "E1 6AN").row(0) == (
            "E1 6AN",
            "E09000030",
            "E14001172",
        )
        manifest = read_manifest(build, PostcodeManifest)
        assert not manifest.names
        assert manifest.fields["lad"].lookup_files == ()

    def test_without_names_itl1_code_still_comes_from_lookup(
        self, tmp_path: Path, directory: Path
    ) -> None:
        build = process_ons_postcode_directory(tmp_path, fields=["itl1"], names=False)
        codes = dict(zip(*postcodes(build).get_columns(), strict=True))
        assert codes == {"AB1 0AA": "TLM", "E1 6AN": "TLI", "ZZ9 9ZZ": None}

    def test_config_setting_turns_names_off(
        self, tmp_path: Path, directory: Path
    ) -> None:
        Config.set("geography.names", False)
        frame = postcodes(process_ons_postcode_directory(tmp_path))
        assert frame.columns == ["postcode", "lad_code", "pcon_code", "itl1_code"]

    def test_no_fields_keep_only_postcodes(
        self, tmp_path: Path, directory: Path
    ) -> None:
        frame = postcodes(process_ons_postcode_directory(tmp_path, fields=[]))
        assert frame.columns == ["postcode"]

    def test_unmatched_code_through_another_code_is_counted(
        self, tmp_path: Path, make_snapshot: Callable[..., Path]
    ) -> None:
        documents = {
            name: content for name, content in _DOCUMENTS.items() if "ITL" not in name
        }
        documents["Documents/ITL125_ITL225_ITL325_LAU125_LAD24_UK_LU.csv"] = (
            "ITL125CD,ITL125NM,LAU125CD\r\nTLI,London,E06000045\r\n"
        )
        make_snapshot(
            _SOURCE_KEY,
            {
                "Data/ONSPD_AUG_2026_UK.csv": "\r\n".join([_HEADER, *_ROWS]),
                **documents,
            },
            archive="zip",
        )
        build = process_ons_postcode_directory(tmp_path, fields=["itl1"])
        # Aberdeen's local administrative unit is not in this lookup.
        assert read_manifest(build, PostcodeManifest).fields["itl1"].unmatched == 1

    def test_unknown_field_raises_key_error_naming_choices(
        self, tmp_path: Path, directory: Path
    ) -> None:
        with pytest.raises(KeyError, match=r"Unknown fields.*expected some of"):
            process_ons_postcode_directory(tmp_path, fields=["district"])

    def test_repeated_field_raises_value_error(
        self, tmp_path: Path, directory: Path
    ) -> None:
        with pytest.raises(ValueError, match="more than once"):
            process_ons_postcode_directory(tmp_path, fields=["lad", "lad"])

    def test_field_without_lookup_file_raises_value_error_and_leaves_nothing(
        self, tmp_path: Path, directory: Path
    ) -> None:
        with pytest.raises(ValueError, match="no lookup for 'wd'"):
            process_ons_postcode_directory(tmp_path, fields=["wd"])
        builds = tmp_path / "processed" / "ons"
        assert not any(path.is_file() for path in builds.rglob("*"))

    def test_field_missing_from_release_raises_value_error(
        self, tmp_path: Path, directory: Path
    ) -> None:
        with pytest.raises(ValueError, match="'cty' column must match"):
            process_ons_postcode_directory(tmp_path, fields=["cty"])

    def test_existing_build_raises_file_exists_error(
        self, tmp_path: Path, directory: Path
    ) -> None:
        process_ons_postcode_directory(tmp_path)
        with pytest.raises(FileExistsError, match="overwrite=True"):
            process_ons_postcode_directory(tmp_path)

    def test_overwrite_with_other_fields_replaces_build(
        self, tmp_path: Path, directory: Path
    ) -> None:
        process_ons_postcode_directory(tmp_path)
        build = process_ons_postcode_directory(tmp_path, fields=["lad"], overwrite=True)
        assert postcodes(build).columns == ["postcode", "lad_code", "lad_name"]
