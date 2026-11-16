# Py-EPC

[![PyPI](https://img.shields.io/pypi/v/py-epc-uk)](https://pypi.org/project/py-epc-uk/)
[![Python](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fdhfwc18%2Fpy-epc%2Fmain%2Fpyproject.toml)](https://pypi.org/project/py-epc-uk/)
[![Licence](https://img.shields.io/pypi/l/py-epc-uk)](https://github.com/dhfwc18/py-epc/blob/main/LICENSE)
[![CI](https://github.com/dhfwc18/py-epc/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/dhfwc18/py-epc/actions/workflows/ci.yml)

Py-EPC downloads the published Energy Performance Certificate (EPC) registers for England and Wales and for Scotland,
processes them into typed Parquet, marks each property's latest certificate, adds the ONS areas of each postcode, and
builds a dataset whose text is cleaned and whose descriptions are parsed into columns. Both registers can be read
together, for Great Britain. Every step is an explicit call, and every result is a lazy [Polars](https://pola.rs) frame.

Py-EPC was inspired by [carbon.place](https://www.carbon.place), by Malcolm Morgan at the University of Leeds. Its
published behaviour informed the design of some modules, but none of its code is used.

## Quick start

Py-EPC is published on [PyPI](https://pypi.org/project/py-epc-uk/) as `py-epc-uk`, and imported as `py_epc`. Install it
with `pip install py-epc-uk`, or `uv add py-epc-uk`.

```python
import polars as pl
import py_epc

root = "data"  # Folder for downloads and builds, created if missing.

# 1. Download the published files, unchanged. England and Wales needs a token from
#    https://get-energy-performance-data.communities.gov.uk; Scotland needs none.
py_epc.download_epc("scotland", "domestic", root)
py_epc.download_ons_postcode_directory(root)

# 2. Process them into typed Parquet.
py_epc.process_epc("scotland", "domestic", root)
py_epc.process_ons_postcode_directory(root)

# 3. Build the dataset, with text cleaned and descriptions parsed into columns.
py_epc.build_epc_dataset("scotland", "domestic", root)

# 4. Read the latest certificate of each property, with its ONS areas.
homes = py_epc.scan_epc("scotland", "domestic", root, stage="dataset", latest=True, geography=True)
homes.group_by("ons_itl1_name").agg(pl.col("lighting_low_energy_percent").mean()).collect()

# 5. Once England and Wales is downloaded, processed and built too, read both for Great Britain.
gb = py_epc.scan_epc("gb", "domestic", root, stage="dataset", latest=True)
gb.group_by("origin", "assessment_method").agg(pl.col("heating_cost_current").median()).collect()
```

## In one call

`run_epc_pipeline` covers every step outlined above for an origin and segment, then reads the result. It reuses the
newest download, build and dataset already made, and only makes what is missing, so running it again is quick.

```python
# The latest certificates of Great Britain, as a lazy frame. England and Wales needs a token the first time.
homes = py_epc.run_epc_pipeline("gb", "domestic", root, latest=True, token="...")

# The same, written to one file, data/outputs/epc-gb-domestic-dataset-latest.parquet, whose path is returned.
path = py_epc.run_epc_pipeline("gb", "domestic", root, latest=True, output_type="parquet")
```

- `output_type` is `"parquet"` or `"sqlite"`. A manifest beside the file, `<name>.manifest.json`, names every download
  it was made from. SQLite stores categories, dates and lists as text, and is much larger, about 1.4 KB a certificate,
  so all of England and Wales is about 42 GB. `output_path` writes the file elsewhere, and `overwrite=True` replaces an
  existing one.
- `stage="processed"` stops before the dataset, and reads the processed builds.
- `update=True` downloads each register again. A download identical to the one already kept is deleted, and its build
  and dataset are reused.
- `clean_up=True`, with an output, deletes the download, build and dataset each register's output was made from, so only
  the output is left. The postcode directory and other snapshots are kept. The next run downloads the registers again.

## What each step does

- **Download** saves each published file unchanged, with a manifest of its source, size and SHA-256, in
  `raw/<source>/<timestamp>/`.
- **Process** checks every yearly file against the register's published columns, parses values into their types, renames
  columns where the two registers mean the same thing, and writes one Parquet file per lodgement year. Values that do
  not parse become null, and are counted in the build's manifest with examples.
- **Latest**: each certificate gets `is_latest`, true for the newest certificate of its property, found by UPRN and, in
  Scotland, the register's building reference.
- **Geography**: the ONS Postcode Directory is processed into the chosen areas of each postcode, with codes and names.
  `scan(..., geography=True)` adds them to each certificate with the prefix `ons_`.
- **Dataset** prepares each lodgement year of a build by the source's dataset definition. England and Wales's bilingual
  values keep their English, and every text column is cleaned. Scotland describes each part of a building, such as an
  extension, so each element is reduced to its main part, with every part kept in `<column>_parts`. Each description is
  then parsed into columns. `scan(..., stage="dataset")` reads the result.
- **Great Britain** reads both registers as one frame, with `scan_epc("gb", ...)`. See [Great Britain](#great-britain).

## Dataset columns

Each parsed description gives columns named after it. The table lists the main ones.

| Published column                | Columns added                                                                      |
| ------------------------------- | ---------------------------------------------------------------------------------- |
| `walls_description`             | `walls_construction`, `walls_insulation`, `walls_u_value`, `walls_assumed`         |
| `roof_description`              | as for walls, with `roof_insulation_mm`                                            |
| `floor_description`             | as for walls                                                                       |
| `mainheat_description`          | `mainheat_system`, `mainheat_emitter`, `mainheat_fuel`, and the same for a second  |
| `hotwater_description`          | `hotwater_system`, `hotwater_cylinder_thermostat`, `hotwater_solar` and others     |
| `lighting_description`          | `lighting_low_energy_percent`, or `lighting_efficiency`                            |
| `construction_age_band`         | `construction_age_from`, `construction_age_to`                                     |
| `floor_level`                   | `floor_level_number`, where 21 means 21 or above, and `floor_level_position`       |
| `main_fuel`                     | `main_fuel` as a category, and `main_fuel_community`                               |
| `air_tightness_description`     | `air_tightness_m3_per_h_m2`, `air_tightness_basis`, `air_tightness_test`, Scotland |
| `photo_supply_description`      | `photo_supply_peak_kw` and the other totals of each dwelling's panels, Scotland    |
| `tenure`, `built_form` and more | the same column, as a category                                                     |
| `lzc_energy_sources`            | the same column, as a list of categories, Scotland                                 |

Each parsed column also gets `<column>_kind`, saying what its phrase turned out to be. Only a `described` phrase gives
values. A placeholder is `not_a_description`, a phrase belonging to another column is `misplaced`, and an impossible
value, such as a share above 100%, is `invalid`. Each of these gives null, so it is never mistaken for a description. A
phrase of no known form is `unknown`, and stops the build, unless it is built with `strict=False`, when it gives null
with a warning. The dataset's manifest counts each column's rows by kind, for every year.

## Great Britain

`scan_epc("gb", ...)` reads the newest build, or dataset, of each register and stacks them. It keeps the columns both
registers have with the same name and type, after an `origin` column of `scotland` or `england-and-wales`. Columns only
one register publishes, such as Scotland's building parts or England and Wales's `constituency`, stay in that register's
own build and dataset. `latest=True` keeps each register's latest certificates, and the ONS areas are added by default,
so every certificate has the same areas whichever register it is from.

Each register's dataset reconciles the columns that differ, so they can be compared. The table shows how.

| Column                           | England and Wales       | Scotland                                          |
| -------------------------------- | ----------------------- | ------------------------------------------------- |
| `assessment_method`              | `report_type`, 2 or 3   | `type_of_assessment`, such as "SAP, new dwelling" |
| `heating_cost_current` and so on | published per year      | a third of the cost published over 3 years        |
| `photo_supply_roof_area_percent` | `photo_supply`, renamed | parsed from `photo_supply_description`            |
| `glazed_area`                    | published as text       | published as a number, parsed to one category     |

`assessment_method` is `rdsap` or `sap`. A processed build has none of these, and leaves out `glazed_area`, so read
Great Britain from the datasets.

## Text and place names

The dataset stage cleans text into ASCII, with `£` the only exception. Part of the text normalisation process removed
accents in place names, such as the circumflex in the Welsh name for Anglesey and the grave accents in many Gaelic place
names. This is a practical choice, so that values work in every tool, and not a judgement on the accents, which are
integral parts of how Welsh and Scottish Gaelic are written and of the heritage of the places they name. The processed
builds keep every value exactly as published, and ONS codes identify each place exactly, so use them wherever the
published spelling matters.

## Settings

Settings are read from `py_epc.Config`, and can be changed, saved and loaded:

```python
py_epc.Config.get("geography.fields")  # ('lad', 'pcon', 'itl1')
py_epc.Config.set("geography.fields", ["lad", "pcon", "itl1", "lsoa21"])
py_epc.Config.set("geography.names", False)  # Codes only, without names.
py_epc.Config.save("py-epc.json")
```

The default areas are the local authority (`lad`), Westminster constituency (`pcon`) and ITL1 region (`itl1`), which
covers the 9 English regions, Wales and Scotland. Changing them needs the postcode directory to be processed again, with
`overwrite=True`, which takes a few seconds.

## Data layout

```text
<root>/
  raw/                                   # Published files, never changed. Ignored by Git.
    epc/scotland/domestic/<timestamp>/
    ons/postcode-directory/<timestamp>/
  processed/                             # Builds, made again from raw at any time. Ignored by Git.
    epc/scotland/domestic/<snapshot>/certificates/lodgement_year=<year>/data.parquet
    ons/postcode-directory/<snapshot>/postcodes.parquet
  datasets/                              # Datasets, made again from processed at any time. Ignored by Git.
    epc/scotland/domestic/<snapshot>/certificates/lodgement_year=<year>/data.parquet
  outputs/                               # Files written by run_epc_pipeline, with their manifests. Ignored by Git.
    epc-gb-domestic-dataset-latest.parquet
    epc-gb-domestic-dataset-latest.manifest.json
```

## Licences

Py-EPC's code is licensed under the [Mozilla Public License 2.0](https://github.com/dhfwc18/py-epc/blob/main/LICENSE).
Py-EPC does not publish any data. The user downloads the data and becomes responsible for its use.
[ATTRIBUTION.md](https://github.com/dhfwc18/py-epc/blob/main/ATTRIBUTION.md) lists every source, its licence, the
attributions it needs, and the restrictions on address data and personal data.
