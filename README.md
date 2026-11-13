# Py-EPC

Py-EPC downloads the published Energy Performance Certificate (EPC) registers for England and Wales and for Scotland,
processes them into typed Parquet, marks each property's latest certificate, and adds the ONS areas of each postcode.
Every step is an explicit call, and every result is a lazy [Polars](https://pola.rs) frame.

Py-EPC was inspired by [carbon.place](https://www.carbon.place), by Malcolm Morgan at the University of Leeds. Its
published behaviour informed the design of some modules, but none of its code is used.

## Quick start

Py-EPC is not yet on PyPI. Install it from a clone with `uv sync`.

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

# 3. Read the latest certificate of each property, with its ONS areas.
homes = py_epc.scan_epc("scotland", "domestic", root, latest=True, geography=True)
homes.group_by("ons_itl1_name").agg(pl.col("current_energy_efficiency").mean()).collect()
```

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
```

## Resources

Measured on a 24-core machine with 15 GB of memory:

| Source                     | Rows         | Processing | Peak memory | Build  |
| -------------------------- | ------------ | ---------- | ----------- | ------ |
| Scotland domestic          | 3.8 million  | 1 minute   | 2.9 GB      | 0.5 GB |
| England and Wales domestic | 30.5 million | 3 minutes  | 4.5 GB      | 2.9 GB |
| ONS Postcode Directory     | 2.7 million  | 5 seconds  | 2.3 GB      | 4 MB   |

## Licences

Py-EPC's code is licensed under the [Mozilla Public License 2.0](LICENSE). Py-EPC does not publish any data. The user
downloads the data and becomes responsible for its use. [ATTRIBUTION.md](ATTRIBUTION.md) lists every source, its
licence, the attributions it needs, and the restrictions on address data and personal data.
