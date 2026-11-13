# Contributing

## Setup and checks

Py-EPC uses [uv](https://docs.astral.sh/uv/). Set up a clone with:

```bash
uv sync
uv run pre-commit install
```

CI runs these three checks, then builds the wheel. Every commit should pass them:

```bash
uv run pre-commit run --all-files   # ruff, black, prettier, markdownlint, ASCII
uv run pyright src tests
uv run pytest
```

## Where things live

Modules are private, with a leading underscore. The public API is what `src/py_epc/__init__.py` exports in `__all__`.

| Module                   | Holds                                                                  |
| ------------------------ | ---------------------------------------------------------------------- |
| `_config.py`             | `Config`, the settings checked against a schema                        |
| `defaults.json`          | The shipped default settings                                           |
| `_source.py`             | `Source`, one published file and where to download it                  |
| `_enums.py`              | Names used in config keys, and `parse_member`                          |
| `_download.py`           | The download functions                                                 |
| `_manifest.py`           | Manifest models, reading and writing them, and `find_completed`        |
| `_files.py`              | Safe folder writes and `.gitignore`                                    |
| `_scan.py`               | `scan` and `scan_epc`, which read processed builds                     |
| `_process/_archive.py`   | Finding a raw snapshot and reading its archive's members one at a time |
| `_process/_register.py`  | Register definitions: how to read, type and name a register's columns  |
| `_process/_registers/`   | The shipped definitions, one JSON file per source                      |
| `_process/_values.py`    | Parsing raw text into each column's type                               |
| `_process/_build.py`     | `process` and `process_epc`, which write typed Parquet builds          |
| `_process/_latest.py`    | Marking each property's latest certificate                             |
| `_process/_postcodes.py` | Processing the ONS Postcode Directory, and `normalise_postcode`        |

A source's columns, types and names belong in its JSON definition, not in code. The code reads the definition.

## Code

- **Style**: black at 88 characters, and ruff with docstrings at 79. Docstrings follow the
  [numpy convention](https://numpydoc.readthedocs.io/en/latest/format.html).
- **Types**: every function is typed, and pyright must pass in standard mode.
- **Text**: files are ASCII, except for `£`. In Markdown, write `&copy;` for the copyright sign. Spelling is British,
  with "licence" as the noun.
- **Constants**: say what a name holds, with a suffix such as `_COLUMN`, `_FILE` or `_FOLDER`.
- **Memory**: work one year at a time, stream with `sink_parquet`, and join with small tables rather than using
  `replace`. Measure peak memory on the real files before relying on a new step.

## Data

- Never commit data. `raw/` and `processed/` are ignored by Git, and tests never download anything.
- Raw files are never changed. Every later step can be deleted and made again from them.
- Address columns are restricted by their licence and are marked `restricted` in each definition. See
  [ATTRIBUTION.md](ATTRIBUTION.md).

## Commits

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org): a type, an optional scope and a
past-tense summary, such as `feat(process): added value parsers for register column types`. Use `refactor` for a change
that keeps behaviour, `perf` for speed or memory, `docs`, `test` and `chore` for the rest, and mark a breaking change
with `!`.

## Licence

Py-EPC is licensed under the [Mozilla Public License 2.0](LICENSE), and contributions are accepted under the same
licence.

## Tests

Tests use pytest and run offline with tiny fixtures. Run them with `uv run pytest`.

### Layout

- One test file per source module, mirroring the source tree: `src/py_epc/_config.py` is tested in
  `tests/test_config.py`, and `src/py_epc/_process/_values.py` in `tests/process/test_values.py`.
- A fixture used by one file stays in that file. A fixture used by several files lives in the `conftest.py` of the
  closest folder that holds them all, such as `tests/process/conftest.py`.

### Names

The naming follows Roy Osherove's
[naming standard for unit tests](https://osherove.com/blog/2005/4/3/naming-standards-for-unit-tests.html),
`UnitOfWork_StateUnderTest_ExpectedBehavior`, split between the class and the method:

```text
Test<Subject>::test_<condition>_<outcome>[<case>]
```

- **Subject**: the class names the one thing under test, as the code names it. `Config.get` is tested in `TestConfigGet`
  and `defaults.json` in `TestDefaultsJson`. Its docstring says what the subject does.
- **Condition**: the input or state that matters, as a noun phrase, such as `unknown_key`, `changed_field` or
  `existing_file_with_overwrite`. Every test names one, even the plain case, such as `field_key`.
- **Outcome**: what should happen, as a present-tense verb phrase, such as `returns_value`, `writes_only_version` or
  `leaves_config_unchanged`. A raised error is named in full: `raises_key_error`, `raises_validation_error`.
- **One behaviour per test**. Use `and` only when both parts describe one behaviour, as in
  `raises_validation_error_and_changes_nothing`, where the rejection is the behaviour.
- **Case**: a parametrized test names its shared condition, and each `pytest.param` has a kebab-case `id` naming the
  case, such as `unknown_key_raises_key_error[unknown-origin]`.

A test reads as a sentence: `TestConfigLoad::test_missing_file_raises_file_not_found_error` reads "`Config.load`, given
a missing file, raises `FileNotFoundError`".
