"""Read each source's dataset definition and build the parsers it names."""

from collections import Counter
from collections.abc import Iterable
from importlib.resources import files
from typing import Annotated, Any, Final, Literal, Self

from pydantic import BaseModel, Field, model_validator

from py_epc._dataset._air_tightness import air_tightness_parser
from py_epc._dataset._categories import (
    PhraseVocabulary,
    category_list_parser,
    category_parser,
)
from py_epc._dataset._construction_age import construction_age_parser
from py_epc._dataset._fabric import FabricVocabulary, fabric_parser
from py_epc._dataset._floor_level import floor_level_parser
from py_epc._dataset._lighting import lighting_parser
from py_epc._dataset._main_fuel import main_fuel_parser
from py_epc._dataset._models import MODEL_CONFIG
from py_epc._dataset._parts import BuildingElement
from py_epc._dataset._photo_supply import photo_supply_parser
from py_epc._dataset._phrases import ColumnParser
from py_epc._dataset._systems import SystemVocabulary, system_parser
from py_epc._dataset._vocabulary import load_vocabulary

# Folder in the package holding the definitions, one JSON file per source.
DEFINITIONS_FOLDER: Final = "_definitions"


class VocabularyParserDefinition(BaseModel):
    """A column parsed with the phrases of a shipped vocabulary."""

    model_config = MODEL_CONFIG

    column: str
    """Column of phrases, such as ``"walls_description"``."""

    parser: Literal["category", "category_list", "fabric", "system", "main_fuel"]
    """Parser of the column, named after its ``<parser>_parser`` function."""

    vocabulary: str
    """Name of the vocabulary, such as ``"walls"``."""


class FixedParserDefinition(BaseModel):
    """A column parsed with a short fixed set of phrases, held in code."""

    model_config = MODEL_CONFIG

    column: str
    """Column of phrases, such as ``"lighting_description"``."""

    parser: Literal[
        "lighting", "air_tightness", "construction_age", "floor_level", "photo_supply"
    ]
    """Parser of the column, named after its ``<parser>_parser`` function."""


# How one column is parsed, told apart by the parser it names.
type ParserDefinition = Annotated[
    VocabularyParserDefinition | FixedParserDefinition, Field(discriminator="parser")
]


class DatasetDefinition(BaseModel):
    """
    How to build one source's dataset from its processed build.

    England and Wales's bilingual columns are first put in English, or
    Scotland's building elements reduced to their main part. Every text
    column of the register is then cleaned, and each column of phrases is
    parsed. Last, some columns are renamed, to match the other register.
    """

    model_config = MODEL_CONFIG

    version: Literal[1]
    """Version of the definition format."""

    source_key: str
    """Config key of the source, such as ``"epc.scotland.domestic"``."""

    bilingual_columns: tuple[str, ...] = ()
    """
    Columns that may be published in English and Welsh, kept in English
    by `keep_english_text`.
    """

    building_elements: tuple[BuildingElement, ...] = ()
    """Elements described part by part, split by `split_building_parts`."""

    parsers: tuple[ParserDefinition, ...] = Field(min_length=1)
    """How each column of phrases is parsed, in order."""

    renamed_columns: dict[str, str] = {}
    """
    New name of each published column renamed, keyed by its published name,
    so it has the name the other register's matching column has.
    """

    @model_validator(mode="after")
    def _check(self) -> Self:
        _check_unique("bilingual column", self.bilingual_columns)
        _check_unique("building element", (e.name for e in self.building_elements))
        _check_unique("parsed column", (p.column for p in self.parsers))
        _check_unique("new column name", self.renamed_columns.values())
        return self


def load_dataset_definition(key: str) -> DatasetDefinition:
    """
    Return the shipped dataset definition for a source.

    Parameters
    ----------
    key : str
        Config key of the source, such as ``"epc.scotland.domestic"``.

    Returns
    -------
    DatasetDefinition
        The definition, checked when loaded.

    Raises
    ------
    KeyError
        If no definition is shipped for the source.
    pydantic.ValidationError
        If the shipped definition is not valid.
    ValueError
        If the shipped definition names a different source.
    """
    path = (
        files("py_epc._dataset") / DEFINITIONS_FOLDER / f"{key.replace('.', '-')}.json"
    )
    if not path.is_file():
        raise KeyError(f"no dataset definition for {key!r}")
    definition = DatasetDefinition.model_validate_json(path.read_bytes())
    if definition.source_key != key:
        raise ValueError(
            f"dataset definition for {key!r} names {definition.source_key!r}"
        )
    return definition


def build_parser(definition: ParserDefinition) -> ColumnParser[Any]:
    """
    Return the parser a definition names, with its vocabulary loaded.

    Parameters
    ----------
    definition : VocabularyParserDefinition or FixedParserDefinition
        How the column is parsed.

    Returns
    -------
    ColumnParser
        A parser for `parse_column`.

    Raises
    ------
    KeyError
        If the vocabulary is not shipped.
    ValueError
        If the parser reads a column other than the one defined, or the
        vocabulary is not of the format the parser reads.
    """
    if isinstance(definition, VocabularyParserDefinition):
        parser = _vocabulary_parser(definition)
    else:
        parser = _fixed_parser(definition)
    if parser.column != definition.column:
        raise ValueError(
            f"{definition.parser} parser reads {parser.column!r},"
            f" not {definition.column!r}"
        )
    return parser


def _vocabulary_parser(definition: VocabularyParserDefinition) -> ColumnParser[Any]:
    """Return the parser of a column parsed with a vocabulary."""
    name = definition.vocabulary
    match definition.parser:
        case "category":
            return category_parser(load_vocabulary(name, PhraseVocabulary))
        case "category_list":
            return category_list_parser(load_vocabulary(name, PhraseVocabulary))
        case "fabric":
            return fabric_parser(load_vocabulary(name, FabricVocabulary))
        case "system":
            return system_parser(load_vocabulary(name, SystemVocabulary))
        case "main_fuel":
            return main_fuel_parser(load_vocabulary(name, PhraseVocabulary))


def _fixed_parser(definition: FixedParserDefinition) -> ColumnParser[Any]:
    """Return the parser of a column parsed with a fixed set of phrases."""
    match definition.parser:
        case "lighting":
            return lighting_parser()
        case "air_tightness":
            return air_tightness_parser()
        case "construction_age":
            return construction_age_parser()
        case "floor_level":
            return floor_level_parser()
        case "photo_supply":
            return photo_supply_parser()


def _check_unique(what: str, names: Iterable[str]) -> None:
    """Raise ValueError naming every value of `names` that repeats."""
    repeated = sorted(name for name, count in Counter(names).items() if count > 1)
    if repeated:
        raise ValueError(f"{what}s must be unique; repeated: {repeated}")
