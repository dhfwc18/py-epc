"""Tests for dataset definitions."""

import json
from typing import Any, get_args

import pytest
from pydantic import TypeAdapter, ValidationError

from py_epc._dataset._definition import (
    DatasetDefinition,
    FixedParserDefinition,
    ParserDefinition,
    VocabularyParserDefinition,
    build_parser,
    load_dataset_definition,
)
from py_epc._process._register import load_register

# Keys of the sources that have a shipped dataset definition.
_SHIPPED_KEYS = ["epc.scotland.domestic", "epc.england-and-wales.domestic"]

# Fields of a minimal valid definition, with one parser.
_DEFINITION: dict[str, Any] = {
    "version": 1,
    "source_key": "epc.scotland.domestic",
    "parsers": [{"column": "lighting_description", "parser": "lighting"}],
}

# Reads one parser definition from JSON, as shipped ones are.
_PARSER_DEFINITION: TypeAdapter[ParserDefinition] = TypeAdapter(ParserDefinition)


def parse(fields: dict[str, Any]) -> DatasetDefinition:
    """Return the definition for `fields`, read from JSON as when shipped."""
    return DatasetDefinition.model_validate_json(json.dumps(fields))


class TestDatasetDefinition:
    """`DatasetDefinition` checks a definition when it is read."""

    def test_minimal_definition_returns_definition(self) -> None:
        definition = parse(_DEFINITION)
        assert definition.parsers == (
            FixedParserDefinition(column="lighting_description", parser="lighting"),
        )

    def test_vocabulary_parser_returns_vocabulary_parser_definition(self) -> None:
        definition = _PARSER_DEFINITION.validate_json(
            json.dumps(
                {
                    "column": "walls_description",
                    "parser": "fabric",
                    "vocabulary": "walls",
                }
            )
        )
        assert definition == VocabularyParserDefinition(
            column="walls_description", parser="fabric", vocabulary="walls"
        )

    @pytest.mark.parametrize(
        "parser",
        [
            pytest.param(
                {"column": "walls_description", "parser": "fabric"},
                id="vocabulary-parser-without-vocabulary",
            ),
            pytest.param(
                {
                    "column": "lighting_description",
                    "parser": "lighting",
                    "vocabulary": "lighting",
                },
                id="fixed-parser-with-vocabulary",
            ),
            pytest.param(
                {"column": "walls_description", "parser": "walls"},
                id="unknown-parser",
            ),
        ],
    )
    def test_invalid_parser_raises_validation_error(
        self, parser: dict[str, str]
    ) -> None:
        with pytest.raises(ValidationError):
            parse({**_DEFINITION, "parsers": [parser]})

    def test_no_parsers_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            parse({**_DEFINITION, "parsers": []})

    @pytest.mark.parametrize(
        ("field", "values", "what"),
        [
            pytest.param(
                "bilingual_columns",
                ["walls_description", "walls_description"],
                "bilingual column",
                id="bilingual-column",
            ),
            pytest.param(
                "building_elements",
                [
                    {
                        "name": "walls",
                        "description": "walls_description",
                        "energy_eff": "walls_energy_eff",
                        "env_eff": "walls_env_eff",
                    }
                ]
                * 2,
                "building element",
                id="building-element",
            ),
            pytest.param(
                "parsers",
                _DEFINITION["parsers"] * 2,
                "parsed column",
                id="parsed-column",
            ),
            pytest.param(
                "renamed_columns",
                {"a": "same", "b": "same"},
                "new column name",
                id="new-column-name",
            ),
        ],
    )
    def test_repeated_name_raises_validation_error_naming_it(
        self, field: str, values: list[Any], what: str
    ) -> None:
        with pytest.raises(ValidationError, match=f"{what}s must be unique"):
            parse({**_DEFINITION, field: values})


class TestLoadDatasetDefinition:
    """`load_dataset_definition` reads a shipped definition."""

    @pytest.mark.parametrize("key", _SHIPPED_KEYS)
    def test_shipped_key_returns_definition_of_source(self, key: str) -> None:
        assert load_dataset_definition(key).source_key == key

    def test_unknown_key_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="no dataset definition"):
            load_dataset_definition("epc.scotland.non-domestic")


class TestBuildParser:
    """`build_parser` returns the parser a definition names."""

    @pytest.mark.parametrize(
        "parser", get_args(FixedParserDefinition.model_fields["parser"].annotation)
    )
    def test_fixed_parser_returns_parser_of_its_column(self, parser: str) -> None:
        shipped = {
            p.parser: p.column
            for key in _SHIPPED_KEYS
            for p in load_dataset_definition(key).parsers
        }
        definition = FixedParserDefinition.model_validate(
            {"column": shipped[parser], "parser": parser}
        )
        assert build_parser(definition).column == shipped[parser]

    def test_other_column_raises_value_error(self) -> None:
        definition = FixedParserDefinition(
            column="walls_description", parser="lighting"
        )
        with pytest.raises(ValueError, match="reads 'lighting_description'"):
            build_parser(definition)

    def test_vocabulary_of_other_format_raises_value_error(self) -> None:
        definition = VocabularyParserDefinition(
            column="lzc_energy_sources",
            parser="category",
            vocabulary="lzc_energy_sources",
        )
        with pytest.raises(ValueError, match="not 'phrase'"):
            build_parser(definition)

    def test_unknown_vocabulary_raises_key_error(self) -> None:
        definition = VocabularyParserDefinition(
            column="walls_description", parser="fabric", vocabulary="wall"
        )
        with pytest.raises(KeyError, match="no vocabulary named 'wall'"):
            build_parser(definition)


class TestShippedDatasetDefinitions:
    """The shipped definitions fit their sources' registers."""

    @pytest.mark.parametrize("key", _SHIPPED_KEYS)
    def test_named_columns_are_in_register(self, key: str) -> None:
        definition = load_dataset_definition(key)
        named = [
            *definition.bilingual_columns,
            *(
                column
                for element in definition.building_elements
                for column in (element.description, element.energy_eff, element.env_eff)
            ),
            *(parser.column for parser in definition.parsers),
        ]
        columns = {column.name for column in load_register(key).columns}
        assert [name for name in named if name not in columns] == []

    @pytest.mark.parametrize("key", _SHIPPED_KEYS)
    def test_bilingual_columns_are_text(self, key: str) -> None:
        types = {column.name: column.type for column in load_register(key).columns}
        definition = load_dataset_definition(key)
        assert {types[name] for name in definition.bilingual_columns} <= {"text"}

    @pytest.mark.parametrize("key", _SHIPPED_KEYS)
    def test_parsers_build_with_outputs_new_to_register(self, key: str) -> None:
        columns = {column.name for column in load_register(key).columns}
        added: list[str] = []
        for definition in load_dataset_definition(key).parsers:
            parser = build_parser(definition)
            outputs = [output.name for output in parser.outputs]
            added.extend(outputs[1:] if parser.replaces_column else outputs)
        assert [name for name in added if name in columns] == []
        assert len(set(added)) == len(added)

    def test_outputs_given_by_both_have_one_type(self) -> None:
        dtypes = [
            {
                output.name: output.dtype
                for definition in load_dataset_definition(key).parsers
                for output in build_parser(definition).outputs
            }
            for key in _SHIPPED_KEYS
        ]
        shared = dtypes[0].keys() & dtypes[1].keys()
        assert "assessment_method" in shared
        assert {
            name: (dtypes[0][name], dtypes[1][name])
            for name in shared
            if dtypes[0][name] != dtypes[1][name]
        } == {}

    @pytest.mark.parametrize("key", _SHIPPED_KEYS)
    def test_renamed_columns_are_in_register_and_new_names_are_not(
        self, key: str
    ) -> None:
        columns = {column.name for column in load_register(key).columns}
        renamed = load_dataset_definition(key).renamed_columns
        assert set(renamed) <= columns
        assert set(renamed.values()) & columns == set()
