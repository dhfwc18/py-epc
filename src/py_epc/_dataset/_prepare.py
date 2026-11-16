"""Prepare one year of a processed build by its dataset definition."""

from typing import Final

import polars as pl

from py_epc._dataset._bilingual import keep_english_text
from py_epc._dataset._construction_age import null_construction_age_after_lodgement
from py_epc._dataset._definition import DatasetDefinition, build_parser
from py_epc._dataset._parts import split_building_parts
from py_epc._dataset._phrases import parse_columns
from py_epc._dataset._text import normalise_text_columns

# Suffix of the column giving the kind of each row's phrase, after the name
# of the parsed column, such as "walls_description_kind".
KIND_COLUMN_SUFFIX: Final = "_kind"


def prepare_certificates(
    frame: pl.LazyFrame, definition: DatasetDefinition, *, strict: bool = True
) -> pl.LazyFrame:
    """
    Return one year of certificates, prepared by `definition`.

    The steps run in this order:

    1. Bilingual columns are reduced to their English.
    2. Every text column is cleaned.
    3. Each building element is reduced to its main part.
    4. Each column of phrases is parsed, and ``<column>_kind`` gives the kind
       of each row's phrase.
    5. A construction age starting after the lodgement date is set to null.
    6. Columns given new names by the definition are renamed.

    The cleaning and parsing steps hold every distinct value of their
    columns in memory, so give one year at a time.

    Parameters
    ----------
    frame : polars.LazyFrame
        One year of a processed build of `DatasetDefinition.source_key`.
    definition : DatasetDefinition
        How to prepare the source's certificates.
    strict : bool, default True
        Whether an odd bilingual value, a character with no ASCII form or an
        unknown phrase raises an error. If false, each gives null, or is
        removed, with a warning.

    Returns
    -------
    polars.LazyFrame
        The certificates, in the same order.

    Raises
    ------
    ValueError
        If `strict` is true and a value cannot be read, as above.
    """
    if definition.bilingual_columns:
        frame = keep_english_text(frame, definition.bilingual_columns, strict=strict)
    text_columns = [
        name for name, dtype in frame.collect_schema().items() if dtype == pl.String()
    ]
    frame = normalise_text_columns(frame, text_columns, strict=strict)
    for element in definition.building_elements:
        frame = split_building_parts(frame, element)
    frame = parse_columns(
        frame,
        [build_parser(parser) for parser in definition.parsers],
        strict=strict,
        kind_columns={
            parser.column: f"{parser.column}{KIND_COLUMN_SUFFIX}"
            for parser in definition.parsers
        },
    )
    if any(parser.parser == "construction_age" for parser in definition.parsers):
        frame = null_construction_age_after_lodgement(frame)
    return frame.rename(definition.renamed_columns)
