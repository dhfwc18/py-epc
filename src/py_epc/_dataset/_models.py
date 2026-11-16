"""Settings shared by the models of the dataset stage."""

from pydantic import ConfigDict

# Models are read-only, reject unknown keys and never coerce types. Errors
# leave out the input, which may be a published value.
MODEL_CONFIG = ConfigDict(
    frozen=True,
    extra="forbid",
    strict=True,
    hide_input_in_errors=True,
    use_attribute_docstrings=True,
)
