"""Read the shipped vocabularies, which list the phrases of a column."""

import re
from importlib.resources import files
from typing import Final, Literal

from pydantic import BaseModel, PrivateAttr

from py_epc._dataset._models import MODEL_CONFIG

# Folder in the package holding the vocabularies, one JSON file each.
VOCABULARIES_FOLDER: Final = "_vocabularies"

# A category key: snake case, such as "fully_double".
_CATEGORY_KEY_PATTERN: Final = re.compile(r"[a-z][a-z0-9_]*")


class Category(BaseModel):
    """One category, such as a glazing, and the phrases that mean it."""

    model_config = MODEL_CONFIG

    label: str
    """Readable name, such as ``"Fully double glazed"``."""

    phrases: tuple[str, ...]
    """Published phrases that mean the category, in any case."""


class Vocabulary(BaseModel):
    """
    What every vocabulary has: its column, and phrases it describes none of.

    Each kind of vocabulary adds a ``format``, naming the kind, and the
    phrases it describes.
    """

    model_config = MODEL_CONFIG

    version: Literal[1]
    """Version of the vocabulary format."""

    column: str
    """Column of phrases described, such as ``"windows_description"``."""

    misplaced: tuple[str, ...] = ()
    """Phrases, or their first parts before a comma, that belong to another
    column, such as a floor's construction published as a window's."""

    not_a_description: tuple[str, ...] = ()
    """Phrases that describe nothing, such as placeholders."""

    _misplaced_folded: frozenset[str] = PrivateAttr()
    _not_a_description_folded: frozenset[str] = PrivateAttr()

    def model_post_init(self, context: object, /) -> None:
        self._misplaced_folded = frozenset(p.casefold() for p in self.misplaced)
        self._not_a_description_folded = frozenset(
            p.casefold() for p in self.not_a_description
        )

    def undescribed_kind(
        self, phrase: str
    ) -> Literal["misplaced", "not_a_description"] | None:
        """
        Return the kind of a phrase this vocabulary describes none of.

        Parameters
        ----------
        phrase : str
            A phrase, already readied by `prepare_phrase`.

        Returns
        -------
        str or None
            ``"not_a_description"`` or ``"misplaced"``, matched in any case,
            or None if the phrase is neither.
        """
        folded = phrase.casefold()
        if folded in self._not_a_description_folded:
            return "not_a_description"
        first_part = folded.partition(",")[0].strip()
        if self._misplaced_folded & {folded, first_part}:
            return "misplaced"
        return None


def check_categories(categories: dict[str, Category], part: str) -> None:
    """
    Raise ValueError if a key is not snake case or a phrase is repeated.

    Parameters
    ----------
    categories : dict[str, Category]
        Categories, keyed by name.
    part : str
        What the categories describe, such as ``"glazing"``, named in errors.
    """
    seen: dict[str, str] = {}
    for key, category in categories.items():
        if not _CATEGORY_KEY_PATTERN.fullmatch(key):
            raise ValueError(f"{part} key {key!r} is not snake case")
        for phrase in category.phrases:
            folded = phrase.casefold()
            if folded in seen:
                raise ValueError(
                    f"{part} phrase {phrase!r} is in {seen[folded]!r} and {key!r}"
                )
            seen[folded] = key


def categories_by_phrase(categories: dict[str, Category]) -> dict[str, str]:
    """Return the key of each category, by each of its phrases casefolded."""
    return {
        phrase.casefold(): key
        for key, category in categories.items()
        for phrase in category.phrases
    }


def load_vocabulary[V: Vocabulary](name: str, model: type[V]) -> V:
    """
    Return the shipped vocabulary named `name`, read as `model`.

    Parameters
    ----------
    name : str
        Name of the vocabulary, such as ``"windows"``.
    model : type of Vocabulary
        Kind of vocabulary, such as `PhraseVocabulary`.

    Returns
    -------
    Vocabulary
        The vocabulary, checked.

    Raises
    ------
    KeyError
        If no vocabulary of that name is shipped.
    pydantic.ValidationError
        If the file is not a valid vocabulary of that kind.
    """
    path = files("py_epc._dataset") / VOCABULARIES_FOLDER / f"{name}.json"
    if not path.is_file():
        raise KeyError(f"no vocabulary named {name!r}")
    return model.model_validate_json(path.read_bytes())
