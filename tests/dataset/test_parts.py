"""Tests for splitting Scotland's values into the parts of a building."""

from collections.abc import Sequence

import polars as pl
import pytest

from py_epc._dataset._parts import (
    BuildingElement,
    misaligned_parts,
    split_building_parts,
)

# The walls of a building, as Scotland publishes them.
_WALLS = BuildingElement(
    name="walls",
    description="walls_description",
    energy_eff="walls_energy_eff",
    env_eff="walls_env_eff",
)


def walls(
    descriptions: Sequence[str | None],
    energy: Sequence[str | None],
    environment: Sequence[str | None] | None = None,
) -> pl.LazyFrame:
    """Return rows of walls, whose environmental ratings default to energy."""
    return pl.LazyFrame(
        {
            "walls_description": list(descriptions),
            "walls_energy_eff": list(energy),
            "walls_env_eff": list(energy if environment is None else environment),
        },
        schema=dict.fromkeys(
            ["walls_description", "walls_energy_eff", "walls_env_eff"], pl.String
        ),
    )


class TestSplitBuildingParts:
    """`split_building_parts` keeps an element's main part, and all parts."""

    def test_single_part_is_main_part(self) -> None:
        frame = split_building_parts(walls(["Cavity wall"], ["Good"]), _WALLS)
        assert frame.select(
            "walls_description", "walls_energy_eff", "walls_part_count"
        ).collect().row(0) == ("Cavity wall", "Good", 1)

    def test_first_part_is_main_part_with_its_ratings(self) -> None:
        frame = walls(
            ["Solid brick | Timber frame"], ["Poor | Good"], ["Poor | Average"]
        )
        result = split_building_parts(frame, _WALLS).collect()
        assert result.select(
            "walls_description", "walls_energy_eff", "walls_env_eff"
        ).row(0) == ("Solid brick", "Poor", "Poor")

    @pytest.mark.parametrize(
        "description",
        [
            pytest.param(" | Solid brick | Timber frame", id="published"),
            pytest.param("| Solid brick | Timber frame", id="trimmed"),
        ],
    )
    def test_empty_first_part_makes_next_part_main(self, description: str) -> None:
        frame = walls([description], [" | Poor | Good"])
        result = split_building_parts(frame, _WALLS).collect()
        assert result.select("walls_description", "walls_energy_eff").row(0) == (
            "Solid brick",
            "Poor",
        )

    def test_every_part_is_kept_in_order(self) -> None:
        frame = walls([" | Solid brick | Timber frame"], [" | Poor | Good"])
        result = split_building_parts(frame, _WALLS).collect()
        assert result.select(
            "walls_description_parts", "walls_energy_eff_parts", "walls_part_count"
        ).row(0) == ([None, "Solid brick", "Timber frame"], [None, "Poor", "Good"], 3)

    def test_missing_description_stays_null(self) -> None:
        result = split_building_parts(walls([None], [None]), _WALLS).collect()
        assert result.select("walls_description", "walls_part_count").row(0) == (
            None,
            None,
        )

    def test_rows_keep_their_order(self) -> None:
        descriptions = [f"Part {n} | Extension {n}" for n in range(20)]
        frame = walls(descriptions, ["Good | Poor"] * 20)
        result = split_building_parts(frame, _WALLS).collect()
        assert result["walls_description"].to_list() == [f"Part {n}" for n in range(20)]


class TestMisalignedParts:
    """`misaligned_parts` counts certificates whose parts do not line up."""

    def test_aligned_parts_count_nothing(self) -> None:
        frame = walls(["A | B", "C"], ["Good | Poor", "Good"])
        assert misaligned_parts(frame, _WALLS).model_dump() == {
            "different_counts": 0,
            "main_part_without_ratings": 0,
        }

    def test_different_numbers_of_parts_are_counted(self) -> None:
        frame = walls(["A | B", "C | D"], ["Good", "Good | Poor"])
        assert misaligned_parts(frame, _WALLS).different_counts == 1

    def test_main_part_without_ratings_is_counted(self) -> None:
        frame = walls(["A", "B"], [None, "Good"], [None, "Good"])
        assert misaligned_parts(frame, _WALLS).main_part_without_ratings == 1
