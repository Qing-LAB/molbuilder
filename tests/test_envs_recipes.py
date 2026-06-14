"""L1 tests for ``molbuilder.envs.recipes``.

Pins the registry shape: every built-in recipe must have non-empty
required fields, every routed recipe must reference a valid category
from ``DEFAULT_ENV_NAMES``, and the name-keyed + category-keyed
lookups must be consistent with the BUILTIN_RECIPES tuple.

The recipes module is the single source of truth for env shape;
these tests catch a stray "oh I'll change `name=` to a different
string for a moment" type regression that would silently disconnect
the recipe from the diagnostics-side category mapping.
"""
from __future__ import annotations

import pytest

from molbuilder.diagnostics import DEFAULT_ENV_NAMES
from molbuilder.envs.recipes import (
    BUILTIN_RECIPES,
    Recipe,
    recipe_by_name,
    recipe_for_category,
)


def test_registry_is_non_empty():
    assert len(BUILTIN_RECIPES) >= 5, (
        "Registry should hold at least the five envs documented in "
        "README_install.md (host + 4 backends)."
    )


def test_every_recipe_has_required_fields():
    """Every recipe must have non-empty name, description, channels,
    and conda_packages.  Empty values would render the recipe
    unusable."""
    for r in BUILTIN_RECIPES:
        assert r.name, f"empty name in recipe {r}"
        assert r.description, f"empty description in {r.name}"
        assert r.channels, f"no channels in {r.name}"
        assert r.conda_packages, f"no conda_packages in {r.name}"


def test_recipe_names_are_unique():
    names = [r.name for r in BUILTIN_RECIPES]
    assert len(names) == len(set(names)), (
        f"duplicate recipe names: {names}"
    )


def test_routed_recipes_match_default_env_names():
    """Every recipe with ``category`` set must have a name matching
    ``DEFAULT_ENV_NAMES[category]``.  Otherwise ``run_tool`` would
    dispatch to a different env name than ``install`` builds."""
    for r in BUILTIN_RECIPES:
        if r.category is None:
            continue
        assert r.category in DEFAULT_ENV_NAMES, (
            f"recipe `{r.name}` has unknown category `{r.category}`. "
            f"Add it to DEFAULT_ENV_NAMES in molbuilder/diagnostics.py "
            f"or fix the recipe."
        )
        assert DEFAULT_ENV_NAMES[r.category] == r.name, (
            f"recipe `{r.name}` declares category `{r.category}` but "
            f"DEFAULT_ENV_NAMES says category maps to "
            f"`{DEFAULT_ENV_NAMES[r.category]}`.  These must agree."
        )


def test_default_env_names_all_have_recipes():
    """Reverse direction: every category in ``DEFAULT_ENV_NAMES``
    must have a recipe.  Without one, ``envs install <category-env>``
    would have no plan."""
    covered = {r.category for r in BUILTIN_RECIPES if r.category}
    missing = set(DEFAULT_ENV_NAMES) - covered
    assert not missing, (
        f"DEFAULT_ENV_NAMES has categories with no Recipe: {missing}. "
        f"Add Recipe entries in molbuilder/envs/recipes.py."
    )


def test_recipe_by_name_finds_every_recipe():
    for r in BUILTIN_RECIPES:
        assert recipe_by_name(r.name) is r


def test_recipe_by_name_returns_none_for_unknown():
    assert recipe_by_name("nope-not-a-real-env") is None


def test_recipe_for_category_finds_every_routed_recipe():
    for r in BUILTIN_RECIPES:
        if r.category is None:
            continue
        assert recipe_for_category(r.category) is r


def test_recipe_for_category_returns_none_for_unknown():
    assert recipe_for_category("not-a-category") is None


def test_recipes_are_frozen():
    """Recipe is a frozen dataclass -- mutating a recipe should
    raise.  Important because we share the BUILTIN_RECIPES tuple
    across process state (tests + production)."""
    r = BUILTIN_RECIPES[0]
    with pytest.raises(Exception):
        # FrozenInstanceError on python 3.10+; on older it's
        # AttributeError.  Either way: not assignable.
        r.name = "mutated"   # type: ignore[misc]


def test_verify_ignore_exit_code_field_present():
    """The MDtools recipe needs this field set to True (tleap exits
    1 even after a successful start).  Pin so a refactor doesn't
    silently drop the field and re-introduce the spurious failure."""
    for r in BUILTIN_RECIPES:
        if r.name == "molbuilder-MDtools":
            assert r.verify_ignore_exit_code is True, (
                "molbuilder-MDtools must have "
                "verify_ignore_exit_code=True (tleap exits 1 even "
                "when the env is healthy)"
            )
            return
    pytest.fail("molbuilder-MDtools recipe not found in registry")


def test_extra_steps_are_tuples_of_tuples():
    """Defensive shape check on extra_steps -- catches the easy
    mistake of writing `extra_steps=(("python","-m","x"),)` vs
    `extra_steps=("python","-m","x")`.  The latter would be parsed
    as four single-character commands."""
    for r in BUILTIN_RECIPES:
        for step in r.extra_steps:
            assert isinstance(step, tuple), (
                f"recipe `{r.name}`: extra_steps entry `{step!r}` "
                f"is not a tuple"
            )
            assert all(isinstance(arg, str) for arg in step), (
                f"recipe `{r.name}`: extra_steps entry has non-str "
                f"argv element: {step!r}"
            )
