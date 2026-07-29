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
        "docs/ops/installation.md (host + 4 backends)."
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


def test_cuda_wheel_tag_derived_from_cuda_version():
    """The PySCF recipe's pip_packages MUST carry cupy + gpu4pyscf
    wheels whose ``cuda<N>x`` suffix matches the project's resolved
    ``_CUDA_VERSION``.  Pins the contract: if the auto-detect /
    env-var override / project-default ladder lands on ``12.*``,
    the pip extras MUST land on ``cuda12x`` -- a hardcoded ``cuda13x``
    here would silently install a wheel incompatible with the user's
    driver."""
    from molbuilder.envs.recipes import (
        _CUDA_VERSION, _CUDA_WHEEL_TAG, _cuda_wheel_tag,
    )
    # Helper extracts the major digit.
    assert _cuda_wheel_tag("13.*") == "cuda13x"
    assert _cuda_wheel_tag("12.4") == "cuda12x"
    assert _cuda_wheel_tag(">=14") == "cuda14x"
    assert _cuda_wheel_tag("") == "cuda13x"  # safe fallback
    # Wheel tag matches the resolved CUDA version.
    import re
    expected = f"cuda{re.search(r'(\d+)', _CUDA_VERSION).group(1)}x"
    assert _CUDA_WHEEL_TAG == expected

    # The PySCF recipe references the derived tag in BOTH the cupy
    # and gpu4pyscf pip entries -- this is the load-bearing piece
    # (the recipe is what ``molbuilder envs install`` runs).
    for r in BUILTIN_RECIPES:
        if r.name == "molbuilder-pySCF":
            assert f"cupy-{_CUDA_WHEEL_TAG}[ctk]" in r.pip_packages, (
                f"PySCF recipe missing cupy-{_CUDA_WHEEL_TAG}[ctk] "
                f"in pip_packages (got: {r.pip_packages!r})"
            )
            assert f"gpu4pyscf-{_CUDA_WHEEL_TAG}" in r.pip_packages, (
                f"PySCF recipe missing gpu4pyscf-{_CUDA_WHEEL_TAG} "
                f"in pip_packages (got: {r.pip_packages!r})"
            )
            return
    pytest.fail("molbuilder-pySCF recipe not found in registry")


def test_resolve_cuda_version_env_override_beats_probe(monkeypatch):
    """Precedence: explicit MOLBUILDER_CUDA_VERSION must win over the
    host-driver probe.  Use case: cross-targeting or downgrading the
    pin on a host whose driver is newer than the project supports."""
    from molbuilder.envs.recipes import _resolve_cuda_version
    monkeypatch.setenv("MOLBUILDER_CUDA_VERSION", "11.8")
    assert _resolve_cuda_version() == "11.8"


def test_resolve_cuda_version_falls_back_to_default_without_driver(monkeypatch):
    """No env var + no nvidia-smi on PATH -> project default ``13.*``."""
    from molbuilder.envs import recipes
    monkeypatch.delenv("MOLBUILDER_CUDA_VERSION", raising=False)
    monkeypatch.setattr(recipes, "_detect_host_cuda_major", lambda: None)
    assert recipes._resolve_cuda_version() == "13.*"


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
