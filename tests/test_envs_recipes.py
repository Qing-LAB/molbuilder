"""L1 tests for ``molbuilder.envs.recipes``.

# Retired 2026-09-10: `@dataclass(frozen=True)` is the enforcement, and
# CPython refuses a non-frozen subclass of a frozen one on its own.
# A test that mutates an instance to watch Python raise tests Python.

Pins the registry shape: every built-in recipe must have non-empty
required fields, every routed recipe must reference a valid category
from ``DEFAULT_ENV_NAMES``, and the name-keyed + category-keyed
lookups must be consistent with the registry, `builtin_recipes()`.

The recipes module is the single source of truth for env shape;
these tests catch a stray "oh I'll change `name=` to a different
string for a moment" type regression that would silently disconnect
the recipe from the diagnostics-side category mapping.
"""
from __future__ import annotations

import pytest

from molbuilder.diagnostics import DEFAULT_ENV_NAMES
from molbuilder.envs.recipes import (
    _PYTHON_SPEC,
    builtin_recipes,
    Recipe,
    recipe_by_name,
    recipe_for_category,
)


def test_registry_is_non_empty():
    assert len(builtin_recipes()) >= 5, (
        "Registry should hold at least the five envs documented in "
        "docs/ops/installation.md (host + 4 backends)."
    )


def test_every_recipe_has_required_fields():
    """Every recipe must have non-empty name, description, channels,
    and conda_packages.  Empty values would render the recipe
    unusable."""
    for r in builtin_recipes():
        assert r.name, f"empty name in recipe {r}"
        assert r.description, f"empty description in {r.name}"
        assert r.channels, f"no channels in {r.name}"
        assert r.conda_packages, f"no conda_packages in {r.name}"


#: Packages EVERY recipe carries, and the rule each one answers to.  Both are
#: about what is available at RUN time on a machine this process cannot probe,
#: which is why neither can be left to "whatever that env happened to need".
_UNIFORM_PACKAGES = (
    # `installation.md`, "Choosing the Python every env is built on": one
    # value, no exception.  A generated wrapper backgrounds `mb_monitor.pyz`
    # with whatever `command -v python3` finds AFTER the env is activated, so
    # an env declaring no python falls through its own empty `bin/` to the
    # COMPUTE NODE's interpreter -- a version nothing declares, probes at prep
    # time, or can promise is installed.  `molbuilder-siesta` was that env
    # until 2026-09-17; measured inside it, `python3` resolved to
    # `/usr/bin/python3`.
    (_PYTHON_SPEC, "the run monitor's interpreter"),
    # `checkpoint.py`'s `GitNotInstalledError` tells the user to activate a
    # molbuilder env because "every molbuilder env ships git as a
    # conda_packages entry" -- a promise the registry has to keep, and HPC
    # sites' system git versions are inconsistent enough that the env's is
    # the only one we control (`_HOST`).
    ("git", "the checkpoint subsystem"),
)


@pytest.mark.parametrize(
    "spec,why",
    _UNIFORM_PACKAGES,
    ids=[w for _, w in _UNIFORM_PACKAGES],
)
def test_every_recipe_declares_the_uniform_packages(spec, why):
    """The packages every env carries, carried by every env.

    ONE test for both, because it is one rule: a package here is declared
    uniformly so that nothing downstream has to remember which env is "the one
    with git" -- and so that a recipe added tomorrow cannot quietly omit it.
    Neither was enforced anywhere before 2026-09-17; `git` was declared in all
    six by hand and `python` in five of six, which is exactly the failure a
    by-hand rule produces.

    `python` is asserted through `_PYTHON_SPEC` rather than the literal
    `python=3.12`, because `MOLBUILDER_PYTHON` moves that value -- a recipe
    spelling it from anywhere else is the drift worth catching.

    The rule is stated, not just measured: `env-framework.md` § 3.2b.
    """
    missing = [r.name for r in builtin_recipes() if spec not in r.conda_specs]
    assert not missing, (
        f"{', '.join(missing)} do not declare {spec!r}, which every env "
        f"carries for {why}."
    )


def test_recipe_names_are_unique():
    names = [r.name for r in builtin_recipes()]
    assert len(names) == len(set(names)), (
        f"duplicate recipe names: {names}"
    )


# `test_routed_recipes_match_default_env_names` stood here.  A routed recipe
# takes its name FROM the table now (`name=DEFAULT_ENV_NAMES["pyscf"]`), so
# both halves it asserted are structural: the names agree by construction, and
# an unknown category is a `KeyError` at import -- demonstrated, not argued.
#
# The reverse direction below is NOT covered by that and stays: a category
# added to `DEFAULT_ENV_NAMES` with no recipe is still writable.

def test_default_env_names_all_have_recipes():
    """Reverse direction: every category in ``DEFAULT_ENV_NAMES``
    must have a recipe.  Without one, ``envs install <category-env>``
    would have no plan."""
    covered = {r.category for r in builtin_recipes() if r.category}
    missing = set(DEFAULT_ENV_NAMES) - covered
    assert not missing, (
        f"DEFAULT_ENV_NAMES has categories with no Recipe: {missing}. "
        f"Add Recipe entries in molbuilder/envs/recipes.py."
    )


def test_recipe_by_name_finds_every_recipe():
    for r in builtin_recipes():
        assert recipe_by_name(r.name) is r


def test_recipe_by_name_returns_none_for_unknown():
    assert recipe_by_name("nope-not-a-real-env") is None


def test_recipe_for_category_finds_every_routed_recipe():
    for r in builtin_recipes():
        if r.category is None:
            continue
        assert recipe_for_category(r.category) is r


def test_recipe_for_category_returns_none_for_unknown():
    assert recipe_for_category("not-a-category") is None


# The CUDA version the GPU recipes carry, at each of its three tiers, is
# `test_the_machine_is_probed_on_first_use.py`'s: read off what `envs install
# --dry-run` would run, the resolution included (plan W36 ⑥).  Three API-level
# tests of `_resolve_cuda_version` and the registry stood here until
# 2026-09-29; the command line shows every fact they asserted, and the probe's
# own tier, which none of them reached.


# `test_extra_steps_are_tuples_of_tuples` stood here.  Its docstring called
# itself a "defensive shape check" and named the mistake --
# `extra_steps=("python","-m","x")` instead of `(("python","-m","x"),)`,
# which the installer runs as one command per character.  The constructor
# refuses both that and a non-str argument now.
