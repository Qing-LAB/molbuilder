"""What the env registry must hold (`ops/env-framework.md` § 3.2b): the
packages every env carries, a recipe for every category, and in the host env
what the server imports.

The recipes module is the single source of truth for env shape.  A recipe's
required fields are `Recipe.__post_init__`'s, which refuses at import.
"""
from __future__ import annotations

import pytest

from molbuilder.diagnostics import DEFAULT_ENV_NAMES
from molbuilder.envs.recipes import (
    _PYTHON_SPEC,
    builtin_recipes,
    recipe_by_name,
)


#: Packages EVERY recipe carries, and the rule each one answers to.  Both are
#: about what is available at RUN time on a machine this process cannot probe,
#: which is why neither can be left to "whatever that env happened to need".
_UNIFORM_PACKAGES = (
    # `installation.md`, "Choosing the Python every env is built on": one
    # value, no exception.  Every python a generated job runs is the
    # activated env's own, `$CONDA_PREFIX/bin/python3` (`running-a-job.md`
    # § 2), so an env declaring none is a job that cannot start.
    (_PYTHON_SPEC, "the job's interpreter"),
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


# A routed recipe takes its name FROM the table (`name=DEFAULT_ENV_NAMES[...]`),
# so the names agree by construction and an unknown category is a `KeyError`
# at import.  The reverse direction below is not covered by that: a category
# added to `DEFAULT_ENV_NAMES` with no recipe is still writable.

def test_default_env_names_all_have_recipes():
    """Reverse direction: every category in ``DEFAULT_ENV_NAMES``
    must have a recipe.  Without one, ``envs install <category-env>``
    would have no plan.  THE SILENT FAILURE it also catches: a recipe copied
    with its category kept, which installs into the other's env, the second
    over the first, while the category it was copied from has no recipe."""
    covered = {r.category for r in builtin_recipes() if r.category}
    missing = set(DEFAULT_ENV_NAMES) - covered
    assert not missing, (
        f"DEFAULT_ENV_NAMES has categories with no Recipe: {missing}. "
        f"Add Recipe entries in molbuilder/envs/recipes.py."
    )


def test_host_env_includes_psutil():
    """``molbuilder serve`` imports `web/blueprints/system_load.py`, which
    imports psutil unconditionally, so without it the server dies before it
    binds a port (measured 2026-06-24, dd09d9824).  THE SILENT FAILURE: a
    working machine keeps the package, so every server test stays green, and
    the host recipe's `verify` does not import it -- only a new env finds out.
    """
    assert "psutil" in " ".join(recipe_by_name("molbuilder").conda_specs)
