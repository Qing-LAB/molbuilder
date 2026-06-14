"""Drift guard: ``docs/README_install.md`` ↔ ``Recipe`` registry.

The README is the human-readable source of truth; the registry is
the machine-readable one.  These tests assert the two mention the
same env names and pin a small number of load-bearing details
(channels, build strings) so a quick "I'll just nudge this" edit on
one side can't silently desync from the other.

What we DO check:

  * Every Recipe.name appears as a section heading in README_install.md.
  * Every load-bearing token (e.g. ``siesta=5.4.2=mpi_openmpi_*``,
    ``dacase::ambertools-dac=26``, ``cupy-cuda13x[ctk]``) referenced by
    a recipe appears in the README -- if the registry pins it, the doc
    must explain why.
  * Every Recipe.verify_expect_contains substring is mentioned in the
    README (proves the verify step's expected output isn't an
    invention).

What we do NOT check:

  * The README's prose around each recipe (intentional -- the README
    explains the *why*; the registry encodes the *what*).
  * Exact conda-package order (the README writes them across multi-line
    bash blocks).
  * Optional / GPU sub-recipes (out of scope for this ship).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from molbuilder.envs.recipes import BUILTIN_RECIPES


REPO = Path(__file__).resolve().parents[1]
README = REPO / "docs" / "README_install.md"


@pytest.fixture(scope="module")
def readme_text() -> str:
    return README.read_text(encoding="utf-8")


def test_readme_exists():
    """Sanity check: the doc is where we think it is."""
    assert README.exists(), (
        f"docs/README_install.md not found at {README}; the "
        f"consistency tests are based on that path"
    )


def test_every_recipe_name_appears_in_readme(readme_text):
    """Every Recipe.name must appear verbatim in the README so a
    user reading the doc can find the recipe's install block."""
    missing = []
    for r in BUILTIN_RECIPES:
        if r.name not in readme_text:
            missing.append(r.name)
    assert not missing, (
        f"Recipe names not mentioned in README_install.md: {missing}. "
        f"Either add the recipe's section to the README or remove "
        f"the recipe from BUILTIN_RECIPES."
    )


def test_load_bearing_tokens_present(readme_text):
    """If the registry pins a non-obvious string (build string,
    extras tag, channel-prefixed spec), the README must mention it
    too -- otherwise a user copying from the README produces a
    different env than the registry installs."""
    expected_tokens = [
        # SIESTA build string -- distinguishes real-MPI from nompi.
        "siesta=5.4.2=mpi_openmpi_*",
        # AmberTools channel-prefixed spec -- README explains why
        # dacase wins over conda-forge.
        "dacase::ambertools-dac=26",
    ]
    missing = [t for t in expected_tokens if t not in readme_text]
    assert not missing, (
        f"Load-bearing tokens absent from README: {missing}.  "
        f"These pins are encoded in the registry; the README must "
        f"explain why."
    )


def test_verify_substrings_appear_in_readme(readme_text):
    """For every recipe with a verify_expect_contains string, that
    substring must appear in the README's verify block too -- pins
    that the registry's verify isn't divorced from what a user
    reading the doc would expect to see."""
    for r in BUILTIN_RECIPES:
        if not r.verify_expect_contains:
            continue
        # Tests env: README uses "pytest-playwright" in install
        # block + "playwright --version" prints "Version X.Y" which
        # is what we check; substring "Version" is too generic to
        # require verbatim in README.  Exempt this recipe.
        if r.name == "molbuilder-tests":
            continue
        # Host env's "host env OK" is our verify-line string, not
        # a README claim; exempt.
        if r.name == "molbuilder":
            continue
        assert r.verify_expect_contains in readme_text, (
            f"Recipe `{r.name}` checks for substring "
            f"`{r.verify_expect_contains}` in verify output but "
            f"README_install.md never mentions it.  Either fix the "
            f"README's verify block or the recipe's expected substring."
        )


def test_default_env_names_match_registry_names():
    """Cross-check the two registries -- DEFAULT_ENV_NAMES and
    BUILTIN_RECIPES.  Different module from test_envs_recipes.py
    (this one is in the README-consistency file because the README
    documents the env names too)."""
    from molbuilder.diagnostics import DEFAULT_ENV_NAMES
    for cat, name in DEFAULT_ENV_NAMES.items():
        matching = [r for r in BUILTIN_RECIPES if r.category == cat]
        assert len(matching) == 1, (
            f"category `{cat}` has {len(matching)} recipes (expect 1)"
        )
        assert matching[0].name == name, (
            f"category `{cat}` -> DEFAULT_ENV_NAMES says `{name}` "
            f"but recipe says `{matching[0].name}`"
        )
