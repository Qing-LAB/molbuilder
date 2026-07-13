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


# --------------------------------------------------------------------- #
#  install-env.sh ↔ recipes.py host-env package list parity (2026-06-24)
#                                                                       #
#  The bootstrap path needs the host-env package list AVAILABLE FROM    #
#  BASH (cannot dispatch into the host env to read the Python recipe   #
#  before the host env exists).  install-env.sh therefore inlines      #
#  ``HOST_CONDA_PACKAGES`` + ``HOST_PIP_PACKAGES`` arrays.  This test  #
#  asserts those bash arrays match the Python source-of-truth recipe  #
#  at molbuilder/envs/recipes.py byte-for-byte so the two cannot       #
#  drift silently.                                                     #
# --------------------------------------------------------------------- #


def _parse_bash_array(text: str, name: str) -> list[str]:
    """Extract a bash array literal of the form ``NAME=( ... )``.

    Returns the tokens with version specifiers preserved (e.g.
    ``python=3.12``).  Comments + whitespace are stripped.
    """
    import re
    m = re.search(
        rf'^{re.escape(name)}=\(\s*(.*?)\s*\)\s*$',
        text, re.DOTALL | re.MULTILINE,
    )
    if m is None:
        raise AssertionError(
            f"could not find bash array {name}=(...) in install-env.sh; "
            f"the test depends on this declaration shape.  Verify the "
            f"script still defines the array literally as expected."
        )
    body = m.group(1)
    # Strip line comments (``# ...`` up to end-of-line); split on
    # whitespace; drop empties; strip a surrounding pair of single or
    # double quotes (the bash array MUST quote entries containing
    # shell metachars like ``>=`` to avoid them being parsed as
    # redirections -- but logically those quotes are not part of the
    # package spec).
    cleaned = re.sub(r"#[^\n]*", "", body)
    out = []
    for tok in cleaned.split():
        if not tok:
            continue
        if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in ('"', "'"):
            tok = tok[1:-1]
        out.append(tok)
    return out


def test_install_env_sh_host_conda_packages_match_recipe():
    """install-env.sh::HOST_CONDA_PACKAGES bash array must match
    Python ``BUILTIN_RECIPES.molbuilder.conda_packages`` exactly.

    The bash array is what the bootstrap uses to create the host env
    on a fresh machine.  A drift between bash and Python means a
    fresh-machine bootstrap creates a different package set from
    what every other entry point + the doctor expects.
    """
    from molbuilder.envs.recipes import recipe_by_name
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    sh_text = (repo_root / "scripts" / "install-env.sh").read_text()
    bash_pkgs = _parse_bash_array(sh_text, "HOST_CONDA_PACKAGES")

    host = recipe_by_name("molbuilder")
    py_pkgs = list(host.conda_packages)

    assert bash_pkgs == py_pkgs, (
        f"install-env.sh::HOST_CONDA_PACKAGES drifted from "
        f"recipes.py::_HOST.conda_packages.\n"
        f"  bash:   {bash_pkgs}\n"
        f"  python: {py_pkgs}\n"
        f"Update both lists in the same commit so a fresh-machine "
        f"bootstrap gets the same packages every other entry point "
        f"sees."
    )


def test_install_env_sh_host_pip_packages_match_recipe():
    """Same parity check for the pip-installable packages."""
    from molbuilder.envs.recipes import recipe_by_name
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    sh_text = (repo_root / "scripts" / "install-env.sh").read_text()
    bash_pkgs = _parse_bash_array(sh_text, "HOST_PIP_PACKAGES")

    host = recipe_by_name("molbuilder")
    py_pkgs = list(host.pip_packages or ())

    assert bash_pkgs == py_pkgs, (
        f"install-env.sh::HOST_PIP_PACKAGES drifted from "
        f"recipes.py::_HOST.pip_packages.\n"
        f"  bash:   {bash_pkgs}\n"
        f"  python: {py_pkgs}\n"
        f"Update both lists in the same commit."
    )
