"""What the bootstrap inlines in bash must equal the Python recipe.

**The subject is a duplication that cannot be removed.** `install-env.sh` needs
the host env's package list FROM BASH, before the host env exists -- so it
cannot dispatch into Python to read `recipes.py`, and it inlines
`HOST_CONDA_PACKAGES` / `HOST_PIP_PACKAGES` as bash arrays.  Two sources of
truth for one list, structurally, with no way to collapse them.  The two tests
here compare the arrays against the recipe exactly, and they are why this file
exists: drift there installs a different host env than the one molbuilder
believes it is running in.  `env-framework.md` § 8 names this check.

**Two weaker ones were retired 2026-09-14** (review D): every `Recipe.name`
appears as a heading in `docs/ops/installation.md`, and every
`verify_expect_contains` substring is mentioned there.  Both grepped a document
for a string.  A mention is not a result -- `doctor` running the verify is what
proves the substring, and a recipe a reader cannot find is a documentation
question, not a test.  (A third check this docstring claimed was measured false
on 2026-09-10 and never existed: editing `siesta=5.4.2=mpi_openmpi_*` in
installation.md left every test green.)
"""
from __future__ import annotations

from pathlib import Path

import pytest



REPO = Path(__file__).resolve().parents[1]
# The migrated installation guide is the human-readable source of truth.
README = REPO / "docs" / "ops" / "installation.md"


@pytest.fixture(scope="module")
def readme_text() -> str:
    return README.read_text(encoding="utf-8")


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

    **``${VAR:-default}`` IS EXPANDED, against this process's environment.**
    The host array is the one place bash must name a package before any python
    exists to read a recipe, so when a spec becomes overridable both halves
    have to resolve it -- and they have to resolve it the SAME way, or this
    guard fails on every machine where the variable happens to be set.
    `recipes.py` reads the variable with the same default at import; expanding
    it here is what keeps the comparison meaningful in both states rather than
    only in the unset one.
    """
    import os
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
        # Only the ``${NAME:-default}`` form, and only that form: anything
        # richer in a package list is a reason to look, not to interpret.
        tok = re.sub(
            r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}",
            lambda m: os.environ.get(m.group(1)) or m.group(2),
            tok,
        )
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
    # The shell array lists conda SPECS; the registry lists records.
    py_pkgs = list(host.conda_specs)

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
    # The shell array lists install specs; the registry lists records.
    # Compare what each would hand pip.
    py_pkgs = list(host.pip_specs)

    assert bash_pkgs == py_pkgs, (
        f"install-env.sh::HOST_PIP_PACKAGES drifted from "
        f"recipes.py::_HOST.pip_packages.\n"
        f"  bash:   {bash_pkgs}\n"
        f"  python: {py_pkgs}\n"
        f"Update both lists in the same commit."
    )
