"""How a remedy is SPELLED -- one home, below every surface that prints one.

A detected problem carries its exact fix command (user, 2026-08-20), and that
makes the spelling a shared fact rather than a display detail: `doctor`'s
`next:` lines, `install`'s hard stop, a recipe's own build-time warning and
`bootstrap`'s closing advice must all name the same thing, or a reader ends up
touring the docs to find out which one is real.

It lives here, on floor 1 with no dependencies, because the surface that used to
own it could not be reached from below.  `recipes.py` is recipe DATA (A7: it may
not depend on the CLI), so the one warning that tells you your GPU env is
missing its toolchain shims hand-copied `_cli._fix_cmd`'s output into a shell
string -- and the copy had drifted to a recipe name `recipe_by_name` does not
accept, so that remedy was itself a usage error (measured 2026-09-12).
"""
from __future__ import annotations

#: The launcher form, and why it is the launcher rather than ``molbuilder envs``:
#: ``bash scripts/install-env.sh <verb> ...`` works from a bare shell -- it finds
#: the manager, ensures the host env and dispatches -- which is exactly the
#: situation a person with a broken env is in.  ``doctor`` alone used to say
#: ``molbuilder envs install``; two spellings of one fix is how a reader ends up
#: reading documentation instead of running the command.
LAUNCHER = "bash scripts/install-env.sh"


def fix_cmd(action: str, recipe_name: str, *flags: str) -> str:
    """The ONE spelling of an env fix command."""
    return " ".join([LAUNCHER, action, recipe_name, *flags])
