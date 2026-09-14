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


def fix_cmd(action: str, *args: str) -> str:
    """The ONE spelling of an env command: the launcher, the verb, then the
    recipe and flags.  ``recipe_name`` was a required second parameter until
    2026-09-13, so the verbs that take none (``bootstrap``, ``doctor``) were
    spelled by hand at three sites (K-D10)."""
    return " ".join([LAUNCHER, action, *args])


def stdin_can_answer() -> bool:
    """Is somebody there to answer a prompt?

    A guarded ``isatty``: a detached or closed stdin raises rather than
    returning False.  Here, at the bottom of the CLI surface, because the
    top-level CLI and the envs verbs both ask it -- it was written twice
    (`cli._stdin_is_a_terminal`, `envs._cli._stdin_can_answer`) "because
    this package may not depend upwards", when the answer was to put the
    one copy where both can reach it (K-D10).
    """
    import sys
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):        # detached or closed
        return False
