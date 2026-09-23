"""How a remedy is SPELLED -- one home, below every surface that prints one.

A detected problem carries its exact fix command, and that makes the spelling a
shared fact rather than a display detail: `doctor`'s `next:` lines, `install`'s
hard stop, a recipe's own build-time warning and `bootstrap`'s closing advice
must all name the same thing, or a reader ends up touring the docs to find out
which one is real.

It lives here, on floor 1 with no dependencies, because a home up in the CLI
cannot be reached from below.  `recipes.py` is recipe DATA (A7: nothing depends
upwards -- it may not import the CLI), so a spelling the CLI owns reaches a
recipe's own build-time warning only as a hand-copied shell string, and a copy
drifts: a remedy naming a recipe `recipe_by_name` does not accept is itself a
usage error, printed by the very code that detected the problem.
"""
from __future__ import annotations

#: The launcher form, and why it is the launcher rather than ``molbuilder envs``:
#: ``bash scripts/install-env.sh <verb> ...`` works from a bare shell -- it finds
#: the manager, ensures the host env and dispatches -- which is exactly the
#: situation a person with a broken env is in.  A second spelling of the same
#: fix -- ``molbuilder envs install`` -- is how a reader ends up reading
#: documentation instead of running the command.
LAUNCHER = "bash scripts/install-env.sh"


def fix_cmd(action: str, *args: str) -> str:
    """The ONE spelling of an env command: the launcher, the verb, then the
    recipe and flags.  The recipe name is one of ``*args`` and not a parameter
    of its own: required, the verbs that take no recipe (``bootstrap``,
    ``doctor``) cannot use this function and get spelled by hand instead."""
    return " ".join([LAUNCHER, action, *args])


def stdin_can_answer() -> bool:
    """Is somebody there to answer a prompt?

    A guarded ``isatty``: a detached or closed stdin raises rather than
    returning False.  Here, at the bottom of the CLI surface, because the
    top-level CLI and the envs verbs both ask it.  *"This package may not
    depend upwards"* is a true statement that argues for two copies; the
    answer to it is one copy placed low enough for both to reach.
    """
    import sys
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):        # detached or closed
        return False
