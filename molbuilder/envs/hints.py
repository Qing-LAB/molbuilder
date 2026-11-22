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

from typing import Optional

#: The launcher form, and why it is the launcher rather than ``molbuilder envs``:
#: ``bash scripts/install-env.sh <verb> ...`` works from a bare shell -- it finds
#: the manager, ensures the host env and dispatches -- which is exactly the
#: situation a person with a broken env is in.  A second spelling of the same
#: fix -- ``molbuilder envs install`` -- is how a reader ends up reading
#: documentation instead of running the command.
LAUNCHER = "bash scripts/install-env.sh"


def no_manager(what: str, refusal: Optional[str] = None) -> str:
    """THE ONE SENTENCE for what found no package manager: ``what`` cannot
    run.  A person at a terminal loads the manager in their shell -- on Sol
    `module load mamba` -- and runs the command again; `envs.manager` is for
    what runs with no one there to load it (`ops/installation.md`, *The
    install script is not the CLI*; the user, 2026-11-19).  ``refusal`` is
    the resolver's, when the manager molbuilder.json records is what does
    not run (`diagnostics.detect`): a recorded manager wins, so loading
    another cannot help, and the record is what to fix."""
    if refusal:
        return (f"the package manager molbuilder.json records cannot run "
                f"({refusal}), so {what} cannot run.  Fix `envs.manager` "
                f"there, or remove it so the one loaded in this shell is "
                f"found.")
    return (f"no conda-compatible package manager was found, so {what} "
            "cannot run.  Load it in this shell -- on Sol, `module load "
            "mamba` -- and run this again; for what runs with no one there "
            "to load it, record it once as `envs.manager` in molbuilder.json "
            "(an absolute path).")


def fix_cmd(action: str, *args: str) -> str:
    """The ONE spelling of an env command: the launcher, the verb, then the
    recipe and flags."""
    return " ".join([LAUNCHER, action, *args])


def install_cmd(recipe: str) -> str:
    """The command that installs a backend recipe's env -- the remedy every
    surface names for one that is not there: `envs`' hints, `molbuilder
    jupyter` and the notebook page, and a test that skips for want of the
    env.  The host env's is :func:`install_again_cmd`'s."""
    return fix_cmd("install", recipe, "--yes")


def install_again_cmd(recipe) -> str:
    """The line that installs ``recipe`` again once its env is gone, or
    re-runs its post-install steps on a present one (doctor's ``then:``):
    the host env by `bootstrap`, which makes the molbuilder every other verb
    needs -- the install script runs its other verbs from that env, and
    refuses them while it is missing -- and writes its path file
    (`ops/deployment.md` § 0); any other by :func:`install_cmd`."""
    return (fix_cmd("bootstrap", "--yes") if recipe.category is None
            else install_cmd(recipe.name))


def rebuild_cmd(recipe: str) -> str:
    """The command that removes a recipe's env and installs it from
    scratch -- a backend's; the host env's is removed from outside it, then
    :func:`install_again_cmd` (`ops/installation.md` M5)."""
    return fix_cmd("install", recipe, "--clean", "--yes")


def opt_in_cmd(recipe: str) -> str:
    """The command that adds a recipe's opt-in records -- for the host env,
    the test tooling (`ops/env-framework.md` § 3.2a): what `doctor`,
    `repair` and a test that skips for want of it all name."""
    return fix_cmd("install", recipe, "--with-dev-tools", "--yes")


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
