"""How a remedy is SPELLED -- one home, below every surface that prints one.

A detected problem carries its exact fix command, and that makes the spelling a
shared fact rather than a display detail: `doctor`'s `next:` lines, `install`'s
hard stop, a recipe's own build-time warning and `bootstrap`'s closing advice
must all name the same thing, or a reader ends up touring the docs to find out
which one is real.

It lives here, on floor 1, asking only the package root (`repo_root`, the
checkout the launcher is named in), because a home up in the CLI cannot be
reached from below.  `recipes.py` is recipe DATA (A7: nothing depends
upwards -- it may not import the CLI), so a spelling the CLI owns reaches a
recipe's own build-time warning only as a hand-copied shell string, and a copy
drifts: a remedy naming a recipe `recipe_by_name` does not accept is itself a
usage error, printed by the very code that detected the problem.
"""
from __future__ import annotations

import shlex
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator, Optional

from .. import repo_root

#: The launcher form, and why it is the launcher rather than ``molbuilder envs``:
#: ``bash <checkout>/scripts/install-env.sh <verb> ...`` works from a bare shell
#: -- it finds the manager, ensures the host env and dispatches -- which is
#: exactly the situation a person with a broken env is in.  A second spelling of
#: the same fix -- ``molbuilder envs install`` -- is how a reader ends up reading
#: documentation instead of running the command.  **By the checkout's full
#: path**, so a line typed in any folder runs, as molbuilder's own command does
#: (the user, 2026-11-22: "Launcher by full path"; `ops/installation.md` § 2).
def launcher_of(checkout) -> str:
    """The launcher of the checkout at ``checkout`` as typed in any folder:
    its install script by full path, quoted for a shell."""
    return "bash " + shlex.quote(str(Path(checkout) / "scripts"
                                     / "install-env.sh"))


LAUNCHER = launcher_of(repo_root())

#: THE LAUNCHER AS TYPED IN MOLBUILDER'S CHECKOUT, whichever machine it lies
#: on -- what a file prepared for another machine names, since nothing prep
#: writes names the machine it was prepared on
#: (`execution/preparing-for-another-machine.md` § 2; the user, 2026-11-22:
#: "Full path here, words there").
LAUNCHER_IN_CHECKOUT = "bash scripts/install-env.sh"

#: Whether an env command composed now is said as typed in molbuilder's
#: checkout (:func:`typed_in_checkout`).
_IN_CHECKOUT: "ContextVar[bool]" = ContextVar("env_commands_in_checkout",
                                              default=False)


@contextmanager
def typed_in_checkout(on: bool = True) -> Iterator[None]:
    """While ``on``, each env command composed is said as typed in
    molbuilder's checkout (:data:`LAUNCHER_IN_CHECKOUT`, :func:`where_typed`)
    rather than by this checkout's full path (:data:`LAUNCHER`): in what
    prep writes for another machine, and in a run script on every machine,
    which is one file whichever machine it was prepared for
    (`execution/architecture.md` § 9)."""
    token = _IN_CHECKOUT.set(on)
    try:
        yield
    finally:
        _IN_CHECKOUT.reset(token)


def where_typed() -> str:
    """Where a printed env command is typed, when it is not anywhere: in
    molbuilder's checkout (:func:`typed_in_checkout`)."""
    return " in molbuilder's checkout" if _IN_CHECKOUT.get() else ""


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
    recipe and flags -- by this checkout's full path, or as typed in
    molbuilder's checkout (:func:`typed_in_checkout`)."""
    launcher = LAUNCHER_IN_CHECKOUT if _IN_CHECKOUT.get() else LAUNCHER
    return " ".join([launcher, action, *args])


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
