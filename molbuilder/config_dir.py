"""Where molbuilder keeps its own per-user files — stated once.

``$XDG_CONFIG_HOME/molbuilder``, else ``~/.config/molbuilder``.  Three
modules computed that same two-line rule independently:

* ``runtime_config._machine_config_file`` -> ``molbuilder.json``
* ``scheduler/record.machine_scope_path`` -> ``environment.json`` and the
  ``environments/`` beside it
* ``auth_setup.default_secret_dir`` -> ``secret_key``

They agreed, and two of them said so in prose -- one docstring reads
*"Mirrors auth_setup.default_secret_dir's convention"*, the other
*"mirrored rather than imported"*.  **A comment is not a mechanism**, and
`configuration.md` M-4 already made this exact call one level down: it gave
``environment.json`` one home for its FILENAME, which "was a string literal
in three modules".  The directory that filename sits in never got the same
treatment.  This is it.

**L1: pure stdlib, no molbuilder deps -- any layer may use it.**  That line
is copied deliberately from ``persist.py``, which is the precedent: the same
shape (one rule, several callers, one of them ``scheduler/record.py``) and
the same resolution.  It is what lets ``record.py`` import this without
giving up the stdlib-only property it claims -- the property is *depends
only on stdlib*, not *imports nothing from molbuilder*, which is why
``record.py`` can already do ``from ..persist import write_json``.

**Why no ``paths.state`` setting to override it** (user decision,
2026-08-23).  ``XDG_CONFIG_HOME`` already moves this directory, and that is
the documented answer to the case that motivates moving it at all --
``auth_setup``'s own docstring: *"a user with ``$XDG_CONFIG_HOME=/scratch/
$USER`` keeps secrets off the NFS-mounted $HOME on HPC nodes."*  A config
key would be a second way to say one thing, and the ordering is *delete >
one home > parameter > abstraction*: one function is one home, a key is a
parameter.  It would also be circular for the first caller, which uses this
to FIND ``molbuilder.json``.  If a need ever appears to split them -- the
config in a repo, the state on scratch -- this function is where the
override hangs, and nothing here has to move first.
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = ["config_dir", "state_dir", "runtime_dir",
           "CONFIG_DIR_ENV", "DIRNAME"]

#: The directory name under the XDG config root.  One string, because it is
#: the half of the path that is not the XDG convention.
DIRNAME = "molbuilder"

#: Name this and it IS the root, exactly as given
#: (`plans/config-access-plan.md` § 3.1).
#:
#: Spelled like ``MOLBUILDER_DATA_DIR`` and ``MOLBUILDER_PROJECTS``, which are
#: already the convention for "the program's own <thing> directory".
CONFIG_DIR_ENV = "MOLBUILDER_CONFIG_DIR"


def config_dir() -> Path:
    """Where this installation's own configuration lives.

    ``$MOLBUILDER_CONFIG_DIR`` if set, else ``$XDG_CONFIG_HOME/molbuilder``,
    else ``~/.config/molbuilder``.

    Not created, and not required to exist -- every caller either writes it
    on demand or treats an absent file as *unset*.  Read at CALL time rather
    than captured at import, so a test (or an operator) that moves the root
    moves every one of the callers above together.

    **The override is used EXACTLY AS GIVEN** -- no ``molbuilder`` component is
    appended.  ``XDG_CONFIG_HOME`` names a root shared by every application, so
    ours must add its own name under it; ``MOLBUILDER_CONFIG_DIR`` names OUR
    directory, and appending to it would put the files somewhere the person did
    not ask for.  The two variables answer different questions and are treated
    differently on purpose.

    **It is an override, not a search step.**  Set it and that is the root,
    entire: nothing falls back past it, and a file in one of the other two
    places is not consulted.  A fallback here would recreate exactly the
    shadowing that `configuration.md` § 2.1a exists to warn about -- one
    setting, two files, one of them silently winning.
    """
    override = os.environ.get(CONFIG_DIR_ENV)
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CONFIG_HOME")
    return (Path(xdg) if xdg else Path.home() / ".config") / DIRNAME


def state_dir() -> Path:
    """Where operational state lives -- logs and reports.

    ``$XDG_STATE_HOME/molbuilder``, else ``~/.local/state/molbuilder``
    (`plans/config-access-plan.md` § 3.2).

    ``XDG_STATE_HOME`` entered the Base Directory spec in 0.8 for state that
    persists across restarts but is not portable or important enough for
    ``$XDG_DATA_HOME`` -- and the spec names LOGS first, which is what this
    holds.  ``~/.var/log`` and ``~/.local/log`` are not conventions:
    ``~/.var/app/`` is flatpak's, and the latter is not in the spec at all.

    **Separate from the config root on purpose.**  Configuration is edited and
    backed up; logs grow and are deleted.  A person may still put both in one
    place -- ``molbuilder.json``'s ``paths`` block names this directory, and
    :func:`molbuilder.runtime_config.logs_dir` is where that override is
    applied.  It cannot be applied HERE: this module is the bootstrap that
    finds ``molbuilder.json``, so it must answer before any config is read.
    """
    xdg = os.environ.get("XDG_STATE_HOME")
    return (Path(xdg) if xdg else Path.home() / ".local" / "state") / DIRNAME


def runtime_dir() -> Path:
    """Where pidfiles and sockets live.

    ``$XDG_RUNTIME_DIR/molbuilder`` when the variable is set, else
    ``state_dir()/run``.

    ``XDG_RUNTIME_DIR`` is the spec's directory for exactly this -- owner-only,
    and **cleared when the session ends**, which is right for a pidfile and
    wrong for anything meant to outlive a logout.  It is not always set (cron,
    a detached ssh, some containers), and the fallback is deliberately the
    STATE directory rather than a temp dir: a supervisor's pidfile that
    vanished under it would leave a running server nothing can find.
    """
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        return Path(xdg) / DIRNAME
    return state_dir() / "run"
