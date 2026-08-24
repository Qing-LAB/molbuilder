"""Where molbuilder keeps its own per-user files — stated once.

``$XDG_CONFIG_HOME/molbuilder``, else ``~/.config/molbuilder``.  Three
modules computed that same two-line rule independently:

* ``runtime_config._per_user_fallback_path`` -> ``molbuilder.json``
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

__all__ = ["config_dir"]

#: The directory name under the XDG config root.  One string, because it is
#: the half of the path that is not the XDG convention.
DIRNAME = "molbuilder"


def config_dir() -> Path:
    """``$XDG_CONFIG_HOME/molbuilder``, else ``~/.config/molbuilder``.

    Not created, and not required to exist -- every caller either writes it
    on demand or treats an absent file as *unset*.  Read at CALL time rather
    than captured at import, so a test (or an operator) that moves
    ``XDG_CONFIG_HOME`` moves every one of the callers above together.
    """
    xdg = os.environ.get("XDG_CONFIG_HOME")
    return (Path(xdg) if xdg else Path.home() / ".config") / DIRNAME
