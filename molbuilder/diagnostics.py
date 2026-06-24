"""Pre-run diagnostic: build a snapshot of what's available on this machine.

The snapshot answers three questions:

  * Which conda envs exist (by name) on this machine?
  * What does ``./molbuilder.json`` say (TLS paths, env-name overrides)?
  * Where is the ``conda`` CLI?

It does NOT pre-probe host PATH for individual tools -- ``shutil.which``
is sub-millisecond and runs on demand in :meth:`Capabilities.tool_available`.
The snapshot is therefore small: three fields, three methods, plus the
singleton lifecycle.

Routing tables
--------------

The four-env model's routing data lives at module scope as three small
dicts, hand-written for clarity.  Renames go in ``molbuilder.json``;
new categories require a code change here AND a documentation change
in ``docs/README_install.md`` and ``docs/protocols/job-layout.md``, so
they're rare.

The split with :mod:`molbuilder.envs`: envs is *dispatch only*
(:func:`~molbuilder.envs.run_in_env`, :func:`~molbuilder.envs.run_tool`).
It imports from this module; never the reverse.

Capabilities is ``frozen=True``: attribute reassignment is rejected.
The ``runtime_config`` field is exposed as a ``dict`` and *treated as
read-only by convention* -- Python doesn't deep-freeze nested mappings
cheaply.  Don't mutate it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from .runtime_config import get_envs, read_config
from .projects import projects_root


# --------------------------------------------------------------------- #
#  Routing tables (compile-time constants)                              #
# --------------------------------------------------------------------- #
#
# Categories describe the four-env model.  Naming maps go in three
# small dicts side-by-side; this is more legible at this scale than a
# dataclass abstraction over four static rows.
#

# Category name -> default conda env name (overridable per-machine via
# the "envs" section of molbuilder.json).
DEFAULT_ENV_NAMES: Mapping[str, str] = {
    "siesta":     "molbuilder-siesta",
    "siesta-gpu": "molbuilder-siesta-gpu",
    "pyscf":      "molbuilder-pySCF",
    "mdtools":    "molbuilder-MDtools",
    "tests":      "molbuilder-tests",
}

# Executable name -> category.  Drives ``env_for_tool``.  Tools not
# listed here are not routed: the caller falls through to host PATH.
TOOL_TO_CATEGORY: Mapping[str, str] = {
    "siesta":      "siesta",
    "tleap":       "mdtools",
    "parmchk2":    "mdtools",
    "antechamber": "mdtools",
    "playwright":  "tests",
}

# Script file extension -> category.  Used by the run-wrapper emitter
# to pick which env a ``.fdf`` or ``.py`` script should execute in.
EXTENSION_TO_CATEGORY: Mapping[str, str] = {
    ".fdf": "siesta",
    ".py":  "pyscf",
}


# --------------------------------------------------------------------- #
#  Capabilities snapshot                                                #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class Capabilities:
    """What molbuilder knows about this machine after a startup probe.

    Attributes
    ----------
    runtime_config
        Parsed ``molbuilder.json``, or ``{}`` if the file is absent.
        Read-only by convention.
    conda_binary
        Absolute path to the ``conda`` CLI, or ``None`` if not reachable.
    conda_envs
        Frozen set of named conda env names that exist on this machine.
    """

    runtime_config: Mapping[str, Any] = field(default_factory=dict)
    conda_binary:   Optional[str]     = None
    conda_envs:     frozenset         = frozenset()

    # ----- env-name resolution (consults config + defaults) --------- #

    def env_for_category(self, category: str) -> Optional[str]:
        """Env name for ``category``, honouring ``molbuilder.json`` overrides.

        Lookup order: ``envs.<category>`` in the config → the
        default from :data:`DEFAULT_ENV_NAMES`.  Returns ``None`` for
        unknown categories.
        """
        overrides = get_envs(self.runtime_config)
        if category in overrides:
            return overrides[category]
        return DEFAULT_ENV_NAMES.get(category)

    def env_for_tool(self, tool: str) -> Optional[str]:
        """Env name a tool would route to, or ``None`` if unregistered.

        Note this does NOT check whether the env exists; use
        :meth:`routed_env` for the "exists too" version.
        """
        category = TOOL_TO_CATEGORY.get(tool)
        if category is None:
            return None
        return self.env_for_category(category)

    def env_available(self, env_name: str) -> bool:
        """Whether the named conda env exists on this machine."""
        return env_name in self.conda_envs

    def routed_env(self, tool: str) -> Optional[str]:
        """Env name the tool routes to **and** which exists, else ``None``.

        Use this when you want the "ready-to-dispatch" answer: a non-
        ``None`` return is an env name you can pass straight to
        ``conda run -n <name>``.
        """
        env = self.env_for_tool(tool)
        if env is not None and self.env_available(env):
            return env
        return None

    def tool_available(self, tool: str) -> bool:
        """Whether ``tool`` is reachable -- either via routed env or host PATH.

        Composes :meth:`routed_env` with ``shutil.which`` for the host-
        PATH check.  Backends use this in their ``is_available()``.
        """
        if self.routed_env(tool) is not None:
            return True
        return shutil.which(tool) is not None

    # ----- file-picker root (projects/ only) ----------------------- #

    def file_picker_roots(self) -> Tuple[Tuple[Path, str], ...]:
        """Resolved root the ``/api/files`` picker is allowed to browse.

        Returns a single-element tuple ``((projects_root(), "projects"),)``.
        The picker is **deliberately scoped to** ``projects/`` -- that
        is molbuilder's single source of truth for run-state on disk.
        Files outside ``projects/`` (laptop downloads, scratch dirs)
        must be moved/copied into ``projects/<project>/<topic>/<structure>/``
        first; molbuilder's generators + the future derive-job flow
        keep that hierarchy honest.

        The projects/ path is always returned even if it doesn't exist
        yet, so the UI can show a "no projects yet" empty state.
        Plural return shape (a tuple) is preserved so a future single-
        line change can reintroduce multi-root if it earns its
        complexity -- but the contract today is single-root.
        """
        try:
            resolved = projects_root().expanduser().resolve()
        except (OSError, RuntimeError):
            # Defensive: a broken cwd symlink or mount loop shouldn't
            # crash the picker -- fall back to an empty tuple so the
            # UI shows "no roots" and the user gets a clear error.
            return ()
        return ((resolved, "projects"),)


# --------------------------------------------------------------------- #
#  Probe                                                                #
# --------------------------------------------------------------------- #


def _find_conda_binary() -> Optional[str]:
    """Locate a conda-compatible env-manager CLI.

    Search order (each step probes ``PATH`` via ``shutil.which``):

      1. ``mamba``     -- conda-compatible CLI with a much faster
                          (libmamba/libsolv) solver.  Drop-in replacement
                          for every ``conda create / run / env list``
                          command molbuilder issues.  Preferred when
                          present because env creation is ~5-10x faster
                          on HPC clusters with slow filesystems / many
                          packages, and the lockstep with conda's
                          API means everything else (subprocess
                          dispatch via ``conda run -n <env>``) works
                          identically.
      2. ``micromamba`` -- statically-linked single-binary mamba.
                          Same CLI surface for the molbuilder use
                          case.  Often the only env manager available
                          on ASU / general HPC clusters where the user
                          doesn't have admin rights to install
                          Miniconda.
      3. ``conda``     -- the reference implementation; always works
                          if installed.  Slowest solver of the three
                          but the universal fallback.

    After PATH probes, fall back to environment variables:
      * ``$MAMBA_EXE``  (set by mamba's activation hook)
      * ``$CONDA_EXE``  (set by conda's activation hook)

    Returns the absolute path to the chosen binary, or ``None`` if
    no manager is reachable.  Callers that need to know WHICH manager
    was picked can compare ``os.path.basename(path)`` against the
    candidate names; for the dispatch layer ``conda run -n <env>``
    and ``mamba run -n <env>`` are identical so the choice is
    transparent.
    """
    for candidate in ("mamba", "micromamba", "conda"):
        path = shutil.which(candidate)
        if path:
            return path
    return os.environ.get("MAMBA_EXE") or os.environ.get("CONDA_EXE") or None


def _list_conda_envs(conda: str) -> frozenset:
    """Run ``conda env list --json``, return *named* env basenames.

    Conda reports envs by full path; we extract basenames for ``-n``
    addressability.  The base installation root (e.g.
    ``/home/u/miniconda3``) is filtered out -- its parent dir isn't
    ``envs/`` and it isn't addressable via ``conda run -n``.

    Returns an empty frozenset on any failure (timeout, non-zero exit,
    malformed JSON) -- callers query the result by membership, so
    "no envs" is a clean answer, not an exception to handle.
    """
    try:
        cp = subprocess.run(
            [conda, "env", "list", "--json"],
            capture_output=True, text=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return frozenset()
    if cp.returncode != 0:
        return frozenset()
    try:
        data = json.loads(cp.stdout)
    except json.JSONDecodeError:
        return frozenset()
    out = set()
    for p in data.get("envs", []):
        # Named envs live under <something>/envs/<name>.
        if os.path.basename(os.path.dirname(p)) == "envs":
            out.add(os.path.basename(p))
    return frozenset(out)


def detect() -> Capabilities:
    """Run the diagnostic and return a fresh ``Capabilities`` snapshot.

    Idempotent and cheap to re-call.  Callers that want process-wide
    state use :func:`initialize` + :func:`get_capabilities` instead.
    """
    cfg = read_config()
    conda = _find_conda_binary()
    envs_set = _list_conda_envs(conda) if conda else frozenset()
    return Capabilities(
        runtime_config = cfg,
        conda_binary   = conda,
        conda_envs     = envs_set,
    )


# --------------------------------------------------------------------- #
#  Process-wide singleton (lazy)                                        #
# --------------------------------------------------------------------- #


_snapshot: Optional[Capabilities] = None


def initialize() -> Capabilities:
    """Run :func:`detect` and bind the result as the process snapshot.

    Idempotent: re-calling overwrites the previous snapshot, which is
    occasionally useful (long-running web app picks up a freshly-created
    env without restart).  Returns the snapshot it bound.  Raises
    whatever :func:`detect` raises (e.g. ``RuntimeConfigError`` from a
    malformed ``molbuilder.json``); startup paths should catch and
    translate to their UI-appropriate error shape.
    """
    global _snapshot
    _snapshot = detect()
    return _snapshot


def get_capabilities() -> Capabilities:
    """Return the bound snapshot, auto-initialising on first call."""
    global _snapshot
    if _snapshot is None:
        _snapshot = detect()
    return _snapshot


def set_capabilities(caps: Capabilities) -> None:
    """Bind a specific snapshot (tests, dependency injection)."""
    global _snapshot
    _snapshot = caps


def reset_capabilities() -> None:
    """Drop the snapshot; next :func:`get_capabilities` re-detects."""
    global _snapshot
    _snapshot = None


__all__ = [
    "DEFAULT_ENV_NAMES", "TOOL_TO_CATEGORY", "EXTENSION_TO_CATEGORY",
    "Capabilities",
    "detect", "initialize",
    "get_capabilities", "set_capabilities", "reset_capabilities",
]
