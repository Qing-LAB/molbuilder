"""Pre-run diagnostic: build a snapshot of what's available on this machine.

The snapshot answers three questions:

  * Which conda envs exist (by name) on this machine?
  * What does the machine config say (TLS paths, env-name overrides)?
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
in ``docs/ops/installation.md`` and ``docs/execution/job-contracts.md``, so
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
import collections.abc as _abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

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
    # Holds JupyterLab and nothing else -- the KERNELS are the other envs
    # (`recipes.py`'s `_JUPYTER`).  No tool routes to it: nothing dispatches
    # a command into this env, the notebook SERVER is launched into it by
    # `molbuilder.jupyter`, so it is absent from TOOL_TO_CATEGORY below.
    "jupyter":    "molbuilder-jupyternb",
}

# Executable name -> category.  Drives ``env_for_tool``.  Tools not
# listed here are not routed: the caller falls through to host PATH.
# ``playwright`` is intentionally NOT routed: browser E2E runs under the
# host env (in-process Flask app), not a dedicated env.
TOOL_TO_CATEGORY: Mapping[str, str] = {
    "siesta":      "siesta",
    "tleap":       "mdtools",
    "parmchk2":    "mdtools",
    "antechamber": "mdtools",
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
        ``{name: prefix}`` for every conda env on this machine -- the one
        reading of the registry, taken once at startup.  Membership is the
        common question (``name in caps.conda_envs``), and the prefix is there
        so that `install._env_prefix` does not pay the 1.2 s registry read
        again per recipe.
    """

    runtime_config: Mapping[str, Any] = field(default_factory=dict)
    conda_binary:   Optional[str]     = None
    #: WHERE conda_binary came from -- "molbuilder.json envs.manager",
    #: "PATH (mamba)" etc., or an error sentence when the RECORDED
    #: manager is unusable (then conda_binary is None and no silent
    #: fallback happened: a recorded fact that is wrong is a defect to
    #: surface, not a hint to second-guess).
    conda_binary_source: Optional[str] = None
    conda_envs:     Mapping[str, str] = field(default_factory=dict)

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

    def env_prefix(self, env_name: str) -> Optional[str]:
        """Where that env is, from the startup reading -- or ``None``.

        ``None`` means *"this snapshot does not know"*, which is not the same as
        *"it does not exist"*: an env created after the snapshot was taken (an
        install, mid-process) is absent here and found by `install._env_prefix`'s
        fresh read.  `reset_capabilities()` is how a caller that just changed the
        machine says so.

        It also answers ``None`` for a snapshot whose ``conda_envs`` is a plain
        set of names.  `detect` always builds the mapping, but a test binding a
        synthetic snapshot usually cares only whether an env EXISTS, and a
        membership-only container can answer *whether* and never *where*.
        ``None`` is the honest answer there rather than an error, and the live
        resolution then runs -- which is what such a test is exercising anyway.
        """
        envs = self.conda_envs
        if not isinstance(envs, _abc.Mapping):
            return None
        value = envs.get(env_name)
        return str(value) if value else None

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


def _find_conda_binary() -> "tuple[Optional[str], Optional[str]]":
    """Locate a conda-compatible env-manager CLI -> (path, source).

    THE PROBE HALF of the one manager door.  The RECORDED half wins
    first: :func:`detect` consults ``envs.manager`` in
    ``molbuilder.json`` before calling this (ops/installation.md,
    "one manager, one door", 2026-08-21) -- and a recorded manager
    that is missing or not executable REFUSES (conda_binary None with
    the reason in ``conda_binary_source``); it never silently falls
    back to a probe, because a wrong recorded fact is a defect to
    surface.

    Probe order (each step checks ``PATH`` via ``shutil.which``):

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
    candidate names.

    IMPORTANT: ``conda run -n <env>`` and ``mamba run -n <env>`` are
    NOT interchangeable on mamba 1.x.  mamba 1.x's ``run`` generates
    a shell stub that uses ``exec --`` which bash rejects with
    ``exec: --: invalid option`` -- every pip / extra-step / verify
    / build step would fail.  That is handled where commands are
    dispatched, not here and not by avoiding ``run``:
    ``molbuilder.envs.builds.dispatch_into_env`` uses the manager's own
    ``run`` (which activates the env, ``activate.d`` hooks included --
    measured on conda 26.7.1), MEASURES that signature if it appears, and
    then switches this process to its ``activation_wrapper``.  So this
    probe's job stops at naming the manager: whether that manager's
    ``run`` works is a question the one door answers by trying it, rather
    than one this function has to predict from a version number
    (`installation.md` M1, M4).
    """
    for candidate in ("mamba", "micromamba", "conda"):
        path = shutil.which(candidate)
        if path:
            return path, f"PATH ({candidate})"
    # The env-var fallback must check the path is EXECUTABLE, the way
    # ``shutil.which`` does above and the way install-env.sh's probe
    # already did (``[[ -n "${v}" && -x "${v}" ]]``).  Returning an
    # unvalidated value let a stale ``MAMBA_EXE`` -- pointing at a
    # removed or renamed install -- beat a correct ``CONDA_EXE`` that
    # the shim had just exported, because MAMBA_EXE is consulted first.
    # The shell rejected that path and Python accepted it, which is the
    # two-probes-disagreeing failure this whole seam exists to end.
    for var in ("MAMBA_EXE", "CONDA_EXE"):
        val = os.environ.get(var)
        if val and os.path.isfile(val) and os.access(val, os.X_OK):
            return val, f"${var}"
    return None, None


def manager_info(conda: str) -> Dict[str, Any]:
    """``<mgr> info --json`` as a dict, or ``{}`` when the manager will not say.

    THE ONE READER of that document.  Three callers parsed it themselves
    until 2026-09-13 -- the prefix resolver (for ``envs`` and ``envs_dirs``),
    the state probe (``envs_dirs`` again) and `init-config` (``root_prefix``)
    -- each with its own timeout and its own list of failures to swallow
    (K-D6).  An empty dict is every failure: the callers ask ``.get`` and
    read "the manager does not know".
    """
    try:
        cp = subprocess.run([conda, "info", "--json"],
                            capture_output=True, text=True, timeout=30)
    except (subprocess.SubprocessError, OSError):
        return {}
    if cp.returncode != 0:
        return {}
    try:
        info = json.loads(cp.stdout)
    except ValueError:
        return {}
    if not isinstance(info, dict):
        return {}
    # libmamba's dialect (micromamba, mamba >= 2) spells the same facts with
    # spaces: "envs directories", "base environment".  Read both, answer in
    # conda's names, so a caller never has to know which manager answered
    # (review B-L5; the libmamba names are from its docs, not measured here).
    if "envs_dirs" not in info and "envs directories" in info:
        info["envs_dirs"] = info["envs directories"]
    if "root_prefix" not in info and "base environment" in info:
        info["root_prefix"] = info["base environment"]
    return info


def conda_env_prefixes(conda: str) -> Dict[str, str]:
    """``{name: prefix}`` for every env the manager's registry lists.

    THE ONE READER of the registry.  It answers with the PREFIX, not just the
    name, because the prefix is what a command is addressed by
    (`installation.md` M2) and because answering with names alone is what made
    three readers of one document necessary.

    **The manager names its own envs.**  ``env list --json`` carries
    ``envs_details`` -- ``{prefix: {"name": ..., "base": true/false, ...}}`` --
    so the name comes from the manager and the base installation is excluded by
    its own ``base`` flag.  That matters beyond tidiness: the base env's name is
    ``base``, and its prefix's basename (``miniconda3``, ``anaconda3``) is not a
    name at all.  Keying it by basename would invent one.

    **The previous rule was "keep only envs whose parent directory is called
    `envs`"**, which excluded the base installation (the point) and every env
    created with ``--prefix`` somewhere else (not the point).  Those are listed
    by the registry and perfectly usable -- since 2026-09-12 a step is addressed
    by prefix, not by ``-n`` -- yet `caps.env_available` said **no** about an env
    `probe_env_state` reported **PRESENT**, and `repair` then said *"env does not
    exist.  Install it first"* about a healthy env.

    **A manager that reports no ``envs_details`` gets every listed prefix keyed
    by basename, with nothing excluded.**  That is deliberate and it is the
    safer of two wrong answers: keeping the old ``envs``-parent filter there
    hides an env created with ``--prefix``, and an env the registry lists but
    this map does not report reads as FRESH to `probe_env_state` -- whereupon
    ``conda create -n <name>`` makes a SECOND env beside the real one.  Measured:
    that filter broke three tests of the state machine the moment the probe
    started asking this function.  An installation root listed under its
    directory's basename is, by contrast, a name nothing ever asks about.

    Asking `info --json` for ``root_prefix`` would settle the degenerate case
    outright and costs a second subprocess -- measured 1.8 s against this read's
    1.2 s, on the startup path of every command -- so it is not paid here.

    Returns ``{}`` on any failure (timeout, non-zero exit, malformed JSON):
    callers ask by membership, so "no envs" is a clean answer rather than an
    exception each of them has to handle.
    """
    try:
        cp = subprocess.run(
            [conda, "env", "list", "--json"],
            capture_output=True, text=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return {}
    if cp.returncode != 0:
        return {}
    try:
        data = json.loads(cp.stdout)
    except json.JSONDecodeError:
        return {}
    details = data.get("envs_details")
    if isinstance(details, dict) and details:
        out: Dict[str, str] = {}
        for prefix, info in details.items():
            if not isinstance(info, dict) or info.get("base"):
                continue
            # `name` when the manager has one; the basename otherwise, which is
            # the handle `envs.<category>` uses for an env it was pointed at by
            # prefix.
            name = info.get("name") or os.path.basename(str(prefix))
            if name:
                out[str(name)] = str(prefix)
        return out
    # No `envs_details`: this document does not say which prefix is an
    # installation root, so nothing is guessed at -- see the docstring for why
    # listing one is the lesser error.
    return {os.path.basename(str(p)): str(p)
            for p in data.get("envs", []) or [] if p}


def detect() -> Capabilities:
    """Run the diagnostic and return a fresh ``Capabilities`` snapshot.

    Idempotent and cheap to re-call.  Callers that want process-wide
    state use :func:`initialize` + :func:`get_capabilities` instead.
    """
    cfg = read_config()
    # THE RECORDED FACT FIRST (ops/installation.md "one manager, one
    # door"): a machine that states its manager is not re-probed, and
    # a stated manager that is unusable is a NAMED defect -- never a
    # silent fallback to whatever PATH holds today.
    from .runtime_config import get_env_manager
    recorded = get_env_manager(cfg)
    if recorded:
        if os.path.isfile(recorded) and os.access(recorded, os.X_OK):
            conda, source = recorded, "molbuilder.json envs.manager"
        else:
            conda, source = None, (
                f"molbuilder.json envs.manager = {recorded!r} is not an "
                f"executable file on this machine -- fix the recorded "
                f"path (or remove the key to fall back to the PATH "
                f"probe)")
    else:
        conda, source = _find_conda_binary()
    envs_map = conda_env_prefixes(conda) if conda else {}
    return Capabilities(
        runtime_config = cfg,
        conda_binary   = conda,
        conda_binary_source = source,
        conda_envs     = envs_map,
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


# --------------------------------------------------------------------- #
#  What travels WITH a machine's record                                  #
# --------------------------------------------------------------------- #


def local_facts(env: "Any") -> "Tuple[Any, Optional[str]]":
    """Attach this machine's three portable facts to a probed record.

    Returns ``(environment, note)`` -- the record with the facts on it, and a
    line to show the operator when there were none to attach.

    **HOW THIS MACHINE ENTERS ITS ENVIRONMENT TRAVELS WITH THE RECORD**
    (2026-08-24).  A wrapper is generated on one machine and executed on
    another; the record is what carries the target across, and activation is as
    much a fact about the target as its core count.  Probing Sol records
    ``module load mamba`` / ``source activate``; copying that record to the
    workstation is then SUFFICIENT to generate a wrapper that runs on Sol.
    Without it, ``prep --target sol`` had Sol's queues and the workstation's
    conda hook, and every job died sourcing a path that exists on neither the
    cluster nor anywhere else it was sent.

    **WHICH ENVIRONMENTS EXIST HERE** travels too -- the other half of the
    pair.  ``conda env list`` enumerates without entering, so this is free from
    whatever env the probe itself runs in.  **AND WHAT THEY WERE BUILT FOR**:
    an env name is not portable, so that list means nothing without the
    instruction set it was seen on (user, 2026-08-26: *"we should know our
    compiled/installed architecture"*).  ``platform.machine()`` -- the machine
    running this, which is the machine those envs live on.

    Here rather than in ``scheduler/record.py`` because it reads live config
    (`runtime_config`, a layer above the scheduler package) and enumerates
    envs, and this module is where "what is true of this machine" already
    lives.  (Until 2026-09-13 this sentence also cited a "stdlib-only,
    ships to the target" contract of `record.py` that did not hold -- K-Y3.)  It was inline in ``jobset probe`` until
    2026-09-08, when ``envs init-config`` became a second caller -- and a
    second copy is a copy that drifts (`configuration.md` line 42).
    """
    import dataclasses as _dc
    try:
        from .runtime_config import get_script_generation
        sg = get_script_generation(project_dir=None)
        sg_rec = {k: v for k, v in (("preamble", sg.get("preamble")),
                                    ("activation", sg.get("activation")))
                  if v}
    except Exception:      # pragma: no cover - a broken config is its own error
        sg_rec = {}
    try:
        envs_here = sorted(get_capabilities().conda_envs or ())
    except Exception:      # pragma: no cover - enumeration is best-effort
        envs_here = []
    env_arch = None
    if envs_here:
        import platform as _pl
        env_arch = _pl.machine() or None
    # The note is about ``script_generation`` and is gated on
    # ``script_generation`` ALONE.  It was composed inside an ``else`` that
    # also required the env list to be empty -- so on any machine with a conda
    # env (which is every machine that can run anything) the warning was
    # suppressed by a fact it has nothing to do with.  It was unreachable
    # twice over: ``notes_sg`` was then assigned and read by nothing.
    note = None if sg_rec else (
        "this machine states no script_generation, so the record carries "
        "none -- a bundle prepped ELSEWHERE for this machine will be refused "
        "until it does")
    if sg_rec or envs_here:
        return _dc.replace(env, script_generation=sg_rec or {},
                           conda_envs=envs_here, env_arch=env_arch), note
    return env, note


__all__ = [
    "DEFAULT_ENV_NAMES", "TOOL_TO_CATEGORY", "EXTENSION_TO_CATEGORY",
    "Capabilities",
    "detect", "initialize", "local_facts",
    "get_capabilities", "set_capabilities", "reset_capabilities",
]
