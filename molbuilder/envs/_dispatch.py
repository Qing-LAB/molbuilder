"""Subprocess dispatch into named conda envs.

Pure dispatch layer on top of :mod:`molbuilder.diagnostics`.  Two
functions, both stateless except for reading the diagnostics singleton:

  * :func:`run_in_env` -- run one argv inside a named env, through the
    one door (`builds.dispatch_into_env`).
  * :func:`run_tool`   -- a tool run in its env, as that env's own program.

Dispatch policy in :func:`run_tool`: a tool runs in the env its routing
entry (:data:`~molbuilder.diagnostics.TOOL_TO_CATEGORY`) names, or the one
its caller names, by its path there -- never from the host's PATH
(`docs/ops/installation.md` M6): a stray system install (e.g. system-wide
AmberTools in ``/usr/local/bin``) makes different chemistry from the
curated env, and nothing about the run would say so.  No env holding it:
refused, naming what installs or repairs it.

Testing hook: tests inject synthetic Capabilities via
:func:`molbuilder.diagnostics.set_capabilities` before calling these
functions, rather than passing a ``caps`` argument.  Keeps production
call sites simple.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional, Sequence

from ..diagnostics import TOOL_TO_CATEGORY, get_capabilities
from . import builds as _builds
from . import hints as _hints
from .builds import conda_run_argv


def run_in_env(env_name: str,
               argv: Sequence[str],
               *,
               cwd: Optional[Path] = None,
               timeout: Optional[int] = None,
               ) -> subprocess.CompletedProcess:
    """Run ``argv`` inside ``env_name`` -- THROUGH THE ONE DOOR (M1).

    `builds.dispatch_into_env`: the manager's own ``run``, re-addressed at
    the env's directory (M2), with the once-per-process measurement of a
    broken ``run`` stub and the activation fallback -- the route every
    installer step and build phase takes.  A private second copy of that
    -- its own stub detection, its own wrapper, addressed by NAME -- is
    wrong twice over: it drifts from the door it duplicates, and to get a
    prefix it has to reach up into `install`, inverting the layering.

    What differs from a build phase is only the STYLE of the answer.  A
    tool call wants a `CompletedProcess` with captured output rather than
    a streamed transcript, so the door runs quietly and the transcript
    comes back as ``stdout`` -- one stream: through the door stdout and
    stderr are not separable, and ``stderr`` is always ``""``.
    ``returncode`` is ``None`` when the command never launched.  A
    ``timeout`` overrun raises `subprocess.TimeoutExpired`, as
    `subprocess.run` would.

    Raises :class:`RuntimeError` if no manager was found.
    """
    caps = get_capabilities()
    if caps.conda_binary is None:
        raise RuntimeError(_hints.no_manager(
            f"`{' '.join(argv)}` in env `{env_name}`",
            caps.conda_binary_source))
    full = conda_run_argv(caps.conda_binary, env_name, *argv)
    rc, out = _builds.dispatch_into_env(
        full, caps.env_prefix(env_name), cwd=cwd, sink=None, timeout=timeout)
    if (rc is None and timeout is not None
            and out.endswith(_builds.timeout_tail(timeout))):
        raise subprocess.TimeoutExpired(list(full), timeout, output=out)
    return subprocess.CompletedProcess(list(full), rc, stdout=out, stderr="")


def run_tool(tool: str,
             argv: Sequence[str],
             *,
             env: Optional[str] = None,
             cwd: Optional[Path] = None,
             timeout: Optional[int] = None,
             ) -> subprocess.CompletedProcess:
    """Run ``tool argv...`` in its env, as that env's own program, by its
    path (`installation.md` M6) -- never one the host's PATH holds.

    Which env (`route`): an explicit ``env=...``, else the one
    :data:`~molbuilder.diagnostics.TOOL_TO_CATEGORY` routes it to; neither
    on this machine raises :class:`FileNotFoundError`, naming what installs
    it.  An env without the tool in its ``bin`` raises too, naming `doctor`,
    which names the repair.

    Output is captured as text; ``cwd`` and ``timeout`` are the two things
    a caller has to say.  ``**popen_kwargs`` cannot be offered: the door is
    not a `subprocess.run`, so a Popen knob would be accepted and silently
    dropped.
    """
    env_name = route(tool, env=env)
    own = get_capabilities().program_of(env_name, tool)
    if own is None:
        raise FileNotFoundError(
            f"`{tool}` is not a program of env `{env_name}`, and no other "
            f"`{tool}` is tried -- the host's PATH included.  "
            f"`{_hints.fix_cmd('doctor')}` names the repair.")
    return run_in_env(env_name, [own, *argv], cwd=cwd, timeout=timeout)


def route(tool: str, *, env: Optional[str] = None) -> str:
    """WHICH ENV runs `tool` -- the decision, with nothing dispatched.

    Returns the env name: an explicit ``env``, else the routed one.  The
    host's PATH is never the answer (`installation.md` M6): a stray system
    AmberTools in ``/usr/local/bin`` makes different chemistry from the
    pinned one, and nothing about the run would say so.  Raises
    :class:`FileNotFoundError` when no env on this machine runs it, naming
    what installs the routed one.

    The policy is the thing worth being sure of, so it is separated from
    the dispatch and can be ASKED.
    """
    caps = get_capabilities()

    # (1) explicit env override
    if env is not None:
        if not caps.env_available(env):
            raise FileNotFoundError(
                f"conda env `{env}` does not exist on this machine "
                f"(asked for `{tool}`).  Available envs: "
                f"{sorted(caps.conda_envs)}."
            )
        return env

    # (2) the routed env, if registered and on this machine.
    routed = caps.routed_env(tool)
    if routed is not None:
        return routed

    # (3) no env runs it -- and the host's PATH is not one.
    routed_name = caps.env_for_tool(tool)
    if routed_name is not None:
        from .recipes import recipe_of_category
        recipe = recipe_of_category(TOOL_TO_CATEGORY[tool])
        raise FileNotFoundError(
            f"`{tool}` runs in its routed env `{routed_name}`, which is not "
            f"on this machine; no other `{tool}` is tried -- the host's "
            f"PATH included.  `{_hints.install_cmd(recipe.name)}` installs "
            f"it.")
    raise FileNotFoundError(
        f"`{tool}`: no conda env routing is registered for it, and "
        f"molbuilder runs nothing from the host's PATH.  Pass env=<name> "
        f"to run_tool() to name the env that holds it.")


__all__ = ["route", "run_in_env", "run_tool"]
