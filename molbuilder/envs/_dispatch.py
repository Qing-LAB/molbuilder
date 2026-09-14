"""Subprocess dispatch into named conda envs.

Pure dispatch layer on top of :mod:`molbuilder.diagnostics`.  Two
functions, both stateless except for reading the diagnostics singleton:

  * :func:`run_in_env` -- run one argv inside a named env, through the
    one door (`builds.dispatch_into_env`).
  * :func:`run_tool`   -- "routed env wins over host PATH" dispatch.

Dispatch policy in :func:`run_tool`: when a tool has a routing entry
(:data:`~molbuilder.diagnostics.TOOL_TO_CATEGORY`) AND the routed env
exists, dispatch into that env even if the tool is also on host PATH.
This prevents a stray system install (e.g. system-wide AmberTools in
``/usr/local/bin``) from silently shadowing the curated env the user
prepared per ``docs/ops/installation.md``.  Falling back to host PATH
happens only when no routing applies, or the routed env doesn't exist.

Testing hook: tests inject synthetic Capabilities via
:func:`molbuilder.diagnostics.set_capabilities` before calling these
functions, rather than passing a ``caps`` argument.  Keeps production
call sites simple.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional, Sequence

from ..diagnostics import get_capabilities
from . import builds as _builds
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
    installer step and build phase takes.  Until 2026-09-13 this function
    kept a second copy of that: its own stub detection, its own wrapper,
    addressed by NAME, and reaching up into `install` for the prefix when
    the copy needed one (J1, Y5).

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
        raise RuntimeError(
            f"conda CLI not found.  Cannot dispatch `{' '.join(argv)}` "
            f"into env `{env_name}`: "
            f"{caps.conda_binary_source or 'no manager on PATH and no $MAMBA_EXE/$CONDA_EXE'}.  "
            f"Record this machine's manager once as `envs.manager` in "
            f"molbuilder.json (an absolute path), or install/activate "
            f"one before invoking molbuilder."
        )
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
    """Dispatch ``tool argv...`` to the routed env (preferred) or host PATH.

    Resolution order (`route`):

      1. Explicit ``env=...`` -- dispatch into that env, or raise
         :class:`FileNotFoundError` if it doesn't exist on this machine.
      2. Routed env (via :data:`~molbuilder.diagnostics.TOOL_TO_CATEGORY`)
         exists -- dispatch into it.
      3. Tool is on host PATH -- run it directly.
      4. Otherwise raise :class:`FileNotFoundError` with a message
         naming every candidate that was tried.

    Output is captured as text either way; ``cwd`` and ``timeout`` are the
    two things a caller has to say.  (It took ``**popen_kwargs`` until
    2026-09-13, when the env branch stopped being a `subprocess.run`.)
    """
    routed = route(tool, env=env)
    if routed is None:
        return subprocess.run([tool, *argv], capture_output=True, text=True,
                              cwd=cwd, timeout=timeout)
    return run_in_env(routed, [tool, *argv], cwd=cwd, timeout=timeout)


def route(tool: str, *, env: Optional[str] = None) -> Optional[str]:
    """WHICH ENV should run `tool` -- the decision, with nothing dispatched.

    Returns the env name, or ``None`` meaning *"run it from host PATH"*.
    Raises :class:`FileNotFoundError` when neither is possible, naming every
    candidate that was tried.

    Split out of :func:`run_tool` on 2026-09-13.  The policy -- routed env
    beats host PATH, an explicit env beats both -- is the thing worth being
    sure of, and while it was tangled with the dispatch the only way to ask
    was to run something: the tests built a fake manager, fake tools, a PATH
    and a log file to read back one decision.  Now they ask.
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

    # (2) routed env, if registered and available.  IT WINS OVER HOST PATH,
    # which is what stops a stray system AmberTools in /usr/local/bin from
    # silently shadowing the curated env the person prepared.
    routed = caps.routed_env(tool)
    if routed is not None:
        return routed

    # (3) host PATH
    if shutil.which(tool):
        return None

    # (4) not reachable -- pick the most informative error
    routed_name = caps.env_for_tool(tool)
    if routed_name is not None:
        raise FileNotFoundError(
            f"`{tool}`: not reachable.  Its routed env `{routed_name}` "
            f"does not exist on this machine, and `{tool}` is not on "
            f"host PATH.  Either create the env (see "
            f"docs/ops/installation.md) or install `{tool}` on the host."
        )
    raise FileNotFoundError(
        f"`{tool}`: not on host PATH, and no conda env routing is "
        f"registered for it.  Install on the host env, or pass "
        f"env=<name> to run_tool() explicitly."
    )


__all__ = ["route", "run_in_env", "run_tool"]
