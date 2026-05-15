"""Subprocess dispatch into named conda envs.

Pure dispatch layer on top of :mod:`molbuilder.diagnostics`.  Two
functions, both stateless except for reading the diagnostics singleton:

  * :func:`run_in_env` -- explicit ``conda run -n <env>`` subprocess wrapper.
  * :func:`run_tool`   -- "routed env wins over host PATH" dispatch.

Dispatch policy in :func:`run_tool`: when a tool has a routing entry
(:data:`~molbuilder.diagnostics.TOOL_TO_CATEGORY`) AND the routed env
exists, dispatch into that env even if the tool is also on host PATH.
This prevents a stray system install (e.g. system-wide AmberTools in
``/usr/local/bin``) from silently shadowing the curated env the user
prepared per ``docs/README_install.md``.  Falling back to host PATH
happens only when no routing applies, or the routed env doesn't exist.

Testing hook: tests inject synthetic Capabilities via
:func:`molbuilder.diagnostics.set_capabilities` before calling these
functions, rather than passing a ``caps`` argument.  Keeps production
call sites simple.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Sequence, Optional

from .diagnostics import get_capabilities


def run_in_env(env_name: str,
               argv: Sequence[str],
               **popen_kwargs) -> subprocess.CompletedProcess:
    """Run ``argv`` inside ``env_name`` via ``conda run``.

    The ``--no-capture-output`` flag is passed so the caller's
    ``capture_output=True`` on the outer ``subprocess.run`` works
    transparently: conda streams the inner process's stdout/stderr
    through, and the outer ``subprocess.run`` captures them into
    ``result.stdout`` / ``result.stderr``.

    Raises :class:`RuntimeError` if the conda CLI itself wasn't found
    by the diagnostic probe.
    """
    caps = get_capabilities()
    if caps.conda_binary is None:
        raise RuntimeError(
            f"conda CLI not found.  Cannot dispatch `{' '.join(argv)}` "
            f"into env `{env_name}` because there's no `conda` on host "
            f"PATH and $CONDA_EXE is unset.  Install conda (or activate "
            f"a parent env that has it) before invoking molbuilder."
        )
    full = [caps.conda_binary, "run", "-n", env_name,
            "--no-capture-output", *argv]
    return subprocess.run(full, **popen_kwargs)


def run_tool(tool: str,
             argv: Sequence[str],
             *,
             env: Optional[str] = None,
             **popen_kwargs) -> subprocess.CompletedProcess:
    """Dispatch ``tool argv...`` to the routed env (preferred) or host PATH.

    Resolution order:

      1. Explicit ``env=...`` -- dispatch into that env, or raise
         :class:`FileNotFoundError` if it doesn't exist on this machine.
      2. Routed env (via :data:`~molbuilder.diagnostics.TOOL_TO_CATEGORY`)
         exists -- dispatch into it.
      3. Tool is on host PATH -- run it directly.
      4. Otherwise raise :class:`FileNotFoundError` with a message
         naming every candidate that was tried.

    Keyword args (``capture_output``, ``text``, ``cwd``, ``timeout``,
    ``env``, ...) flow through to ``subprocess.run`` unchanged.
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
        return run_in_env(env, [tool, *argv], **popen_kwargs)

    # (2) routed env, if registered and available
    routed = caps.routed_env(tool)
    if routed is not None:
        return run_in_env(routed, [tool, *argv], **popen_kwargs)

    # (3) host PATH
    if shutil.which(tool):
        return subprocess.run([tool, *argv], **popen_kwargs)

    # (4) not reachable -- pick the most informative error
    routed_name = caps.env_for_tool(tool)
    if routed_name is not None:
        raise FileNotFoundError(
            f"`{tool}`: not reachable.  Its routed env `{routed_name}` "
            f"does not exist on this machine, and `{tool}` is not on "
            f"host PATH.  Either create the env (see "
            f"docs/README_install.md) or install `{tool}` on the host."
        )
    raise FileNotFoundError(
        f"`{tool}`: not on host PATH, and no conda env routing is "
        f"registered for it.  Install on the host env, or pass "
        f"env=<name> to run_tool() explicitly."
    )


__all__ = ["run_in_env", "run_tool"]
