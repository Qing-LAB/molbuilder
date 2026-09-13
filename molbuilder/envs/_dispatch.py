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
from typing import Sequence, Optional

from ..diagnostics import get_capabilities
# The run spelling, the signature of a broken manager `run`, the activation
# fallback and the once-per-process measurement all live with the install
# door (`builds`); this module shares them rather than keeping a second
# opinion.  What differs here is only the subprocess STYLE: a tool call wants
# a CompletedProcess with captured output, not a streamed transcript.
from . import builds as _builds
from .builds import conda_run_argv


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
            f"into env `{env_name}`: "
            f"{caps.conda_binary_source or 'no manager on PATH and no $MAMBA_EXE/$CONDA_EXE'}.  "
            f"Record this machine's manager once as `envs.manager` in "
            f"molbuilder.json (an absolute path), or install/activate "
            f"one before invoking molbuilder."
        )
    # ONE SPELLING of the run command line, shared with the install path.
    #
    # S17, closed 2026-09-12.  A manager whose `run` is broken -- mamba 1.x,
    # whose stub does ``exec -- "$@"`` whereupon bash rejects ``--`` -- used to
    # make this the one dispatch with no workaround: `run_tool("tleap", ...)`
    # died on a shell error about mamba's own file, naming nothing to do with
    # AmberTools.  The fix is the same RETRY the install door uses, and it keeps
    # this hot path (once per structure build) at exactly one subprocess on a
    # working manager: the prefix `activation_wrapper` needs is resolved only
    # after the stub has actually been seen, on the machines that have it.
    full = conda_run_argv(caps.conda_binary, env_name, *argv)
    if _builds.manager_run_unusable():
        wrapped = _wrap_for_broken_manager(full, env_name, caps.conda_binary)
        if wrapped is not None:
            return subprocess.run(list(wrapped), **popen_kwargs)
    done = subprocess.run(list(full), **popen_kwargs)
    if not _shows_broken_manager_run(done):
        return done
    _builds._MANAGER_RUN_UNUSABLE["seen"] = True
    wrapped = _wrap_for_broken_manager(full, env_name, caps.conda_binary)
    if wrapped is None:
        return done  # nothing better to offer; the original failure stands
    return subprocess.run(list(wrapped), **popen_kwargs)


def _shows_broken_manager_run(done: subprocess.CompletedProcess) -> bool:
    """Did this call fail with the broken-`run`-stub signature?

    Either stream, str or bytes, and ``None`` when the caller did not capture --
    in which case there is nothing to read and the answer is no.
    """
    if done.returncode == 0:
        return False
    sig = _builds.MANAGER_RUN_STUB_SIGNATURE
    for stream in (done.stdout, done.stderr):
        if isinstance(stream, bytes):
            if sig.encode() in stream:
                return True
        elif isinstance(stream, str) and sig in stream:
            return True
    return False


def _wrap_for_broken_manager(full, env_name: str, conda_binary: str):
    """The activation fallback for this argv, or ``None`` if the prefix is
    unknown -- in which case there is no wrapper to build and the caller keeps
    the manager's own failure."""
    from .install import _env_prefix  # imported here: `install` is above us
    prefix = _env_prefix(env_name, conda_binary)
    if prefix is None:
        return None
    return _builds.activation_wrapper(full, prefix)


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
    routed = route(tool, env=env)
    if routed is None:
        return subprocess.run([tool, *argv], **popen_kwargs)
    return run_in_env(routed, [tool, *argv], **popen_kwargs)


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
