"""Subprocess dispatch (``molbuilder.envs``).

Two functions to cover, both stateless except for reading the
diagnostics singleton.  Each test binds a synthetic Capabilities
snapshot via ``set_capabilities``; the autouse fixture in
``tests/conftest.py`` resets it afterwards.

Coverage:
  * ``run_in_env`` argv construction + the no-conda error path
  * ``run_tool`` dispatch policy: env-first, host-PATH-fallback,
    explicit-env override, informative errors
"""

from __future__ import annotations

import subprocess
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from molbuilder import diagnostics, envs
from molbuilder.diagnostics import Capabilities, set_capabilities


def _bind_caps(*, conda_binary="/usr/bin/conda", conda_envs=()):
    """Bind a synthetic snapshot for the current test."""
    set_capabilities(Capabilities(
        runtime_config = {},
        conda_binary   = conda_binary,
        conda_envs     = frozenset(conda_envs),
    ))


# --------------------------------------------------------------------- #
#  run_in_env -- conda subprocess argv shape                            #
# --------------------------------------------------------------------- #


def test_run_in_env_constructs_conda_run_command(monkeypatch):
    _bind_caps(conda_binary="/usr/bin/conda")
    calls: List[Dict[str, Any]] = []
    def fake_run(argv, **kw):
        calls.append({"argv": list(argv), "kw": kw})
        return MagicMock(returncode=0)
    monkeypatch.setattr(envs.subprocess, "run", fake_run)

    envs.run_in_env("my-env", ["echo", "hi"],
                    capture_output=True, text=True)

    assert calls[-1]["argv"] == [
        "/usr/bin/conda", "run", "-n", "my-env",
        "--no-capture-output", "echo", "hi",
    ]
    assert calls[-1]["kw"]["capture_output"] is True
    assert calls[-1]["kw"]["text"] is True


def test_run_in_env_raises_when_no_conda_binary():
    _bind_caps(conda_binary=None)
    with pytest.raises(RuntimeError, match="conda CLI not found"):
        envs.run_in_env("my-env", ["echo"])


# --------------------------------------------------------------------- #
#  run_tool -- dispatch policy                                          #
# --------------------------------------------------------------------- #


def _record_subprocess(monkeypatch) -> List[List[str]]:
    """Replace subprocess.run with a recorder; return the recording list."""
    calls: List[List[str]] = []
    def fake_run(argv, **kw):
        calls.append(list(argv))
        return MagicMock(returncode=0)
    monkeypatch.setattr(envs.subprocess, "run", fake_run)
    return calls


def test_run_tool_dispatches_to_routed_env_when_available(monkeypatch):
    """Env-first: routed env wins over host PATH.  Load-bearing -- this
    is what prevents a stray system AmberTools from silently shadowing
    molbuilder-MDtools."""
    _bind_caps(conda_envs={"molbuilder-MDtools"})
    # Tleap IS on host PATH too -- env should still win.
    monkeypatch.setattr(envs.shutil, "which",
                         lambda t: "/usr/bin/tleap" if t == "tleap" else None)
    calls = _record_subprocess(monkeypatch)

    envs.run_tool("tleap", ["-f", "build.in"])

    assert calls[-1][:4] == ["/usr/bin/conda", "run", "-n", "molbuilder-MDtools"]
    assert "tleap" in calls[-1]
    assert "build.in" in calls[-1]


def test_run_tool_falls_back_to_host_path_when_env_missing(monkeypatch):
    """No routed env -> use host PATH directly (no conda hop)."""
    _bind_caps(conda_envs=set())
    monkeypatch.setattr(envs.shutil, "which",
                         lambda t: "/usr/bin/tleap" if t == "tleap" else None)
    calls = _record_subprocess(monkeypatch)

    envs.run_tool("tleap", ["-f", "build.in"])

    # Direct host-PATH invocation: no `conda run` prefix.
    assert calls[-1] == ["tleap", "-f", "build.in"]


def test_run_tool_routed_env_missing_and_no_path_raises(monkeypatch):
    _bind_caps(conda_envs=set())
    monkeypatch.setattr(envs.shutil, "which", lambda t: None)
    with pytest.raises(FileNotFoundError, match="routed env"):
        envs.run_tool("tleap", ["-f", "build.in"])


def test_run_tool_unknown_tool_with_no_path_raises(monkeypatch):
    _bind_caps()
    monkeypatch.setattr(envs.shutil, "which", lambda t: None)
    with pytest.raises(FileNotFoundError, match="no conda env routing"):
        envs.run_tool("definitely-not-a-real-tool", ["--flag"])


def test_run_tool_explicit_env_override(monkeypatch):
    """Caller can force a specific env even for an unrouted tool."""
    _bind_caps(conda_envs={"custom-env"})
    calls = _record_subprocess(monkeypatch)
    envs.run_tool("python", ["script.py"], env="custom-env")
    assert calls[-1][:4] == ["/usr/bin/conda", "run", "-n", "custom-env"]


def test_run_tool_explicit_env_missing_raises():
    """Explicit env that doesn't exist -> clean error."""
    _bind_caps(conda_envs={"something-else"})
    with pytest.raises(FileNotFoundError, match="does not exist"):
        envs.run_tool("tleap", ["-f", "x"], env="nonexistent-env")
