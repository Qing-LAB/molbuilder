"""Subprocess dispatch (``molbuilder.envs``): which env a tool runs in.

Two functions, both stateless except for reading the diagnostics snapshot:

  * ``run_in_env`` -- the command line, and the error when no manager exists
  * ``run_tool``   -- the dispatch POLICY: the routed env wins over host PATH,
    host PATH is the fallback, an explicit env overrides both, and each failure
    says which candidates were tried

**Nothing here is monkeypatched.**  The manager and the host tool are real
scripts in a temp directory that write down what they were asked to do, and the
inputs are the things a machine actually varies: ``PATH``, and the snapshot
(bound through `set_capabilities`, which is the public door).

That is a change of method, made 2026-09-13.  These tests used to patch
``molbuilder.envs.subprocess`` and ``molbuilder.envs.shutil`` -- and the only
reason those names existed at module scope was so that tests could patch them:
two imports in production code, carrying a comment saying they were
"re-exported for test monkeypatching".  Production shaped by its tests is
backwards, and a stub that intercepts one named function stops applying the
moment the code calls a different one, silently.  The imports are gone with
this rewrite.
"""

from __future__ import annotations

import os
import stat

import pytest

from molbuilder import envs
from molbuilder.diagnostics import Capabilities, set_capabilities


def _script(path, log):
    """A real executable that appends its own name + argv to ``log``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n"
                    f'echo "{path.name} $@" >> {log}\n'
                    'echo "ran"\n'
                    "exit 0\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _asked(log):
    return log.read_text().splitlines() if log.exists() else []


@pytest.fixture
def machine(tmp_path, monkeypatch):
    """A machine whose manager and host tools are recording scripts.

    ``PATH`` is emptied to just this box's fake ``bin``: what is on PATH is an
    INPUT to the policy under test, so the test sets it rather than patching
    the function that reads it.
    """
    log = tmp_path / "asked.log"
    mgr = _script(tmp_path / "conda", log)
    hostbin = tmp_path / "hostbin"
    hostbin.mkdir()
    monkeypatch.setenv("PATH", str(hostbin))

    class _Machine:
        manager = str(mgr)
        path_dir = hostbin
        asked = staticmethod(lambda: _asked(log))

        @staticmethod
        def on_host_path(tool):
            return _script(hostbin / tool, log)

        @staticmethod
        def envs_are(*names, manager=str(mgr)):
            set_capabilities(Capabilities(
                runtime_config={}, conda_binary=manager,
                conda_envs={n: f"/prefix/{n}" for n in names}))

    return _Machine


# --------------------------------------------------------------------- #
#  run_in_env                                                           #
# --------------------------------------------------------------------- #

def test_run_in_env_enters_the_named_env(machine):
    machine.envs_are("my-env")

    done = envs.run_in_env("my-env", ["echo", "hi"],
                           capture_output=True, text=True)

    assert machine.asked() == [
        "conda run -n my-env --no-capture-output echo hi"]
    assert done.returncode == 0
    # `--no-capture-output` is what lets the CALLER's capture_output work: the
    # manager streams the inner output through instead of eating it.
    assert "ran" in done.stdout


def test_run_in_env_refuses_when_no_manager_was_found(machine):
    machine.envs_are("my-env", manager=None)

    with pytest.raises(RuntimeError, match="conda CLI not found"):
        envs.run_in_env("my-env", ["echo"])


# --------------------------------------------------------------------- #
#  run_tool -- the policy                                               #
# --------------------------------------------------------------------- #

def test_the_routed_env_wins_even_when_the_tool_is_on_host_path(machine):
    """Load-bearing: this is what stops a stray system AmberTools silently
    shadowing the curated `molbuilder-MDtools`."""
    machine.envs_are("molbuilder-MDtools")
    machine.on_host_path("tleap")          # it IS on PATH as well

    envs.run_tool("tleap", ["-f", "build.in"])

    assert machine.asked() == [
        "conda run -n molbuilder-MDtools --no-capture-output "
        "tleap -f build.in"]


def test_host_path_is_the_fallback_when_the_routed_env_is_absent(machine):
    machine.envs_are()                     # no envs at all
    machine.on_host_path("tleap")

    envs.run_tool("tleap", ["-f", "build.in"])

    # Straight to the tool: no manager in the record at all.
    assert machine.asked() == ["tleap -f build.in"]


def test_an_explicit_env_overrides_the_routing(machine):
    machine.envs_are("custom-env")

    envs.run_tool("python", ["script.py"], env="custom-env")

    assert machine.asked() == [
        "conda run -n custom-env --no-capture-output python script.py"]


class TestWhatItSaysWhenItCannotRun:
    """Each refusal names what was tried, because the caller cannot see it."""

    def test_a_routed_tool_with_no_env_and_no_path(self, machine):
        machine.envs_are()
        with pytest.raises(FileNotFoundError, match="routed env"):
            envs.run_tool("tleap", ["-f", "build.in"])
        assert machine.asked() == [], "it dispatched something anyway"

    def test_an_unrouted_tool_that_is_not_on_path(self, machine):
        machine.envs_are()
        with pytest.raises(FileNotFoundError, match="no conda env routing"):
            envs.run_tool("definitely-not-a-real-tool", ["--flag"])

    def test_an_explicit_env_that_does_not_exist(self, machine):
        machine.envs_are("something-else")
        with pytest.raises(FileNotFoundError, match="does not exist"):
            envs.run_tool("tleap", ["-f", "x"], env="nonexistent-env")
        assert machine.asked() == []
