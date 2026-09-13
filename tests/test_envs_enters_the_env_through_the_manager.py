"""How a step gets INTO an env: the manager's own `run`, not an imitation.

`docs/ops/installation.md` § "One door for RUNNING" (M1-M4) and
`docs/ops/env-framework.md` § 5.6.  The rule came from the user, 2026-09-12:
*"different machine could have different setups.  It might not be conda at all.
It could be mamba... use the detected activation method and the correct way to
execute the script rather than directly hacking to the directory."*

Until then `run_step` rewrote EVERY step into a hand-written bash wrapper that
re-implemented `conda activate` -- so no machine used its own manager's
activation, two copies of that wrapper had drifted four ways, and the one
dispatch that did NOT have the rewrite (`_dispatch.run_in_env`, the tool router)
simply failed on the machines the rewrite existed for.

This machine has only conda, so mamba 1.x -- the reason the wrapper exists -- is
reproduced here by a fake manager that emits its signature.
"""
from __future__ import annotations

import os
import stat

import pytest

from molbuilder.envs import builds as B


@pytest.fixture(autouse=True)
def _forget_the_measurement():
    """The broken-manager measurement is per PROCESS, so a test that provokes
    it would otherwise decide the next test's dispatch."""
    B.reset_manager_run_measurement()
    yield
    B.reset_manager_run_measurement()


def _fake_manager(tmp_path, name, body):
    p = tmp_path / name
    p.write_text("#!/bin/sh\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(p)


def _prefix(tmp_path):
    prefix = tmp_path / "envs" / "someenv"
    (prefix / "bin").mkdir(parents=True, exist_ok=True)
    return str(prefix)


# --------------------------------------------------------------------------- #
#  M1 -- the manager activates                                                #
# --------------------------------------------------------------------------- #

def test_a_working_manager_is_used_directly_and_no_shell_is_generated(
        tmp_path, monkeypatch):
    """The route: one launch, the manager's own `run`, no `bash -c` anywhere."""
    launched = []

    def fake_stream(argv, **kw):
        launched.append(tuple(argv))
        return (0, "done")

    monkeypatch.setattr(B, "run_streaming", fake_stream)
    argv = B.conda_run_argv("/opt/mgr/bin/micromamba", "someenv", "python", "-V")

    rc, out = B.dispatch_into_env(argv, _prefix(tmp_path))

    assert (rc, out) == (0, "done")
    assert len(launched) == 1, f"more than one attempt: {launched}"
    assert "bash" not in launched[0], (
        "a working manager's activation was replaced by a generated shell:\n"
        f"{launched[0]}")
    assert launched[0][:2] == ("/opt/mgr/bin/micromamba", "run")


def test_the_env_is_addressed_by_the_prefix_the_registry_gave_us(
        tmp_path, monkeypatch):
    """M2 + H13.  conda resolves `-n` against `envs_dirs` ONLY
    (`locate_prefix_by_name` raises `EnvironmentNameNotFound` otherwise), so an
    env created with `--prefix` on a scratch filesystem cannot be entered by
    name at all.  The planner has only a name -- `plan_install` runs nothing --
    so the door narrows the address to the prefix it was handed."""
    launched = []
    monkeypatch.setattr(B, "run_streaming",
                        lambda argv, **kw: (launched.append(tuple(argv)), (0, ""))[1])
    out_of_tree = tmp_path / "scratch" / "mb"
    (out_of_tree / "bin").mkdir(parents=True)

    B.dispatch_into_env(B.conda_run_argv("/m", "mb", "python"), str(out_of_tree))

    assert "-n" not in launched[0], launched[0]
    assert launched[0][2:4] == ("--prefix", str(out_of_tree)), launched[0]


def test_the_manager_s_own_command_is_not_wrapped(tmp_path, monkeypatch):
    """`conda create` / `conda install -n ...` ARE the manager's command: there
    is no env to enter yet and nothing to carry.  Wrapping `argv[5:]` of a
    create command would produce a plausible shell string that runs the wrong
    thing, so the wrapper refuses them outright."""
    launched = []
    monkeypatch.setattr(B, "run_streaming",
                        lambda argv, **kw: (launched.append(tuple(argv)), (0, ""))[1])
    create = ("/m", "create", "-n", "someenv", "-y", "python=3.12", "pip")

    B.dispatch_into_env(create, _prefix(tmp_path))

    assert launched[0] == create
    with pytest.raises(ValueError):
        B.activation_wrapper(create, _prefix(tmp_path))


# --------------------------------------------------------------------------- #
#  M4 -- a manager's bug is measured, then worked around                      #
# --------------------------------------------------------------------------- #

def test_a_mamba_1x_run_stub_is_measured_and_the_step_still_succeeds(
        tmp_path, capsys):
    """THE CLUSTER CASE, reproduced.  mamba 1.x's `run` writes a stub that does
    ``exec -- "$@"``; bash rejects `--` and the dispatch dies on line 5 of
    mamba's own file, BEFORE the inner command starts -- which is what makes
    retrying safe rather than a second half-install.

    No mamba on this machine (conda 26.7.1 only), so the manager is faked at
    exactly that failure.  The step must still run, through the activation
    fallback, and the real python must be what answers.
    """
    broken = _fake_manager(
        tmp_path, "mamba",
        'echo "/tmp/mamba-stub.sh: line 5: exec: --: invalid option" >&2\n'
        "exit 126\n")
    prefix = _prefix(tmp_path)
    argv = B.conda_run_argv(broken, "someenv",
                            "python", "-c", "print('the step ran')")

    rc, out = B.dispatch_into_env(argv, prefix, sink=None)

    assert rc == 0, f"the step did not survive a broken manager `run`:\n{out}"
    assert "the step ran" in out, out
    assert B.manager_run_unusable() is True, (
        "the failure was worked around but not remembered, so every later step "
        "pays the same wasted launch")


def test_once_measured_the_broken_manager_is_not_tried_again(
        tmp_path, monkeypatch):
    """One wasted launch per process, not per step."""
    attempts = []

    def fake_stream(argv, **kw):
        attempts.append(tuple(argv))
        if "bash" not in argv[0]:
            return (126, "sh: line 5: exec: --: invalid option")
        return (0, "ok")

    monkeypatch.setattr(B, "run_streaming", fake_stream)
    prefix = _prefix(tmp_path)
    argv = B.conda_run_argv("/m/mamba", "someenv", "python", "-V")

    B.dispatch_into_env(argv, prefix)
    first = len(attempts)
    B.dispatch_into_env(argv, prefix)

    assert first == 2, f"expected native-then-wrapper, got {attempts}"
    assert len(attempts) == 3, (
        f"the second step retried the manager that was already measured "
        f"broken: {attempts}")
    assert attempts[-1][0] == "bash"


def test_a_real_failure_is_not_mistaken_for_the_stub_bug(tmp_path, monkeypatch):
    """A step that genuinely fails must NOT be retried through the wrapper --
    that would run a failed pip install twice and call the manager broken."""
    attempts = []

    def fake_stream(argv, **kw):
        attempts.append(tuple(argv))
        return (1, "ERROR: Could not find a version that satisfies nosuchpkg")

    monkeypatch.setattr(B, "run_streaming", fake_stream)
    rc, _out = B.dispatch_into_env(
        B.conda_run_argv("/m", "someenv", "python", "-m", "pip", "install", "x"),
        _prefix(tmp_path))

    assert rc == 1
    assert len(attempts) == 1, f"a real failure was retried: {attempts}"
    assert B.manager_run_unusable() is False


# --------------------------------------------------------------------------- #
#  The environment a step runs in                                             #
# --------------------------------------------------------------------------- #

def test_every_step_gets_the_clean_slate_only_builds_used_to_get(
        tmp_path, monkeypatch):
    """H3's live consequence.  `run_step`'s `env` parameter existed and no
    caller passed it, while `builds.py` passed `build_subprocess_env()` at two
    sites -- so every pip step and every extra step ran with exactly the host
    `CPATH` / `CFLAGS` / `CUDA_HOME` / `OMPI_*` leakage that module exists to
    strip, two functions away from the stripper."""
    monkeypatch.setenv("CPATH", "/usr/include/evil")
    monkeypatch.setenv("CUDA_HOME", "/opt/cuda-system")
    monkeypatch.setenv("OMPI_MCA_btl", "self,tcp")
    monkeypatch.setenv("KEEP_ME", "yes")
    prefix = _prefix(tmp_path)

    env = B.env_for_step(prefix)

    assert "CPATH" not in env and "CUDA_HOME" not in env
    assert "OMPI_MCA_btl" not in env
    assert env["KEEP_ME"] == "yes"
    # Temp and cache inside the prefix, so removing the env really cleans up
    # and a small /tmp on a cluster cannot fail a wheel build.
    assert env["TMPDIR"] == f"{prefix}/var/tmp"
    assert env["PIP_CACHE_DIR"] == f"{prefix}/var/cache/pip"
    assert os.path.isdir(env["TMPDIR"])


def test_run_step_hands_that_environment_to_the_door(tmp_path, monkeypatch):
    """The dead parameter is gone and the door does it instead -- asserted
    through `run_step`, because "the function exists" was never the gap."""
    from molbuilder.envs import install as I

    seen = {}

    def fake_dispatch(argv, prefix, *, env=None, **kw):
        seen["env"] = env
        seen["prefix"] = prefix
        return (0, "Version 1.40")

    monkeypatch.setattr(I._builds, "dispatch_into_env", fake_dispatch)
    monkeypatch.setenv("CFLAGS", "-march=native-from-the-user")
    prefix = _prefix(tmp_path)
    step = I.InstallStep(
        label="pip", role=I.StepRole.PACKAGES,
        argv=B.conda_run_argv("/m", "someenv", "python", "-m", "pip",
                              "install", "pubchempy"))

    done = I.run_step(step, prefix=prefix)

    assert done.outcome is I.Outcome.OK
    assert "CFLAGS" not in seen["env"], (
        "a pip step still runs with the user's compiler flags visible")
    assert seen["env"]["TMPDIR"].startswith(prefix)


# --------------------------------------------------------------------------- #
#  S17 -- the tool router gets the same answer                                #
# --------------------------------------------------------------------------- #

def test_the_tool_router_survives_a_broken_manager_too(tmp_path, monkeypatch):
    """S17, closed.  `run_in_env` dispatched `<mgr> run` with no workaround,
    and its own comment recorded why: the workaround needed a prefix, and
    resolving one costs up to four manager subprocesses -- too much on a path
    walked once per structure build.  So on a mamba-1.x host `run_tool("tleap",
    ...)` died with a shell error about mamba's generated file, naming nothing
    to do with AmberTools.

    Measuring instead of predicting resolves that trade: the prefix is resolved
    only AFTER the stub has actually been seen, so a working manager still pays
    exactly one subprocess and the broken one pays the resolution once.
    """
    from molbuilder.diagnostics import Capabilities, set_capabilities
    from molbuilder.envs import _dispatch, install as I

    broken = _fake_manager(
        tmp_path, "mamba",
        'echo "/tmp/stub.sh: line 5: exec: --: invalid option" >&2\nexit 126\n')
    prefix = _prefix(tmp_path)
    set_capabilities(Capabilities(runtime_config={}, conda_binary=broken,
                                 conda_envs=frozenset({"molbuilder-MDtools"})))
    monkeypatch.setattr(I, "_env_prefix", lambda name, binary: prefix)

    done = _dispatch.run_in_env(
        "molbuilder-MDtools",
        ["python", "-c", "print('tleap would have run')"],
        capture_output=True, text=True, timeout=120)

    assert done.returncode == 0, (
        f"the tool call died on the manager's stub:\n{done.stdout}{done.stderr}")
    assert "tleap would have run" in done.stdout


def test_the_tool_router_does_not_retry_an_ordinary_tool_failure(
        tmp_path, monkeypatch):
    """A tool that exits non-zero for its own reasons is reported, not run
    twice -- a retried `tleap` would redo whatever the first one did."""
    from molbuilder.diagnostics import Capabilities, set_capabilities
    from molbuilder.envs import _dispatch

    calls = []
    mgr = _fake_manager(tmp_path, "conda", 'echo "tleap: bad input" >&2\nexit 1\n')
    set_capabilities(Capabilities(runtime_config={}, conda_binary=mgr,
                                 conda_envs=frozenset({"molbuilder-MDtools"})))
    real_run = _dispatch.subprocess.run

    def counting(argv, **kw):
        calls.append(tuple(argv))
        return real_run(argv, **kw)

    monkeypatch.setattr(_dispatch.subprocess, "run", counting)

    done = _dispatch.run_in_env("molbuilder-MDtools", ["tleap"],
                                capture_output=True, text=True, timeout=120)

    assert done.returncode == 1
    assert len(calls) == 1, f"an ordinary failure was retried: {calls}"
    assert B.manager_run_unusable() is False
