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

conda, mamba and micromamba take the same command in the same shape, so there
is one way in and nothing per-manager to simulate.  molbuilder used to carry a
re-implementation of `conda activate` as a fallback for one bug in mamba 1.x;
that is deleted -- a manager whose `run` does not work is one to replace, and
the failure now says so and names `envs.manager`.
"""
from __future__ import annotations

import os

import pytest

from molbuilder.envs import builds as B


@pytest.fixture(autouse=True)
def _forget_the_broken_manager():
    """The fallback is remembered per PROCESS, so a test that provokes it would
    otherwise decide the next test's dispatch."""
    B.reset_manager_run_measurement()
    yield
    B.reset_manager_run_measurement()


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


def test_the_manager_s_own_command_goes_through_untouched(tmp_path,
                                                          monkeypatch):
    """`conda create` / `conda install -n ...` ARE the manager's command: there
    is no env to enter yet, so nothing is re-addressed and nothing is added."""
    launched = []
    monkeypatch.setattr(B, "run_streaming",
                        lambda argv, **kw: (launched.append(tuple(argv)),
                                            (0, ""))[1])
    create = ("/m", "create", "-n", "someenv", "-y", "python=3.12", "pip")

    B.dispatch_into_env(create, _prefix(tmp_path))

    assert launched[0] == create


def test_a_broken_manager_run_falls_back_and_the_step_still_runs(
        tmp_path, monkeypatch):
    """mamba 1.x's `run` writes a shell stub containing ``exec -- "$@"``, and
    bash refuses `--`, so the command never starts.  2.x fixed that line;
    everything else about the two is identical.

    The fallback is unconditional protection, so molbuilder does not need to
    know or care which version it is talking to -- and this test does not
    either.  It hands the door that failure once and checks it switched.

    No fake binary: the door's dependency is `run_streaming`, so saying "this
    is what came back" is the whole setup.  There used to be a shell script
    per case, a PATH and a log to read the answer out of, for this.
    """
    attempts = []

    def fake_stream(argv, **kw):
        attempts.append(tuple(argv))
        if argv[0] == "bash":
            return (0, "the step ran")
        return (126, "mamba-stub.sh: line 5: exec: --: invalid option")

    monkeypatch.setattr(B, "run_streaming", fake_stream)
    argv = B.conda_run_argv("/m/mamba", "someenv", "python", "-V")

    rc, out = B.dispatch_into_env(argv, _prefix(tmp_path))

    assert (rc, out) == (0, "the step ran")
    assert attempts[0][:2] == ("/m/mamba", "run"), attempts
    assert attempts[1][0] == "bash", "it did not fall back"
    assert B.manager_run_unusable() is True, (
        "measured but not remembered -- every later step pays the same "
        "wasted launch")

    # and a step that fails for its OWN reasons is not retried: a pip install
    # that ran and failed must not run a second time.
    B.reset_manager_run_measurement()
    attempts.clear()
    monkeypatch.setattr(B, "run_streaming",
                        lambda argv, **kw: (attempts.append(tuple(argv)),
                                            (1, "ERROR: no such package"))[1])
    B.dispatch_into_env(argv, _prefix(tmp_path))
    assert len(attempts) == 1, attempts


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

