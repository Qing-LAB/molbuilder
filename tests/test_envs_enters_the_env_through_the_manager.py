"""How a step gets INTO an env: the manager's own `run`, not an imitation.

`docs/ops/installation.md` § "One door for RUNNING" (M1-M4) and
`docs/ops/env-framework.md` § 5.6.  The rule came from the user, 2026-09-12:
*"different machine could have different setups.  It might not be conda at all.
It could be mamba... use the detected activation method and the correct way to
execute the script rather than directly hacking to the directory."*

conda, mamba and micromamba take the same command in the same shape, so there
is one way in and nothing per-manager to simulate.
"""
from __future__ import annotations

import os
from pathlib import Path

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
    """An env's folder as the manager leaves one: its own Python in ``bin``
    (`installation.md` M6 -- the door starts that one, by its path)."""
    prefix = tmp_path / "envs" / "someenv"
    (prefix / "bin").mkdir(parents=True, exist_ok=True)
    (prefix / "bin" / "python").touch()
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
    (out_of_tree / "bin" / "python").touch()

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
        tmp_path, monkeypatch, capsys):
    """Some mambas' `run` (1.x; the 2.9.0 module on ASU Sol too) writes a
    shell stub containing ``exec -- "$@"``, and bash refuses `--`, so the
    command never starts.  Everything else about the managers is identical.

    The fallback is unconditional protection, so molbuilder does not need to
    know or care which version it is talking to -- and this test does not
    either.  It hands the door that failure once and checks it switched.

    No fake binary: the door's dependency is `run_streaming`, so saying "this
    is what came back" is the whole setup.
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
    # M4: the switch is REPORTED, to whoever asked -- this dispatch passed no
    # sink, and until 2026-09-14 that meant silence (review A-1.3).
    err = capsys.readouterr().err
    assert "does not work on this machine" in err
    # ...AND SAYS WHERE THE FACT WENT: the suite's machine record (the
    # conftest's isolated config) holds it from now on
    # (`configuration.md` § 5 M-2, `manager_run`).
    assert "Saved in this machine's record" in err, err

    # and a step that fails for its OWN reasons is not retried: a pip install
    # that ran and failed must not run a second time.
    B.reset_manager_run_measurement()
    attempts.clear()
    monkeypatch.setattr(B, "run_streaming",
                        lambda argv, **kw: (attempts.append(tuple(argv)),
                                            (1, "ERROR: no such package"))[1])
    B.dispatch_into_env(argv, _prefix(tmp_path))
    assert len(attempts) == 1, attempts


# --------------------------------------------------------------------------- #
#  M4 -- the machine's record remembers the measurement (2026-10-18)         #
# --------------------------------------------------------------------------- #

def test_a_record_saying_run_is_broken_is_taken_without_a_try(
        tmp_path, monkeypatch, capsys):
    """The user, 2026-10-18: "remember the measurement in the machine record
    so it stops repeating".  A record that says this manager's `run` is
    broken sends the command straight through the activation hooks: one
    launch, the wrapper's, and nothing said.

    Silent failure: the door ignores the record and pays the wasted launch
    -- and prints the note -- on every command of every process on Sol."""
    attempts = []

    def fake_stream(argv, **kw):
        attempts.append(tuple(argv))
        if argv[0] == "bash":
            return (0, "the step ran")
        return (126, "mamba-stub.sh: line 5: exec: --: invalid option")

    monkeypatch.setattr(B, "run_streaming", fake_stream)
    monkeypatch.setattr(B, "recorded_manager_run",
                        lambda: {"manager": "/m/mamba", "run_works": False})

    rc, out = B.dispatch_into_env(
        B.conda_run_argv("/m/mamba", "someenv", "python", "-V"),
        _prefix(tmp_path))

    assert (rc, out) == (0, "the step ran")
    assert [a[0] for a in attempts] == ["bash"], attempts
    assert capsys.readouterr().err == ""


def test_a_record_about_another_manager_decides_nothing(tmp_path,
                                                        monkeypatch):
    """The fact is about the binary the probe found; a record about a
    different manager -- the person pointed `envs.manager` elsewhere since
    -- does not condemn the one in hand: the door tries it, as it would with
    no record."""
    attempts = []
    monkeypatch.setattr(B, "run_streaming",
                        lambda argv, **kw: (attempts.append(tuple(argv)),
                                            (0, "ran"))[1])
    monkeypatch.setattr(B, "recorded_manager_run",
                        lambda: {"manager": "/elsewhere/mamba",
                                 "run_works": False})

    B.dispatch_into_env(B.conda_run_argv("/m/mamba", "someenv", "python"),
                        _prefix(tmp_path))

    assert attempts[0][:2] == ("/m/mamba", "run"), attempts


def test_the_measurement_is_saved_in_this_machine_s_record_as_it_is_made(
        tmp_path, monkeypatch, capsys):
    """The user, 2026-10-18: "facts has only one version.  Why do we have to
    keep updating one same fact?"  The dispatch that meets the stub writes
    `manager_run` into this machine's record, through the record's door,
    there and then -- and the next process reads it and tries nothing.

    Silent failure: the fact is measured on every command forever, or a
    person is sent to re-run a probe for a fact molbuilder already has."""
    from molbuilder.scheduler import record as R
    path = tmp_path / "environment.json"
    R.write_environment(R.Environment(scheduler=R.Scheduler(kind="workstation")),
                        path)
    monkeypatch.setattr("molbuilder.scheduler.machine_scope_path",
                        lambda: path)
    attempts = []

    def fake_stream(argv, **kw):
        attempts.append(tuple(argv))
        if argv[0] == "bash":
            return (0, "ran")
        return (126, "mamba-stub.sh: line 5: exec: --: invalid option")

    monkeypatch.setattr(B, "run_streaming", fake_stream)
    argv = B.conda_run_argv("/m/mamba", "someenv", "python", "-V")

    B.dispatch_into_env(argv, _prefix(tmp_path))

    saved = R.read_environment(path)
    assert saved.manager_run == {"manager": "/m/mamba", "run_works": False}
    assert "Saved in this machine's record" in capsys.readouterr().err
    # THE NEXT PROCESS: the record's word, no try, nothing said.
    B.reset_manager_run_measurement()
    attempts.clear()
    B.dispatch_into_env(argv, _prefix(tmp_path))
    assert [a[0] for a in attempts] == ["bash"], attempts
    assert capsys.readouterr().err == ""


def test_no_record_here_means_nothing_is_created(tmp_path, monkeypatch,
                                                 capsys):
    """A machine with no record cannot hold the fact: the door creates none
    (a record is the probe's to create, `configuration.md` M-3) and says
    so; the measurement is made again next time."""
    path = tmp_path / "environment.json"
    monkeypatch.setattr("molbuilder.scheduler.machine_scope_path",
                        lambda: path)
    assert B.remember_manager_run("/m/mamba", False) is None
    assert not path.exists()
    # ...AND THE NOTE SAYS SO, naming the probe that writes a record.
    monkeypatch.setattr(B, "run_streaming", lambda argv, **kw: (
        (0, "ran") if argv[0] == "bash"
        else (126, "mamba-stub.sh: line 5: exec: --: invalid option")))
    B.dispatch_into_env(B.conda_run_argv("/m/mamba", "someenv", "python"),
                        _prefix(tmp_path))
    err = capsys.readouterr().err
    assert "no record" in err and "jobset probe" in err, err
    assert not path.exists()


def test_the_record_is_read_through_its_door_once(config_root, monkeypatch):
    """`configuration.md` M-4: no consumer reads the file.  The door asks
    `read_environment(machine_scope_path())`, once per process, and takes
    the `manager_run` block as the record holds it."""
    from molbuilder import scheduler
    from molbuilder.scheduler.record import (Environment, Scheduler,
                                             write_environment)
    # THE RECORD, as the probe makes one, in this test's machine folder...
    record = write_environment(
        Environment(scheduler=Scheduler(kind="workstation")),
        scheduler.machine_scope_path())
    # ...the measurement saved into it by the door that saves it, then
    # forgotten by this process, as a fresh one starts.
    assert B.remember_manager_run("/m/mamba", False) == record
    B.reset_manager_run_measurement()
    # Each read the door makes, counted -- the reader itself still reads.
    reads = []
    real_read = scheduler.read_environment

    def counted(path):
        reads.append(path)
        return real_read(path)

    monkeypatch.setattr("molbuilder.scheduler.read_environment", counted)

    assert B.recorded_manager_run() == {"manager": "/m/mamba",
                                        "run_works": False}
    assert B.recorded_manager_run()["run_works"] is False
    assert reads == [record], reads


def test_the_probe_s_measurement_is_the_door_s_own():
    """`measure_manager_run` reads the same signature the door reads: the stub
    is `False`, a command that ran is `True`, any other failure is unknown
    (`None`) -- the record then says nothing, and the door measures."""
    answers = {}

    def fake_stream(argv, **kw):
        answers["argv"] = tuple(argv)
        return answers["reply"]

    import pytest as _pt
    with _pt.MonkeyPatch.context() as mp:
        mp.setattr(B, "run_streaming", fake_stream)
        answers["reply"] = (126, "x.sh: line 5: exec: --: invalid option")
        assert B.measure_manager_run("/m/mamba", "/envs/here") is False
        assert answers["argv"][:5] == ("/m/mamba", "run", "--prefix",
                                       "/envs/here", "--no-capture-output")
        assert answers["argv"][5:] == ("true",)
        answers["reply"] = (0, "")
        assert B.measure_manager_run("/m/mamba", "/envs/here") is True
        answers["reply"] = (1, "some other failure")
        assert B.measure_manager_run("/m/mamba", "/envs/here") is None


def test_every_step_gets_the_clean_slate_only_builds_used_to_get(
        tmp_path, monkeypatch):
    """H3's live consequence.  `run_step`'s `env` parameter existed and no
    caller passed it, while `builds.py` passed `build_subprocess_env()` at two
    sites -- so every pip step and every extra step ran with exactly the host
    `CPATH` / `CFLAGS` / `CUDA_HOME` / `OMPI_*` leakage that module exists to
    strip, two functions away from the stripper."""
    monkeypatch.setenv("CPATH", "/usr/include/evil")
    monkeypatch.setenv("CUDA_HOME", "/opt/cuda-system")
    # npm reads its variables in either case: a host's upper-case cache
    # would compete with the env's.
    monkeypatch.setenv("NPM_CONFIG_CACHE", "/home/someone/.npm")
    monkeypatch.setenv("OMPI_MCA_btl", "self,tcp")
    monkeypatch.setenv("KEEP_ME", "yes")
    prefix = _prefix(tmp_path)

    env = B.env_for_step(prefix)

    assert "CPATH" not in env and "CUDA_HOME" not in env
    assert "OMPI_MCA_btl" not in env
    assert env["KEEP_ME"] == "yes"
    # Temp and caches inside the prefix, so removing the env really cleans up
    # and a small /tmp on a cluster cannot fail a wheel build -- npm's too,
    # whose default is ~/.npm (`env-framework.md` § 3.2a).
    assert env["TMPDIR"] == f"{prefix}/var/tmp"
    assert env["PIP_CACHE_DIR"] == f"{prefix}/var/cache/pip"
    assert env["npm_config_cache"] == f"{prefix}/var/cache/npm"
    assert "NPM_CONFIG_CACHE" not in env
    assert os.path.isdir(env["TMPDIR"])


#: (the step, its role, whether the env already has its folders, the temp
#: folder handed): a step that installs makes the env's folders; a verify
#: creates nothing and is handed only those that are there, the host's own
#: temp folder otherwise (`env-framework.md` § 5.6).
STEP_FOLDER_ROWS = [
    ("an install step, an env with no folders yet", "PACKAGES", False, "env"),
    ("a verify, an env with its folders", "VERIFY", True, "env"),
    ("a verify, an env without them", "VERIFY", False, "host"),
]


@pytest.mark.parametrize("what, role, has_folders, handed", STEP_FOLDER_ROWS,
                         ids=[r[0] for r in STEP_FOLDER_ROWS])
def test_a_step_is_handed_its_envs_folders_as_its_role_allows(
        tmp_path, monkeypatch, what, role, has_folders, handed):
    """`env-framework.md` § 5.6: temp and caches kept inside the prefix; a
    verify, "which creates nothing in the env it reads, is handed only those
    folders that are there" -- asserted through `run_step`, the door
    `doctor` and `install` ask, its dispatch faked as the test above fakes
    it (the step's environment is what it receives).

    THE SILENT FAILURE: a verify told TMPDIR=<prefix>/var/tmp in an env
    with no such folder -- anything writing a temp file fails, and a
    healthy env reads FAILED; or an install step handed a folder never
    made."""
    from molbuilder.envs import install as I
    seen = {}

    def fake_dispatch(argv, prefix, *, env=None, **kw):
        seen["env"] = env
        return (0, "ok")

    monkeypatch.setattr(I._builds, "dispatch_into_env", fake_dispatch)
    host_tmp = tmp_path / "host-tmp"
    host_tmp.mkdir()
    monkeypatch.setenv("TMPDIR", str(host_tmp))
    prefix = _prefix(tmp_path)
    if has_folders:
        Path(prefix, "var", "tmp").mkdir(parents=True)
    step = I.InstallStep(label="a step", role=getattr(I.StepRole, role),
                         argv=B.conda_run_argv("/m", "someenv", "true"))

    I.run_step(step, prefix=prefix)

    tmp = seen["env"]["TMPDIR"]
    assert tmp == (f"{prefix}/var/tmp" if handed == "env" else str(host_tmp))
    assert os.path.isdir(tmp), tmp
    if role == "VERIFY" and not has_folders:
        assert not Path(prefix, "var").exists(), "a verify made folders"


def test_run_step_hands_that_environment_to_the_door(tmp_path, monkeypatch):
    """The door hands the step that environment -- asserted through
    `run_step`, because "the function exists" was never the gap."""
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


def test_a_package_install_into_an_env_is_addressed_at_its_directory(
        tmp_path, monkeypatch):
    """`repair` installs into an env conda may not be able to find by name --
    one created with `--prefix` outside envs_dirs (the contract's supported
    case).  conda resolves `-n` against envs_dirs only (its own
    `locate_prefix_by_name`), so the door re-addresses `install -n` the way it
    re-addresses `run -n` (review B-L2, 2026-09-14).  Asserted through
    `run_step`, the way `repair` asks."""
    from molbuilder.envs import install as I
    from molbuilder.envs.recipes import recipe_by_name

    launched = []

    def fake_stream(argv, **kw):
        launched.append(tuple(argv))
        return (0, "done")

    monkeypatch.setattr(B, "run_streaming", fake_stream)
    # Under the suite's sandbox home (testing.md § 2b): the step makes its
    # temp and cache folders under the prefix it is given.
    prefix = str(Path.home() / "scratch" / "molbuilder-pySCF")
    step = I.conda_step_for(["numpy"], recipe_by_name("molbuilder-pySCF"),
                            "/opt/mgr/bin/conda", "molbuilder-pySCF")
    done = I.run_step(step, prefix=prefix)
    assert done.outcome is I.Outcome.OK
    (argv,) = launched
    assert argv[1] == "install" and "-n" not in argv, argv
    assert argv[2:4] == ("--prefix", prefix), argv

    # ...and so is the removal `--clean` dispatches.
    launched.clear()
    I.run_step(I.remove_step_for("molbuilder-pySCF", "/opt/mgr/bin/conda"),
               prefix=prefix)
    (argv,) = launched
    assert argv[1:5] == ("env", "remove", "--prefix", prefix), argv

    # ...while an inner command's own `-n` is not the address and is left
    # alone: a scan for the first `-n` rewrote `mpirun -n 2` inside a
    # prefix-addressed `run` (review A-1.2, measured 2026-09-14).
    launched.clear()
    B.dispatch_into_env(
        B.conda_run_prefix_argv("/opt/mgr/bin/conda", prefix,
                                "mpirun", "-n", "2", "siesta", "--version"),
        prefix)
    (argv,) = launched
    assert argv[5:8] == ("mpirun", "-n", "2"), argv


# --------------------------------------------------------------------------- #
#  S17 -- the tool router gets the same answer                                #
# --------------------------------------------------------------------------- #

def _machine_with(env_name, prefix, manager="/opt/mgr/bin/conda"):
    from molbuilder.diagnostics import Capabilities, set_capabilities
    set_capabilities(Capabilities(
        runtime_config={}, conda_binary=manager,
        conda_envs={env_name: prefix}))


def test_the_tool_router_enters_through_the_door_addressed_by_prefix(
        tmp_path, monkeypatch):
    """`run_tool` takes the same route as an install step: the door, with
    the env's directory (M1, M2).  Until 2026-09-13 the router kept its own
    copy of the broken-`run` detection and wrapper, addressed the env by
    NAME, and reached into `install` for a prefix when its copy needed one.

    Asked through `run_tool` with a fake door: what the door was handed, and
    what the caller got back."""
    from molbuilder import envs

    seen = {}

    def fake_door(argv, prefix, *, cwd=None, sink=None, timeout=None, **kw):
        seen.update(argv=tuple(argv), prefix=prefix, cwd=cwd, sink=sink,
                    timeout=timeout)
        return (0, "tleap: done\n")

    monkeypatch.setattr(B, "dispatch_into_env", fake_door)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))    # no host tleap
    # The env as its recipe leaves it: its own tleap in its bin, which the
    # router names by its path (`installation.md` M6).
    mdtools = tmp_path / "envs" / "mdtools"
    (mdtools / "bin").mkdir(parents=True)
    (mdtools / "bin" / "tleap").touch()
    _machine_with("molbuilder-MDtools", str(mdtools))

    done = envs.run_tool("tleap", ["-f", "in.leap"], cwd=tmp_path, timeout=7)

    assert seen["argv"] == B.conda_run_argv(
        "/opt/mgr/bin/conda", "molbuilder-MDtools",
        str(mdtools / "bin" / "tleap"), "-f", "in.leap")
    assert seen["prefix"] == str(mdtools)
    assert seen["cwd"] == tmp_path and seen["timeout"] == 7
    assert seen["sink"] is None, "a tool call is read, not watched"
    assert done.returncode == 0 and done.stdout == "tleap: done\n"


def test_a_tool_that_overruns_its_timeout_is_stopped_at_it(tmp_path):
    """A tool that hangs is stopped at its limit -- it and what it started --
    and its caller gets `subprocess.TimeoutExpired`, as from
    `subprocess.run`.  Through the real door; the env manager, the outside
    world, stood in by a script that starts a child and never ends.

    THE SILENT FAILURE (to 2.70.16): the door read the tool's output until
    the tool ended and only then started the clock, so a hung `tleap`, or a
    hung conda or pip step of an install, ran for ever.  Fails on the clock
    started after the reading, and on the manager stopped without its
    child, which keeps the output open."""
    import subprocess
    import time
    from molbuilder import envs

    manager = tmp_path / "conda"
    manager.write_text("#!/bin/sh\necho started\nsleep 30\necho late\n")
    manager.chmod(0o755)
    _machine_with("molbuilder-MDtools", _prefix(tmp_path), str(manager))

    began = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        envs.run_in_env("molbuilder-MDtools", ["tleap"], timeout=1)
    assert time.monotonic() - began < 10


# --------------------------------------------------------------------------- #
#  What the door does with the output                                         #
# --------------------------------------------------------------------------- #

def test_no_sink_means_captured_only(capsys):
    """`run_streaming(sink=None)` hands the output back and shows nothing;
    a caller with a person watching passes the stream it wants.

    Until 2026-09-13 ``None`` meant ``sys.stderr``, so the one caller that
    said nothing -- `doctor`'s verify probe, whose comment promised "captured
    rather than streamed" -- had the probe's output land in the middle of the
    report.
    """
    import sys

    rc, out = B.run_streaming(["echo", "quiet-line"])
    assert rc == 0 and "quiet-line" in out
    assert "quiet-line" not in capsys.readouterr().err

    rc, out = B.run_streaming(["echo", "shown-line"], sink=sys.stderr)
    assert rc == 0 and "shown-line" in out
    assert "shown-line" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
#  M6 -- every Python is the env's own, named by its path (2026-11-21)        #
# --------------------------------------------------------------------------- #
#
# The user, 2026-11-21: "make sure that no bare python or out-of-env python is
# ever tried. period."; "apply the same rule to ctest, make, mpirun and tleap
# too".  Each place molbuilder starts a Python or an env's program, run for
# real -- this machine's own manager, an env it made -- with decoys of each
# first on PATH (`Decoys.NAMES`) that leave a mark and fail.  None may be
# reached.
# Measured 2026-11-21, conda 26.7.1: called from a shell with an env active,
# the manager's `run` puts the entered env's `bin` where the active env's
# stood on PATH, not first -- so a bare `python` reached the decoy for an env
# holding none AND for one holding its own.

import subprocess
import sys

from molbuilder.envs import install as I
from molbuilder.envs.recipes import builtin_recipes


@pytest.fixture(scope="module")
def real_manager():
    """The manager the product would use (`diagnostics`' resolver)."""
    from molbuilder import diagnostics as D
    found, _source = D._find_conda_binary()
    if not found:
        pytest.skip("no env manager on this machine")
    return found


def _made_env(manager, prefix):
    """An env this machine's manager makes holding nothing: no Python."""
    done = subprocess.run(
        [manager, "create", "-y", "-q", "--offline", "-p", str(prefix)],
        capture_output=True, text=True, timeout=180)
    assert done.returncode == 0, done.stderr
    assert not (prefix / "bin" / "python").exists()
    return prefix


@pytest.fixture(scope="module")
def env_without_python(real_manager, tmp_path_factory):
    return _made_env(real_manager,
                     tmp_path_factory.mktemp("m6") / "envs" / "nopython")


@pytest.fixture(scope="module")
def env_with_python(real_manager, tmp_path_factory):
    """An env the manager made, its own `python` and `python3` in `bin`."""
    prefix = _made_env(real_manager,
                       tmp_path_factory.mktemp("m6") / "envs" / "withpython")
    (prefix / "bin").mkdir(exist_ok=True)
    for name in ("python", "python3"):
        (prefix / "bin" / name).symlink_to(sys.executable)
    return prefix


class Decoys:
    """A `python`, `python3`, `jupyter`, `ctest`, `make`, `mpirun` and
    `tleap` first on PATH, each leaving a mark and failing -- the programs
    M6 says are never tried."""

    NAMES = ("python", "python3", "python3.12", "jupyter",
             "ctest", "make", "mpirun", "mpiexec", "tleap", "siesta", "npm")

    def __init__(self, where: Path):
        self.folder = where / "decoys"
        self.folder.mkdir()
        self.marks = where / "reached"
        for name in self.NAMES:
            exe = self.folder / name
            exe.write_text(f'#!/bin/sh\necho "{name}" >> "{self.marks}"\n'
                           f'echo "DECOY {name} REACHED" >&2\nexit 97\n')
            exe.chmod(0o755)

    def reached(self):
        return (self.marks.read_text().split() if self.marks.exists()
                else [])


@pytest.fixture
def decoys(tmp_path, monkeypatch):
    """...in a shell where envs are active, as molbuilder's own always is --
    base, and the host env stacked on it: their marks set, their `bin`s on
    PATH behind the decoys."""
    found = Decoys(tmp_path)
    folders = []
    for name in ("base-env", "active-env"):
        active = tmp_path / name
        (active / "bin").mkdir(parents=True)
        (active / "conda-meta").mkdir()
        (active / "conda-meta" / "history").touch()
        folders.append(active)
    base, active = folders
    monkeypatch.setenv("PATH", os.pathsep.join(
        [str(found.folder), str(active / "bin"), str(base / "bin"),
         os.environ.get("PATH", "")]))
    monkeypatch.setenv("CONDA_PREFIX", str(active))
    monkeypatch.setenv("CONDA_PREFIX_1", str(base))
    monkeypatch.setenv("CONDA_SHLVL", "2")
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "active-env")
    return found


DOOR_ROWS = [
    ("a bare python", ("python", "-c", "print('ran')")),
    ("a bare python3", ("python3", "-c", "print('ran')")),
    ("a bare python3.N", ("python3.12", "-c", "print('ran')")),
]


@pytest.mark.parametrize("what, cmd", DOOR_ROWS,
                         ids=[r[0] for r in DOOR_ROWS])
def test_an_env_without_its_own_python_is_refused_and_no_other_is_tried(
        what, cmd, real_manager, env_without_python, decoys):
    """The door, for a command it carries: nothing starts, and the refusal
    names the env and where the repair is named."""
    prefix = str(env_without_python)
    rc, said = B.dispatch_into_env(
        B.conda_run_prefix_argv(real_manager, prefix, *cmd), prefix)
    assert decoys.reached() == [], f"{what}: {said}"
    assert rc is None, said
    assert prefix in said and "no Python of its own" in said, said
    assert "doctor" in said, said


def test_a_path_to_another_python_is_refused(real_manager, env_with_python,
                                             decoys):
    """A Python named by a path outside the env's `bin` is not the env's,
    however the env is entered."""
    prefix = str(env_with_python)
    rc, said = B.dispatch_into_env(
        B.conda_run_prefix_argv(real_manager, prefix,
                                str(decoys.folder / "python3"), "-V"),
        prefix)
    assert decoys.reached() == [], said
    assert rc is None and "not a Python of the environment" in said, said


@pytest.mark.parametrize("word", ["python", "python3"])
def test_an_env_s_own_python_is_the_one_started(word, real_manager,
                                                env_with_python, decoys):
    prefix = str(env_with_python)
    rc, out = B.dispatch_into_env(
        B.conda_run_prefix_argv(real_manager, prefix, word, "-c",
                                "import sys; print(sys.executable)"),
        prefix)
    assert decoys.reached() == []
    assert rc == 0, out
    assert str(env_with_python / "bin" / word) in out, out


RECIPES_WITH_A_SELF_TEST = [r for r in builtin_recipes() if r.verify_argv]


@pytest.mark.parametrize("recipe", RECIPES_WITH_A_SELF_TEST,
                         ids=[r.name for r in RECIPES_WITH_A_SELF_TEST])
def test_a_recipe_s_self_test_reaches_no_other_program(
        recipe, real_manager, env_with_python, decoys):
    """...and on an env that holds its Python but none of the programs the
    self-test asks -- `siesta`, `tleap`, JupyterLab: the program half
    names the env's own too, so no other answers and calls the env
    verified."""
    step = I.verify_step_for(recipe, real_manager, "withpython")
    done = I.run_step(step, prefix=str(env_with_python), timeout=120)
    assert decoys.reached() == [], done.output


@pytest.mark.parametrize("recipe", RECIPES_WITH_A_SELF_TEST,
                         ids=[r.name for r in RECIPES_WITH_A_SELF_TEST])
def test_a_recipe_s_self_test_reaches_no_other_python(
        recipe, real_manager, env_without_python, decoys):
    """`doctor`'s and `install`'s check of an env (`install.verify_step_for`
    through `run_step`), on an env without Python: it fails, and no other
    Python or `jupyter` answered for the env."""
    step = I.verify_step_for(recipe, real_manager, "nopython")
    done = I.run_step(step, prefix=str(env_without_python), timeout=120)
    assert decoys.reached() == [], done.output
    assert not done.outcome.is_success, done.output


def test_a_package_install_reaches_no_other_python(real_manager,
                                                   env_without_python,
                                                   decoys):
    """`python -m pip install` run by whichever Python answered installs
    there -- the host env, `~/.local` -- and leaves the env as it was."""
    host = next(r for r in builtin_recipes() if r.category is None)
    step = I.pip_steps_for(host, real_manager, "nopython")[0]
    done = I.run_step(step, prefix=str(env_without_python), timeout=120)
    assert decoys.reached() == [], done.output
    assert not done.outcome.is_success, done.output


def test_an_extra_step_reaches_no_other_python(real_manager, tmp_path,
                                               decoys):
    """A step written as a shell line names the env's Python itself (the
    browser the test tooling installs).  Its own env: the step writes a hook
    into the env it is run in."""
    prefix = _made_env(real_manager, tmp_path / "envs" / "extra")
    host = next(r for r in builtin_recipes() if r.category is None)
    steps = [s for s in I.extra_steps_for(host, real_manager, "extra",
                                          include_opt_in=True)
             if "playwright" in " ".join(s.argv)]
    assert steps, "the host recipe's browser step"
    done = I.run_step(steps[0], prefix=str(prefix), timeout=120)
    assert decoys.reached() == [], done.output
    assert not done.outcome.is_success, done.output


def test_the_notebook_is_started_on_no_other_python(real_manager,
                                                    env_without_python,
                                                    tmp_path, decoys, capfd):
    """The notebook's command line (`jupyter.notebook_argv`), through the
    door as the shepherd sends it -- no pipe, its stderr the notebook log:
    an env without Python starts nothing, no `jupyter` from PATH, and the
    log says why (`web/jupyter.md` § 7)."""
    from molbuilder.jupyter import notebook_argv
    argv = notebook_argv(real_manager, "nopython", port=1,
                         root_dir=str(tmp_path))
    rc, said = B.dispatch_into_env(argv, str(env_without_python),
                                   inherit_stdio=True)
    assert decoys.reached() == [], said
    assert rc is None and "no Python of its own" in said, said
    assert said in capfd.readouterr().err


def test_an_env_whose_folder_is_not_known_is_refused(real_manager, decoys):
    """Its Python cannot be named, so nothing starts -- not the manager's
    run by name, which would leave the word to PATH."""
    rc, said = B.dispatch_into_env(
        B.conda_run_argv(real_manager, "an-env-nowhere", "python", "-c",
                         "print('ran')"), None)
    assert decoys.reached() == [], said
    assert rc is None and "is not known" in said, said


def test_a_program_the_env_holds_is_the_one_found_first(real_manager,
                                                       env_with_python,
                                                       decoys):
    """The door hands the manager no active env, so its activation puts
    the env's `bin` first on PATH: a program the env holds -- its `make`,
    its `mpirun` -- answers ahead of any other (the user, 2026-11-21:
    "apply the same rule to ctest, make, mpirun and tleap too")."""
    make = env_with_python / "bin" / "make"
    make.write_text("#!/bin/sh\necho \"the env's make\"\n")
    make.chmod(0o755)
    prefix = str(env_with_python)
    rc, out = B.dispatch_into_env(
        B.conda_run_prefix_argv(real_manager, prefix, "make"), prefix)
    assert decoys.reached() == [], out
    assert rc == 0 and "the env's make" in out, out


def _routing_mdtools_to(manager, prefix):
    from molbuilder.diagnostics import Capabilities, set_capabilities
    set_capabilities(Capabilities(
        runtime_config={}, conda_binary=manager,
        conda_envs={} if prefix is None
        else {"molbuilder-MDtools": str(prefix)}))


def test_a_routed_tool_runs_from_its_env_by_its_path(real_manager,
                                                     env_with_python,
                                                     decoys):
    """`envs.run_tool` -- how a structure build runs `tleap` -- runs the
    env's own, never one the host's PATH holds."""
    from molbuilder import envs
    tleap = env_with_python / "bin" / "tleap"
    tleap.write_text("#!/bin/sh\necho \"the env's tleap\"\n")
    tleap.chmod(0o755)
    _routing_mdtools_to(real_manager, env_with_python)
    done = envs.run_tool("tleap", ["-f", "build.in"], timeout=60)
    assert decoys.reached() == [], done.stdout
    assert done.returncode == 0 and "the env's tleap" in done.stdout


@pytest.mark.parametrize("where", ["an env without it", "no env at all"])
def test_a_routed_tool_its_env_lacks_is_refused(where, real_manager,
                                                env_without_python, decoys):
    """Where its env lacks the tool, or no env holds it, nothing runs --
    the host's PATH is never the answer -- and the refusal names what
    repairs or installs it."""
    from molbuilder import envs
    _routing_mdtools_to(real_manager, env_without_python
                        if where == "an env without it" else None)
    with pytest.raises(FileNotFoundError) as refused:
        envs.run_tool("tleap", ["-f", "build.in"], timeout=60)
    assert decoys.reached() == []
    said = str(refused.value)
    assert "no other `tleap` is tried" in said, said
    assert ("doctor" if where == "an env without it" else "install") in said


def _a_built_gpu_env(prefix):
    """An env folder with what each probe looks for before its program --
    the build's CTest file, ELPA's Makefile and GPU validator."""
    from molbuilder.envs.recipes import recipe_by_name
    recipe = recipe_by_name("molbuilder-siesta-gpu")
    paths = B.resolve_paths(recipe.build_spec, str(prefix))
    siesta = paths.component_build("siesta")
    siesta.mkdir(parents=True, exist_ok=True)
    (siesta / "CTestTestfile.cmake").touch()
    elpa = paths.component_build("elpa")
    elpa.mkdir(parents=True, exist_ok=True)
    (elpa / "Makefile").touch()
    (elpa / "validate_real_double_eigenvectors_1stage_gpu_random").touch()
    return recipe


PROBE_ROWS = [("ctest", (), "_probe_siesta_ctest_simple"),
              ("make", (), "_probe_elpa_make_check"),
              ("mpiexec", ("make",), "_probe_elpa_make_check"),
              ("mpirun", (), "_probe_elpa_gpu_codepath")]


@pytest.mark.parametrize("program, holds, probe", PROBE_ROWS,
                         ids=[r[0] for r in PROBE_ROWS])
def test_a_validator_probe_runs_no_other_program(program, holds, probe,
                                                 tmp_path, decoys):
    """`envs validate`'s probes run the env's own `ctest`, `make`,
    `mpiexec` and `mpirun`, by path -- the build's cmake, the build's make,
    the MPI the validators were built against.  An env without one fails
    the probe, said, the repair named; the host's is never tried."""
    from molbuilder.envs import validate as V
    prefix = tmp_path / "envs" / "gpu"
    (prefix / "bin").mkdir(parents=True)
    for name in holds:
        (prefix / "bin" / name).touch()
    found = getattr(V, probe)(str(prefix), _a_built_gpu_env(prefix))
    assert decoys.reached() == [], found
    assert found.passed is False, found
    assert f"no {program} of its own" in found.detail, found.detail
    assert "doctor" in found.detail, found.detail


def test_elpa_s_check_runs_its_validators_under_the_env_s_mpi(
        real_manager, tmp_path, decoys):
    """ELPA's `make check` starts each validator with a bare `mpiexec` (its
    Makefile's ``MPI_BINARY``): the probe enters the env through the door,
    so the env's own answers -- never the machine's MPI, which the
    validators were not built against.  The env's `make` stood in by one
    that starts `mpiexec` as ELPA's check scripts do (a named need: a real
    check is the GPU build's thirty minutes)."""
    from molbuilder.diagnostics import Capabilities, set_capabilities
    from molbuilder.envs import validate as V
    prefix = _made_env(real_manager, tmp_path / "envs" / "gpu")
    (prefix / "bin").mkdir(exist_ok=True)
    (prefix / "bin" / "make").write_text("#!/bin/sh\nexec mpiexec -n 2 true\n")
    (prefix / "bin" / "mpiexec").write_text(
        "#!/bin/sh\necho \"the env's mpiexec\"\n")
    for name in ("make", "mpiexec"):
        (prefix / "bin" / name).chmod(0o755)
    set_capabilities(Capabilities(runtime_config={},
                                  conda_binary=real_manager, conda_envs={}))
    found = V._probe_elpa_make_check(str(prefix), _a_built_gpu_env(prefix))
    assert decoys.reached() == [], found
    assert found.passed is True, found
    assert "the env's mpiexec" in found.output, found.output


def test_the_cuda_probe_reads_no_other_python(env_without_python, decoys):
    """`envs validate`'s CUDA probe loads the driver's library on the env's
    Python.  The
    driver's own tool stood in (a named need: the probe reaches its Python
    only after `nvidia-smi` answers)."""
    from molbuilder.envs import validate as V
    smi = decoys.folder / "nvidia-smi"
    smi.write_text("#!/bin/sh\necho 'Stand-in GPU, 8.0, 550.00'\n")
    smi.chmod(0o755)
    found = V._probe_cuda_stack(str(env_without_python), None)
    assert decoys.reached() == [], found
    assert found.passed is False
    assert "no Python of its own" in found.detail, found.detail
    assert "doctor" in found.detail, found.detail


def test_a_written_script_starts_no_python_but_the_env_s(env_with_python,
                                                         tmp_path, decoys):
    """The Makov-Payne correction script molbuilder writes beside a charged
    SIESTA deck: no interpreter line, and each run line it prints starts the
    env's own Python -- with no env entered, none."""
    from molbuilder.siesta.makov_payne import render_correction_script
    from molbuilder.runfiles import MAKOV_PAYNE_SCRIPT
    text = render_correction_script("mol", 1, stage="scf")
    assert not text.startswith("#!"), text.splitlines()[0]
    (tmp_path / MAKOV_PAYNE_SCRIPT).write_text(text)
    lines = [ln.strip() for ln in text.splitlines()
             if MAKOV_PAYNE_SCRIPT in ln and ln.strip().startswith('"')]
    assert lines, "the script's run lines"
    base = {k: v for k, v in os.environ.items() if k != "CONDA_PREFIX"}
    for line in lines:
        bare = subprocess.run(["bash", "-c", line], env=base, cwd=tmp_path,
                              capture_output=True, text=True, timeout=60)
        assert bare.returncode != 0 and decoys.reached() == [], bare.stderr
        entered = subprocess.run(
            ["bash", "-c", line],
            env={**base, "CONDA_PREFIX": str(env_with_python)},
            cwd=tmp_path, capture_output=True, text=True, timeout=60)
        assert decoys.reached() == [], entered.stderr
        # Started on the env's Python: the script's own word -- no run
        # output, or no monitor's bundle, beside it -- not the shell's.
        assert entered.stderr.startswith("error: "), entered.stderr


def test_the_notebook_s_kernel_is_its_env_s_own_python(tmp_path):
    """The notebook server's environment (`jupyter.notebook_environment`)
    makes Jupyter take the env's own kernel ahead of one in the person's
    folders: a `--user` kernel installed once, on another Python, is never
    the one a cell runs on (`web/jupyter.md` § 7, M6).  The notebook env
    of this machine, its own Python; a person's home holding such a
    kernel.  Run with no activation, it stands in for an env the account
    does not own -- a lab-shared one -- where Jupyter's own preference for
    the env does not hold."""
    import json
    from molbuilder import diagnostics as D
    from molbuilder.envs.recipes import effective_name, recipe_of_category
    from molbuilder.jupyter import notebook_environment
    caps = D.detect()
    prefix = caps.env_prefix(effective_name(recipe_of_category("jupyter"),
                                            caps))
    if not prefix or not (Path(prefix) / "bin" / "python").is_file():
        pytest.skip("no notebook env on this machine")
    home = tmp_path / "home"
    theirs = home / ".local" / "share" / "jupyter" / "kernels" / "python3"
    theirs.mkdir(parents=True)
    (theirs / "kernel.json").write_text(json.dumps({
        "argv": ["/usr/bin/python3", "-m", "ipykernel_launcher", "-f",
                 "{connection_file}"],
        "display_name": "Python 3 (their own)", "language": "python"}))
    base = {"HOME": str(home), "PATH": "/usr/bin:/bin",
            "LANG": os.environ.get("LANG", "C.UTF-8")}
    listed = subprocess.run(
        [str(Path(prefix) / "bin" / "python"), "-m", "jupyter",
         "kernelspec", "list", "--json"],
        env=notebook_environment("a-token", base), cwd=tmp_path,
        capture_output=True, text=True, timeout=120)
    assert listed.returncode == 0, listed.stderr
    where = json.loads(listed.stdout)["kernelspecs"]["python3"]
    assert Path(where["resource_dir"]).is_relative_to(prefix), where
