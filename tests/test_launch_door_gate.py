"""The launch-door gate, one start per run + config provenance (user
decisions, 2026-08-12, 2026-10-31).

Contract: `job-contracts.md` § 2.6 (the Launch-door gate row: one launch
door; `submit` sets ``MB_LAUNCHED_BY``, and the script takes no other
start (2026-11-01); the One start per run and Session log
file rows: a run folder holding a start refuses another, and what starts
nothing writes no session log -- `project-layout.md` § 1.6.1; the one
exit trap's row: a failure of the script's own concludes, a caught signal does
not -- `project-layout.md` § 1.6.3) · the
provenance allowlist in
`runtime_config` (paths + execution-relevant values only — the machine file
also carries auth/tls, which must never reach a terminal or a shipped log).
"""
from __future__ import annotations

import os
import subprocess

import pytest

from molbuilder.runtime_config import config_provenance, format_provenance
from molbuilder.scheduler.kinds import LAUNCH_CLAIM
from support.road import SHELL_CARRIED, prepared_here

pytestmark = pytest.mark.usefixtures("in_a_folder_of_its_own")


def _prepared(tmp_path, monkeypatch, engine="siesta", activation="false"):
    """A stage prepared on the road for this machine, whose record says a
    shell enters its environment with ``activation`` -- by default one that
    fails, so a start that gets past the gate goes no further:
    ``(its run script, the run's folder, the stage's names)``."""
    _bundle, run, names = prepared_here(tmp_path, monkeypatch, activation,
                                        engine=engine)
    return run / names.run_script, run, names


def _start(sh, folder, args=(), claim=None):
    """The run script started in ``folder`` with ``args``, no terminal --
    with the launch claim ``claim``, or none."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("MB_LAUNCHED_BY", *SHELL_CARRIED)}
    if claim is not None:
        env["MB_LAUNCHED_BY"] = claim
    return subprocess.run(["bash", str(sh), *args], cwd=str(folder),
                          stdin=subprocess.DEVNULL, capture_output=True,
                          text=True, env=env)


# ---- the gate, in the emitted text and under execution ---------------- #


@pytest.mark.parametrize("claim", [None, "manual"],
                         ids=["no-claim", "a-claim-made-by-hand"])
def test_a_start_not_made_by_its_launch_is_refused(tmp_path, monkeypatch,
                                                    claim):
    """`job-contracts.md` § 2.6 -- the run script takes one start, its
    launch's: started any other way -- by hand, from cron, a header sent
    by hand, whatever it claims -- it refuses with exit 2 before anything
    is written, naming the launch that starts it (the user, 2026-11-01:
    "we always have molbuilder as the foundation of the whole system").
    THE SILENT FAILURE: a run no launch recorded, read by nothing -- or a
    stage that builds on another run started from its deck alone."""
    sh, run, names = _prepared(tmp_path, monkeypatch)
    cp = _start(sh, run, claim=claim)
    assert cp.returncode == 2, cp.stderr
    assert "molbuilder jobset launch task --stage coarse" in cp.stderr
    assert not (run / names.name(".runwrap.log")).exists()


@pytest.mark.parametrize("engine", ["siesta", "pyscf"])
def test_a_dry_run_a_person_starts_passes_the_gate(tmp_path, monkeypatch,
                                                   engine):
    """`job-contracts.md` § 2.6 -- a dry run starts nothing, so it needs no
    launch's claim: started by a person, it passes the gate, says so in its
    verdict line, and goes on to its bootstrap -- stopped here where this
    machine has no python for the job's env.  THE SILENT FAILURE: the
    preview a person is offered dying on the verdict line itself, a claim
    unset under ``set -u``."""
    sh, run, _names = _prepared(tmp_path, monkeypatch, engine,
                                activation="true")
    cp = _start(sh, run, ["--dry-run"])
    assert cp.returncode != 2, (cp.stdout, cp.stderr)
    assert "launched-by: NONE" in cp.stdout.splitlines(), (cp.stdout,
                                                          cp.stderr)


# ---- one start per run ------------------------------------------------- #

def test_a_folder_that_holds_a_start_refuses_another(tmp_path, monkeypatch):
    """A run folder holds one start, its session log saying it has one: a
    second start there refuses before anything is written, naming the
    launch that opens the next run (`project-layout.md` § 1.6.1).  THE
    SILENT FAILURE: the engine run again over the first start's output,
    its log and its conclusion overwritten."""
    sh, run, names = _prepared(tmp_path, monkeypatch)
    _start(sh, run, claim=LAUNCH_CLAIM)
    log = run / names.name(".runwrap.log")
    first = log.read_bytes()
    cp = _start(sh, run, claim=LAUNCH_CLAIM)
    assert cp.returncode == 2, cp.stderr
    assert "already holds a start" in cp.stderr
    assert "jobset launch" in cp.stderr
    assert log.read_bytes() == first


@pytest.mark.parametrize("engine,args,claim", [
    ("siesta", ["-h"], None),
    ("siesta", [], None),
    ("siesta", ["--dry-run"], None),
    ("siesta", ["--bogus"], LAUNCH_CLAIM),
    ("siesta", ["-np", "--dry-run"], LAUNCH_CLAIM),
    ("pyscf", ["-h"], None),
    ("pyscf", ["--bogus"], LAUNCH_CLAIM),
    ("pyscf", ["-omp", "--dry-run"], LAUNCH_CLAIM),
], ids=["a-help-request", "a-start-the-gate-refuses", "a-dry-run",
        "an-argument-that-does-not-read", "a-count-flag-given-no-count",
        "pyscf-a-help-request", "pyscf-an-argument-that-does-not-read",
        "pyscf-a-count-flag-given-no-count"])
def test_what_starts_nothing_writes_no_session_log(engine, args, claim,
                                                   tmp_path, monkeypatch):
    """The session log is the run's one start's (`job-contracts.md` § 2.6):
    a help request, a start the launch gate refuses, a dry run and a start
    whose arguments do not read start nothing, so none writes it, nor a
    conclusion -- and the folder stays free for the start it was opened
    for.  THE SILENT FAILURE: a mistyped flag on a dry run by hand, after
    which the corrected command is refused as a second start; or a count
    flag swallowing ``--dry-run`` and running the engine for real.  (The
    dry run stops at this test's activation, which fails -- after the point
    the log would have opened.)"""
    sh, run, names = _prepared(tmp_path, monkeypatch, engine)
    _start(sh, run, args, claim)
    assert not (run / names.name(".runwrap.log")).exists()
    assert not (run / names.name(".concluded")).exists()


# ---- the one exit trap -------------------------------------------------- #

@pytest.mark.parametrize("activation,rc,note", [
    ("false", 1, "stopped before its end"),
    ("true", 1, "no env python"),
    ("kill -TERM $$;", 143, None),
    ("kill -HUP $$;", 129, None),
], ids=["an-activation-that-fails", "a-marker-the-main-line-wrote",
        "a-caught-signal", "a-lost-terminal"])
def test_a_failure_of_its_own_concludes_a_stop_signal_does_not(
        activation, rc, note, tmp_path, monkeypatch):
    """The run script's one exit trap (`job-contracts.md` § 2.6): a start
    that ends before its engine with no signal -- its activation failing
    under ``set -e`` -- concludes, its marker the exit code and *stopped
    before its end* (`project-layout.md` § 1.6.3); a marker the main line
    wrote stands -- here the job's env gave it no python of its own
    (`job.NO_ENV_PYTHON`); and a signal -- a scheduler's stop, a lost
    terminal -- ends it through its handler, and no marker is written, a
    forced stop being its scheduler's and its monitor's to tell.  THE
    SILENT FAILURE: a run that died in its activation reading *queued* --
    or, once started, *running* -- for ever; a hang-up read as a run that
    ended on its own, exit 0, *finished*.  The marker is read through its
    one reader (`runrecord.read_concluded`)."""
    from molbuilder.runrecord import read_concluded
    sh, run, names = _prepared(tmp_path, monkeypatch, activation=activation)
    cp = _start(sh, run, claim=LAUNCH_CLAIM)
    assert cp.returncode == rc, cp.stderr
    assert (run / names.name(".runwrap.log")).is_file()
    marker = run / names.name(".concluded")
    assert marker.exists() is (note is not None), cp.stderr
    if note is not None:
        said = read_concluded(marker.read_text())
        assert said["code"] == rc and said["note"].startswith(note), said


# ---- config provenance ------------------------------------------------ #

def test_provenance_names_each_values_source(tmp_path, machine_config):
    machine_config(
        {"paths": {"projects": "/srv/projects"},
         "launch": {"mode": "direct"}})
    bundle = tmp_path / "calc"
    bundle.mkdir()
    prov = config_provenance(project_dir=bundle)
    assert prov["effective"]["paths.projects"] == {
        "value": "/srv/projects", "from": "machine"}
    assert prov["effective"]["launch.mode"] == {
        "value": "direct", "from": "machine"}
    scopes = {s["scope"]: s for s in prov["sources"]}
    assert scopes["machine"]["found"]


def test_provenance_never_carries_secret_material(tmp_path, machine_config):
    """The allowlist is the guarantee: `auth` and `tls` live in the same
    machine file and must never reach the formatted output, which lands in
    terminals, STAGE-PLAN.md and shipped logs."""
    machine_config(
        {"launch": {"mode": "direct"},
         "tls": {"key": "PEMKEYMATERIAL"}})
    text = format_provenance(config_provenance(project_dir=None))
    assert "PEMKEYMATERIAL" not in text
    assert "tls" not in text
    assert "secret" not in text.lower()
    assert "launch.mode = 'direct'" in text
