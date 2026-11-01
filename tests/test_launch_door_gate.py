"""The launch-door gate, one start per run + config provenance (user
decisions, 2026-08-12, 2026-10-31).

Contract: `job-contracts.md` § 2.6 (the Launch-door gate row: one launch
door; `submit` sets ``MB_LAUNCHED_BY``, a bare call warns or refuses,
``manual`` is the logged override; the One start per run and Session log
file rows: a run folder holding a start refuses another, and what starts
nothing writes no session log -- `project-layout.md` § 1.6.1) · the
provenance allowlist in
`runtime_config` (paths + execution-relevant values only — the machine file
also carries auth/tls, which must never reach a terminal or a shipped log).
"""
from __future__ import annotations

import os
import subprocess


from molbuilder.runtime_config import config_provenance, format_provenance
from molbuilder.runwrap import write_run_wrapper
from molbuilder.jobset.model import Resources
import pytest
from molbuilder.runfiles import RunNames
from support.machines import SLURM


@pytest.fixture(autouse=True)
def _sandbox(tmp_path, monkeypatch):
    """An isolated cwd; conftest isolates the config root, so nothing here
    reads the developer's config."""
    monkeypatch.chdir(tmp_path)


#: Each engine's deck, as small as its run script renders over.
_DECKS = {"siesta": (".fdf", "SystemLabel JOB\n"),
          "pyscf": (".py", 'JOB = "JOB"\nimport pyscf\n')}


def _wrapper(tmp_path, engine="siesta"):
    names = RunNames("JOB", "01_coarse")
    suffix, text = _DECKS[engine]
    deck = tmp_path / names.name(suffix)
    deck.write_text(text)
    # A PROBED MACHINE, its record saying how a shell enters an environment
    # there -- the activation the generator reads (`configuration.md` § 4); a
    # wrapper is not rendered without one.  Its shape is STATED, as every
    # run's is (`architecture.md` § 5.2), and it carries the run settings
    # every job carries (`engines/template.md` § 5.4).
    from molbuilder.scheduler import Environment as _Env, Topology as _Topo
    (tmp_path / "environment.json").write_text(
        _Env(machine="this", scheduler=SLURM,
             topology=_Topo(sockets=2, cores_per_socket=32),
             env_init={"activation": "conda activate",
                                "preamble": "true"}).to_json()
        + "\n")
    return write_run_wrapper(deck, env="e", names=names,
                             resources=Resources(
                                 mpi_np=2 if engine == "siesta" else None,
                                 cpus_per_task=1, wall_slack=600),
                             resumes=True, relaunch_continues=True,
                             trial=False, emit_sbatch=False)


def _start(sh, folder, args=(), claim=None):
    """The run script started in ``folder`` with ``args``, no terminal --
    with the launch claim ``claim``, or none."""
    env = {k: v for k, v in os.environ.items() if k != "MB_LAUNCHED_BY"}
    if claim is not None:
        env["MB_LAUNCHED_BY"] = claim
    return subprocess.run(["bash", str(sh), *args], cwd=str(folder),
                          stdin=subprocess.DEVNULL, capture_output=True,
                          text=True, env=env)


# ---- the gate, in the emitted text and under execution ---------------- #


def test_a_bare_noninteractive_call_refuses_with_the_fix(tmp_path):
    """No claim + no terminal -> exit 2 before ANY work, naming both the
    right door and the deliberate override.  This is the `nohup`, cron and
    hand-`sbatch` case."""
    sh = _wrapper(tmp_path)
    cp = _start(sh, tmp_path)
    assert cp.returncode == 2
    assert "jobset launch" in cp.stderr
    assert "MB_LAUNCHED_BY=manual" in cp.stderr


# ---- one start per run ------------------------------------------------- #

def test_a_folder_that_holds_a_start_refuses_another(tmp_path):
    """A run folder holds one start, its session log saying it has one: a
    second start there refuses before anything is written, naming the
    launch that opens the next run (`project-layout.md` § 1.6.1).  THE
    SILENT FAILURE: the engine run again over the first start's output,
    its log and its conclusion overwritten."""
    sh = _wrapper(tmp_path)
    log = tmp_path / RunNames("JOB", "01_coarse").name(".runwrap.log")
    log.write_text("the first start's log\n")
    cp = _start(sh, tmp_path, claim="manual")
    assert cp.returncode == 2, cp.stderr
    assert "already holds a start" in cp.stderr
    assert "jobset launch" in cp.stderr
    assert log.read_text() == "the first start's log\n"


@pytest.mark.parametrize("engine,args,claim", [
    ("siesta", ["-h"], None),
    ("siesta", [], None),
    ("siesta", ["--dry-run"], "manual"),
    ("siesta", ["--bogus"], "manual"),
    ("siesta", ["-np", "--dry-run"], "manual"),
    ("pyscf", ["-h"], None),
    ("pyscf", ["--bogus"], "manual"),
    ("pyscf", ["-omp", "--dry-run"], "manual"),
], ids=["a-help-request", "a-start-the-gate-refuses", "a-dry-run",
        "an-argument-that-does-not-read", "a-count-flag-given-no-count",
        "pyscf-a-help-request", "pyscf-an-argument-that-does-not-read",
        "pyscf-a-count-flag-given-no-count"])
def test_what_starts_nothing_writes_no_session_log(engine, args, claim,
                                                   tmp_path):
    """The session log is the run's one start's (`job-contracts.md` § 2.6):
    a help request, a start the launch gate refuses, a dry run and a start
    whose arguments do not read start nothing, so none writes it, nor a
    conclusion -- and the folder stays free for the start it was opened
    for.  THE SILENT FAILURE: a mistyped flag on a start by hand, after
    which the corrected command is refused as a second start; or a count
    flag swallowing ``--dry-run`` and running the engine for real.  (The
    dry run stops at this test's activation, which names an env this
    machine lacks -- after the point the log would have opened.)"""
    sh = _wrapper(tmp_path, engine)
    _start(sh, tmp_path, args, claim)
    names = RunNames("JOB", "01_coarse")
    assert not (tmp_path / names.name(".runwrap.log")).exists()
    assert not (tmp_path / names.name(".concluded")).exists()


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
