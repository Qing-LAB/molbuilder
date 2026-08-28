"""The conclusion marker — did this attempt's process get to say goodbye.

`project-layout.md` § 1.6, *the other file* (user decision, 2026-08-28).
``run.json`` says a run STARTED; nothing said it was OVER, and *launched*
spans three states deserving opposite acts:

  * ran to its own end (converged OR errored — both conclusions),
  * still running (continuing would copy torn warm files),
  * force-stopped (walltime, kill — continuing is exactly what a person
    wants; the saved state is valid).

The wrapper writes ``<basename>-run<N>.concluded`` as its LAST act on the
main path — an engine error still reaches it, a kill never does — and the
launch lane asks the user over an unconcluded attempt rather than
deciding: *"the user has to do the judgment; we're not doing it over
them."*

Wrapper tests run the REAL rendered script with a stub engine, the way
`test_runwrap_retry.py` does — the marker's whole meaning lives in shell
control flow no unit test of Python can see.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

from molbuilder.jobset.materialize import (attempt_concluded,
                                           write_run_launch)
from molbuilder.jobset.model import Job, JobSet, Resources, WarmFile
from molbuilder.jobset.submit import SubmitError, submit_jobset
from molbuilder.runwrap import render_run_wrapper


# ------------------------------------------------------------ the wrapper

def _wrapper_dir(tmp_path, engine_body: str) -> Path:
    """A directory holding a rendered wrapper whose `siesta` is a stub."""
    d = tmp_path
    deck = d / "J_01_coarse.fdf"
    deck.write_text("SystemLabel J\nNumberOfAtoms 2\n")
    (d / "bin").mkdir()
    stub = d / "bin" / "siesta"
    stub.write_text("#!/bin/bash\n" + engine_body)
    stub.chmod(0o755)
    text = render_run_wrapper(deck, resources=Resources(mpi_np=1), env=None)
    (d / "J_01_coarse.run.sh").write_text(text)
    return d


def _run_wrapper(d: Path, *, background: bool = False):
    env = dict(os.environ,
               PATH=f"{d / 'bin'}:{os.environ['PATH']}",
               MB_MONITOR="0", MB_LAUNCHED_BY="manual")
    kw = dict(cwd=str(d), env=env,
              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if background:
        return subprocess.Popen(
            ["bash", "J_01_coarse.run.sh", "-np", "1"],
            start_new_session=True, **kw)
    return subprocess.run(["bash", "J_01_coarse.run.sh", "-np", "1"],
                          timeout=60, **kw)


def test_a_clean_end_writes_the_marker_with_rc0(tmp_path):
    d = _wrapper_dir(tmp_path, 'echo "Job completed"\nexit 0\n')
    _run_wrapper(d)
    mark = attempt_concluded(d, "J_01_coarse")
    assert mark is not None and mark.startswith("rc=0"), (
        f"a run that ended cleanly must conclude: {mark!r}")


def test_an_engine_ERROR_is_still_a_conclusion(tmp_path):
    """*"The calculation has been done — because of error or whatever —
    but the process is done."*  Nonzero exit reaches the wrapper's tail,
    so the marker is written WITH that code."""
    d = _wrapper_dir(tmp_path, 'echo "boom" >&2\nexit 7\n')
    _run_wrapper(d)
    mark = attempt_concluded(d, "J_01_coarse")
    assert mark is not None and mark.startswith("rc=7"), (
        f"an engine error is a conclusion and carries its code: {mark!r}")


def test_a_forced_stop_leaves_NO_marker(tmp_path):
    """The property the whole design rests on: a kill at any point stops
    the script before the main-line write — the cleanup trap runs on
    SIGTERM, which is exactly why the marker must not live there."""
    d = _wrapper_dir(tmp_path, 'sleep 30\n')
    proc = _run_wrapper(d, background=True)
    deadline = time.time() + 20
    while time.time() < deadline:                 # engine actually up
        if any(p.name.endswith("-run0.out") for p in d.iterdir()):
            break
        time.sleep(0.2)
    os.killpg(proc.pid, signal.SIGTERM)           # the walltime kill
    proc.wait(timeout=20)
    assert attempt_concluded(d, "J_01_coarse") is None, (
        "a force-stopped run wrote a conclusion marker -- absence is the "
        "only honest spelling of 'never got to say goodbye'")


# ----------------------------------------------------- the reader's rule

def test_the_question_is_asked_of_the_HIGHEST_out_index(tmp_path):
    """A warm-retry chain execs fresh wrappers; an old index's marker
    beside a newer unconcluded .out is a previous goodbye, not this
    one's."""
    (tmp_path / "J-run0.out").write_text("old")
    (tmp_path / "J-run0.concluded").write_text("rc=0 at earlier")
    (tmp_path / "J-run1.out").write_text("newer, killed")
    assert attempt_concluded(tmp_path, "J") is None
    (tmp_path / "J-run1.concluded").write_text("rc=0 at now")
    assert attempt_concluded(tmp_path, "J").startswith("rc=0")


def test_no_out_at_all_reads_unconcluded(tmp_path):
    assert attempt_concluded(tmp_path, "J") is None


# ------------------------------------------------------- the launch gate

def _launched_attempt(tmp_path, *, concluded: bool):
    js = JobSet(name="J", engine="siesta", kind="ladder", shared=[],
                jobs=[Job(name="coarse", script="J_01_coarse.fdf",
                          resources=Resources(mpi_np=1),
                          warm=[WarmFile(name="J.XV")])])
    d = tmp_path / "01_coarse" / "run-0"
    d.mkdir(parents=True)
    # the deck READS its warm set, so a continue has something to do
    deck = ("SystemLabel J\nMD.UseSaveXV .true.\nDM.UseSaveDM .true.\n")
    (tmp_path / "J_01_coarse.fdf").write_text(deck)
    (d / "J_01_coarse.fdf").write_text(deck)
    # a stub wrapper, so the judgement path can LAUNCH the continued
    # attempt for real (direct mode runs it; the gate is the subject,
    # not the engine)
    (tmp_path / "J_01_coarse.run.sh").write_text("#!/bin/bash\nexit 0\n")
    (d / "J_01_coarse-run0.out").write_text("output\n")
    (d / "J.XV").write_text("warm")
    write_run_launch(d, mode="direct", command=["bash", "x"])
    if concluded:
        (d / "J_01_coarse-run0.concluded").write_text("rc=0 at then\n")
    return js


def test_a_concluded_attempt_continues_and_says_so(tmp_path):
    js = _launched_attempt(tmp_path, concluded=True)
    res = submit_jobset(js, tmp_path, mode="direct", only="coarse",
                        dry_run=True)
    texts = " | ".join(r.status for r in res)
    assert "run-0" in texts, texts


def test_an_UNCONCLUDED_attempt_is_a_question_not_a_decision(tmp_path):
    """Still running and force-stopped look the same on disk; the refusal
    names both and hands the judgement over — never silently continues,
    never silently refuses forever."""
    js = _launched_attempt(tmp_path, concluded=False)
    with pytest.raises(SubmitError) as e:
        submit_jobset(js, tmp_path, mode="direct", only="coarse")
    msg = str(e.value)
    assert "never CONCLUDED" in msg
    assert "RUNNING" in msg and "force-stopped" in msg, (
        "the refusal must name BOTH states the files cannot separate")
    assert "--yes" in msg, "the refusal owes the way to record a judgement"


def test_the_recorded_judgement_continues_anyway(tmp_path):
    """`--yes` is the user's judgement, honoured — the framework said its
    piece and steps aside."""
    js = _launched_attempt(tmp_path, concluded=False)
    res = submit_jobset(js, tmp_path, mode="direct", only="coarse",
                        dry_run=False, continue_unconcluded=True)
    texts = " | ".join(r.status for r in res)
    assert "NOT concluded" in texts and "your judgement" in texts, texts
