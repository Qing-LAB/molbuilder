"""The readers a catalogue row names -- `job-contracts.md` § 2.2, the
``reads`` column of `runfiles.WRITTEN`.

A run output's row names the code that reads it, by question (how the run
ended, where its engine stops, its live pass, ...): `runfiles.reader` is the
one door every reader asks, on the machine and beside a job, where the
monitor's bundle carries each reader a row names (`runwrap.monitor_companions`).
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys

import pytest

from molbuilder import runfiles
from molbuilder.runfiles import (READ_BESIDE_A_JOB, ROLE_SIESTA_OUT,
                                 RunFileError, WRITTEN)

#: Every (role, question) a row names a reader for.
ASKED = [(a.role or a.name, q) for a in WRITTEN for q, _ref in a.reads]


@pytest.mark.parametrize("role,question", ASKED,
                         ids=[f"{r} {q}" for r, q in ASKED])
def test_every_reader_a_row_names_resolves(role, question):
    """THE SILENT FAILURE: a reader renamed or moved while its row still
    names it -- found only when a job asks how its run ended, and then as
    an ending that cannot be read."""
    assert runfiles.reader(role, question) is not None


_OUT = next(a for a in WRITTEN if a.role == ROLE_SIESTA_OUT)
_NOT_AN_OUTPUT = next(a for a in WRITTEN if not a.output and a.reads)
_A_FIXED_NAME = next(a for a in WRITTEN if a.name and not a.role)
_STOPS = dict(_OUT.reads)["stops"]
_ENDING = dict(_OUT.reads)["ending"]


@pytest.mark.parametrize("row,said", [
    (dataclasses.replace(_OUT, reads=_OUT.reads + (
        ("weather", _STOPS),)), "names no question of READ_QUESTIONS"),
    (dataclasses.replace(_OUT, reads=_OUT.reads + (
        ("stops", _STOPS),)), "names a question twice"),
    (dataclasses.replace(_OUT, reads=tuple(
        (q, ref.split(":")[-1]) for q, ref in _OUT.reads)),
     "names readers that are no `molbuilder.<module>:NAME`"),
    (dataclasses.replace(_OUT, reads=tuple(
        (q, ref) for q, ref in _OUT.reads if q != "ending")),
     "is a run output and names no `ending` reader"),
    (dataclasses.replace(_NOT_AN_OUTPUT, reads=_NOT_AN_OUTPUT.reads + (
        ("stops", _STOPS),)), "is no run output but names ['stops']"),
    (dataclasses.replace(_NOT_AN_OUTPUT, reads=_NOT_AN_OUTPUT.reads + (
        ("ending", _ENDING),)), "is no run output but names ['ending']"),
    (dataclasses.replace(_NOT_AN_OUTPUT, stderr_apart=True),
     "is no run output but names ['stderr_apart']"),
    (dataclasses.replace(_A_FIXED_NAME, reads=(("contract", _ENDING),)),
     "names readers but no role to ask them by"),
], ids=["a-question-no-reader-answers", "a-question-twice",
        "a-reader-that-is-no-part", "a-run-output-with-no-ending",
        "stops-named-by-no-run-output", "an-ending-named-by-no-run-output",
        "stderr-apart-on-no-run-output", "readers-on-a-row-with-no-role"])
def test_a_row_naming_its_readers_wrong_is_refused_at_load(row, said):
    """The check the catalogue runs when it loads (`runfiles._check_reads`),
    so the gap stops the first thing that imports it.  THE SILENT FAILURE:
    a run output with no ``ending`` reader, read by no one -- a job whose
    end is never judged."""
    with pytest.raises(RunFileError) as exc:
        runfiles._check_reads((row,))
    assert said in str(exc.value), str(exc.value)


@pytest.mark.parametrize("left_out,added", [
    ("out", None),
    (None, (".molwatch.log",)),
], ids=["a-file-whose-row-names-a-record-reader", "a-file-whose-row-names-none"])
def test_the_run_record_reads_the_files_the_rows_name(left_out, added):
    """The run record's table names the fields each file answers; the
    file's catalogue row names its ``record`` reader -- one set of files,
    checked as the record loads (`parse.dirs.record`).  THE SILENT FAILURE:
    a new engine output whose row names a reader the record never asks, or
    a table row reading a file whose row names none -- its fields gone from
    every record, and nothing said."""
    from molbuilder.parse.dirs import record as R
    rows = tuple(r for r in R.CONTRIBUTORS if r.name != left_out)
    if added:
        rows += (R.Contributor("added", ("computation.added",), added[0]),)
    with pytest.raises(ValueError, match="one set of files"):
        R._a_row_per_record_reader(rows)


class _Mol:
    """What the progress log's writer asks of a molecule: two atoms."""
    natm = 2

    def atom_coords(self, unit="Ang"):
        import numpy as np
        return np.zeros((2, 3))

    def atom_symbol(self, i):
        return "O"


def test_the_monitor_reads_the_files_whose_rows_name_a_live_pass(tmp_path):
    """The monitor asks each file whose catalogue row names a ``live``
    reading pass (`runfiles.read_roles`), and no other file in the run's
    folder -- a name there may be no file of ours: an editor's backup of
    the output beside the run's progress log, written by the deck's own
    writer (`trajectory_log.emitter`).  THE SILENT FAILURE: the backup's
    role asked of the catalogue, refused, and the monitor gone -- no
    samples, no report, for the rest of the job."""
    from molbuilder.monitor import WatchedRun
    from molbuilder.runfiles import ROLE_MOLWATCH, ROLE_SIESTA_OUT, RunNames
    from molbuilder.trajectory_log.emitter import MolwatchEmitter
    names = RunNames("job", "01_relax")
    MolwatchEmitter(str(tmp_path / names.name(ROLE_MOLWATCH)), "job", _Mol(),
                    runtime_info={"mpi_np": 2})
    (tmp_path / (names.name(ROLE_SIESTA_OUT) + "~")).write_text("")
    st = WatchedRun(names=names, directory=tmp_path).read(0.0, 1.0)
    assert st.elapsed_s == 1.0


def test_an_output_whose_name_says_no_role_says_no_stops():
    """A stopped run's output named outside the catalogue -- a hand-started
    run's ``run.output``, which the SIESTA parser reads by its content --
    says no stop at the job's wall and keeps no advice, as an output whose
    row names no ``stops`` reader (`_run_ending`).  THE SILENT FAILURE:
    the Results tab failing to load the file, its stops asked of a role
    the catalogue refuses."""
    from molbuilder.parse.engines._run_ending import (advice_of, cause_words,
                                                      wall_causes)
    from molbuilder.runfiles import role_of
    role = role_of("run.output")
    assert (wall_causes(role), advice_of(role), cause_words(role)) == (
        (), {}, {})


def test_the_readers_a_job_asks_beside_it_load_from_its_bundle_alone(
        tmp_path):
    """Beside a job, molbuilder is not installed: each reader a row names
    for a question the job asks there (`READ_BESIDE_A_JOB`) comes out of
    `mb_monitor.pyz`, gathered by the rows (`runwrap.monitor_companions`).
    MUTATION THIS MUST FAIL AGAINST: the bundle's members gathered from a
    hand list that leaves out a reader a row names (the trajectory log's
    live pass, `molwatch_reader`) -- every SIESTA run's monitor then stops
    on its first read."""
    from molbuilder.runfiles import MONITOR_BUNDLE
    from molbuilder.runwrap import monitor_bundle
    (tmp_path / MONITOR_BUNDLE).write_bytes(monitor_bundle())
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    # The condition has to be real, or this proves nothing.
    control = subprocess.run([sys.executable, "-c", "import molbuilder"],
                             cwd=tmp_path, env=env, capture_output=True,
                             text=True, timeout=120)
    assert control.returncode != 0, (
        "molbuilder is importable from the job's folder, so this test cannot "
        "reproduce a compute node.  Is it pip-installed?")
    asked = sorted(f"{a.role} {q}" for a in WRITTEN for q, _ref in a.reads
                   if q in READ_BESIDE_A_JOB)
    script = "\n".join([
        "import json, os, sys",
        f"sys.path.insert(0, {MONITOR_BUNDLE!r})",
        "import runfiles",
        "got = {}",
        "for a in runfiles.WRITTEN:",
        "    for q, ref in a.reads:",
        "        if q in runfiles.READ_BESIDE_A_JOB:",
        "            runfiles.reader(a.role, q)",
        "            mod = sys.modules[ref.split(':')[0].rsplit('.', 1)[-1]]",
        "            got[f'{a.role} {q}'] = os.path.basename(",
        "                os.path.dirname(mod.__file__))",
        "print(json.dumps(got))",
    ])
    done = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                          env=env, capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 0, (
        f"a reader a row names does not load from {MONITOR_BUNDLE} without "
        f"molbuilder:\n{done.stdout}{done.stderr}")
    got = json.loads(done.stdout.strip().splitlines()[-1])
    assert sorted(got) == asked
    assert set(got.values()) == {MONITOR_BUNDLE}, got
