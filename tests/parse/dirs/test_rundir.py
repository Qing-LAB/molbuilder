"""`JobDirParser` — the door a run directory is asked through.

`model/parse.md` § 5.0 / § 5.1; `plans/plan.md` § 5c, step 1.

**What these three hold, and why there are three.** The chain inside
`openable_in` is already held — three tests in `test_path_framework_doors.py`
cover rung 1, rung 4 and the dotted-filename fall-through, and they move onto
this function when step 3 deletes the copy in `web/blueprints/watch.py`. What
those cannot hold is what step 1 ADDS:

1. **the registry answers.** Until 2026-09-18 no DirParser was registered, so
   `parse_dir` could only raise -- which is why six functions across three
   modules are called by name and the seventh consumer (the Results file
   picker) guesses from filenames in the browser instead of asking.
2. **a directory that has not run yet is still a run directory.** A prepped
   stage is `not_run`, not `not mine`; refusing it here would make every
   consumer that asks about a ladder rung before it runs raise instead.
3. **`active` and `openable` are different questions** (§ 5.1). Collapsing
   them is the trap that section exists to mark, and nothing in the tree
   held it.

The other fields (`engine`, `files`, `status`) are pass-throughs of readers
with their own tests; re-asserting them here would be a test per field.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import pytest

from molbuilder.parse import parse_dir
from molbuilder.parse.dirs import openable_in
from molbuilder.parse.errors import UnknownFormatError
from molbuilder.parse.types import RunDirResult


def _live_molwatch(run, name: str) -> pathlib.Path:
    """A molwatch log with NO conclusion footer — a run in progress.

    Unconcluded on purpose: that is the whole separation being tested.  A
    concluded log is a RESULT and votes for `active`; this one is a live view
    and must not.
    """
    p = pathlib.Path(run) / name
    p.write_text(
        "# molwatch trajectory log v1\n"
        "# engine: siesta\n"
        "# job: junction\n"
        "# units: energy=eV, force=eV/Ang, coords=Ang\n"
        "\n"
        "==== molwatch step 0 begin ====\n"
        "step_index: 0\n"
        "kind: scf\n"
        "n_atoms: 1\n"
        "coordinates (Ang):\n"
        "   H       0.0 0.0 0.0\n"
        "==== molwatch step 0 end ====\n",
        encoding="utf-8")
    return p


def test_the_registry_answers_for_a_run_directory(tmp_path):
    """`parse_dir(<a run directory>)` returns the composed answer.

    This is step 1's whole point.  `model/parse.md` § 5's banner said
    `parse_dir` and `detect` "can only raise" on a directory, for want of a
    registered DirParser -- so every consumer reached past the registry and
    called `run_status`, `_enumerate_files`, `engine_of` and the web layer's
    private resolver by name, and the one consumer that CANNOT import Python
    guessed from filenames instead.
    """
    from support.junction import job_run_dir
    run = job_run_dir(tmp_path)

    got = parse_dir(run)

    assert isinstance(got, RunDirResult), (
        "the registry dispatched somewhere else -- there is one DirParser")
    assert got.parser_name == "jobdir"
    assert got.engine == "siesta"
    assert got.status["state"] == "finished", got.status
    assert "hemeC-stage2-run3-finished-42fr.out" in [
        pathlib.Path(p).name for p in got.files["out"]]


def test_a_prepped_stage_that_has_not_run_is_still_a_run_directory(tmp_path):
    """A deck and no output: `not_run`, not `not mine`.

    `can_parse` decides whether the registry claims a directory at all, so a
    predicate that wanted an OUTPUT would make every consumer asking about a
    ladder rung before it runs -- which is the ordinary case on the Results
    tab, where four of five transport rungs are typically pending -- raise
    `UnknownFormatError` rather than answer "not run yet".
    """
    from support.junction import run_dir
    run = run_dir(tmp_path)                       # the .fdf, nothing else
    assert not list(pathlib.Path(run).glob("*.out"))

    got = parse_dir(run)

    assert got.status["state"] == "running", got.status
    assert got.status["detail"] == "no result file yet", got.status
    assert got.active is None, "nothing here speaks for a run that has not run"

    # ...and the predicate still discriminates, or the claim above is free:
    # an empty directory is nobody's run.
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(UnknownFormatError):
        parse_dir(empty)


def test_openable_is_not_active(tmp_path):
    """§ 5.1's two questions, separated by the case that separates them.

    A run in progress: the finished `.out` of an earlier attempt sits beside
    the molwatch log being written right now.

    * `active` -- *whose run-state is the status* -- considers RESULT files
      only, so the unconcluded log does not vote and the `.out` answers.
    * `openable` -- *what a viewer should load* -- prefers the live log,
      because that is the run somebody opened the tab to watch.

    Answering one with the other is wrong in both directions: a viewer sent
    to the `.out` watches a finished run while the live one scrolls past, and
    a status taken from the log reports "running" for ever on a directory
    whose result is already on disk (`running-a-job.md` § 4: a log with no
    conclusion footer is not a result).
    """
    from support.junction import job_run_dir
    run = job_run_dir(tmp_path)
    log = _live_molwatch(run, "junction_01_coarse.molwatch.log")

    got = parse_dir(run)

    assert pathlib.Path(got.active).name.endswith(".out"), (
        f"an unconcluded log voted for the status: {got.active}")
    assert pathlib.Path(got.openable).name == log.name, (
        f"the viewer was sent to a finished result: {got.openable}")
    assert got.active != got.openable

    # The same two answers through the function, which is what step 3's
    # callers will hold onto once the web layer's private copy is deleted.
    chosen, attempts = openable_in(str(run))
    assert pathlib.Path(chosen).name == log.name
    assert any("molwatch" in a for a in attempts), attempts
