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
    # Keyed by the ROLE since 2026-09-18 -- `_enumerate_files`' run-output
    # buckets were `"out"` / `"molwatch"`, a private two-word nickname for a
    # three-row catalogue column whose third row therefore had no bucket.
    assert "hemeC-stage2-run3-finished-42fr.out" in [
        pathlib.Path(p).name for p in got.files[".out"]]


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


# ---- § 5.5: what should a viewer open -- the CALCULATION decides -------- #


def _spectrum_dir(tmp_path, *, say_calculation=True):
    """A finished vibration run: the sidecar, and a molwatch log that is a stub.

    The stub is the real shape, not a simplification -- measured on a CO2
    spectrum run driven through the UI 2026-09-18: header, ONE
    `kind: initial_preview` block, `# concluded:` footer.  A spectrum has no
    geometry sequence to log, so the progress channel every run seeds says
    nothing about this one.
    """
    import json
    (tmp_path / "co2spec.spectra.json").write_text(json.dumps({
        "schema_version": 1, "engine": "pyscf", "engine_version": "x",
        "molbuilder_version": "x", "timestamp": "2026-09-18T00:00:00Z",
        "structure_hash": "sha256:0", "n_atoms_total": 3,
        "free_atom_idxs": [0, 1, 2], "frozen_atom_idxs": [],
        "equilibrium_scf_eh": -188.4, "equilibrium_mo_energies_eh": [-1.0],
        "equilibrium_homo_idx": 0, "modes": [], "selected_mode_idxs_1based": [],
        "config": {}, "methods_text": "", "bibliography_keys": [],
        "phase_frequencies": "complete", "phase_raman": "complete",
        "phase_es": "complete", "phase_relaxation": "complete",
    }), encoding="utf-8")
    _mw_log(tmp_path, "co2spec_01_freq.molwatch.log", concluded=True)
    if say_calculation:
        (tmp_path / "task.json").write_text(json.dumps({
            "schema": "molbuilder/task@1", "engine": {"name": "pyscf"},
            "shape": "flat", "calculation": "vibration",
            "run": {"name": "co2spec", "id": "co2spec_CO2"},
            "structure": {"source": "co2spec.source.xyz",
                          "formula": "CO2", "atoms": 3},
            "stages": [{"name": "freq", "enabled": True, "overrides": {}}],
        }), encoding="utf-8")
    return tmp_path


def _mw_log(dirpath, name, *, concluded):
    body = ("# molwatch trajectory log v1\n# engine: pyscf\n# job: co2spec\n"
            "# units: energy=eV, force=eV/Ang, coords=Ang\n\n"
            "==== molwatch step 0 begin ====\nstep_index: 0\n"
            "kind: initial_preview\nn_atoms: 1\ncoordinates (Ang):\n"
            "   H       0.0 0.0 0.0\n==== molwatch step 0 end ====\n")
    if concluded:
        body += "\n# concluded: 2026-09-18T00:00:00\n"
    p = pathlib.Path(dirpath) / name
    p.write_text(body, encoding="utf-8")
    return p


def test_a_spectrum_run_opens_its_spectrum_not_its_molwatch_stub(tmp_path):
    """THE CALCULATION DECIDES, and there is no preference order to tune.

    A vibration run is FOR its `.spectra.json`: the deck rewrites it
    atomically at every phase boundary and it carries its own `phase_*`
    flags, so it is the live view DURING the run and the result after it.
    The molwatch log every run seeds is, for this kind, one preview block.

    Until 2026-09-18 the chain's first rung was *any molwatch log, newest
    wins*, which fired before anything else -- so every spectrum run's
    viewer got the stub.  That is an OPTIMIZATION-shaped rule generalised to
    every kind, which is why the fix deletes the ladder rather than
    reordering it.

    MUTATION THIS MUST FAIL AGAINST: put `.molwatch.log` first, or stop
    asking `task.json` what calculation this is.
    """
    d = _spectrum_dir(tmp_path)
    got, attempts = openable_in(str(d))
    assert got is not None, attempts
    assert pathlib.Path(got).name == "co2spec.spectra.json", (
        f"got {pathlib.Path(got).name!r} -- the trail was:\n  "
        + "\n  ".join(attempts))


def test_an_optimization_still_opens_its_trajectory(tmp_path):
    """The other half of the same rule: a run whose calculation names no
    product of its own opens the progress channel, which for an optimization
    is the trajectory that grows per step.  `task.json` omits `calculation`
    for the default kind, so this is also the not-stated path."""
    _mw_log(tmp_path, "co2flat_01_coarse.molwatch.log", concluded=True)
    got, attempts = openable_in(str(tmp_path))
    assert got is not None and pathlib.Path(got).name.endswith(".molwatch.log"), (
        attempts)


def test_the_door_never_offers_a_file_the_registry_refuses(tmp_path):
    """*What is a run's output* and *what can a person open* are different
    questions with different owners (§ 5.5), and this is the second one.

    Measured 2026-09-18 on a finished CO2 spectrum run with its molwatch log
    removed -- the exact shape of `projects/BDT/spectrum/BDT-only`: the chain
    returned `<job>_<stage>.log`, PySCF's own verbose logger, and the
    caller's very next step was `detect()`, which refused it.  A directory
    holding nothing openable must answer None, so the refusal names the
    directory instead of a file that cannot be read.

    MUTATION THIS MUST FAIL AGAINST: drop the `_claimed` filter.
    """
    (tmp_path / "co2spec_01_freq.log").write_text("PySCF verbose log\n" * 20,
                                                  encoding="utf-8")
    (tmp_path / "co2spec_01_freq-run0.pyscf.log").write_text("Job complete in 1.0 s\n",
                                                             encoding="utf-8")
    (tmp_path / "co2spec_01_freq.py").write_text('JOB = "co2spec"\n',
                                                 encoding="utf-8")
    got, attempts = openable_in(str(tmp_path))
    assert got is None, (
        f"offered {pathlib.Path(got).name!r}, which no parser claims:\n  "
        + "\n  ".join(attempts))
