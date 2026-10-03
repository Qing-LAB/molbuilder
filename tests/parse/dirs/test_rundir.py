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
3. **what speaks for the status and what a viewer opens are different
   questions** (§ 5.1). Collapsing them is the trap that section exists to
   mark, and nothing in the tree held it.

The other fields (`engine`, `status`) are pass-throughs of readers with their
own tests; re-asserting them here would be a test per field.
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
    # THE RECORD rides the same answer (`model/parse.md` § 5d): what ran,
    # judged by the same `run_status` call the status above is.
    assert got.record["verdict"]["state"] == got.status["state"], got.record


def test_a_prepped_stage_that_has_not_run_is_still_a_run_directory(tmp_path):
    """A deck and no output: `not_run`, not `not mine`.

    `can_parse` decides whether the registry claims a directory at all, so a
    predicate that wanted an OUTPUT would make every consumer asking about a
    ladder rung before it runs -- which is the ordinary case on the Results
    tab, where four of five transport rungs are typically pending -- raise
    `UnknownFormatError` rather than answer "not run yet".

    A PREPPED stage is stamped a run (`project-layout.md` § 1.4a), as prep
    stamps it, so it has a run state before it writes a byte: ``pending`` --
    prepped, never launched, as its missing launch record says (§ 1.6).  It
    answered "running -- no result file yet" until 2026-09-26, while the
    jobset layer answered the same directory "pending".  The same deck in a
    folder nobody described is read ALONE: claimed, listed, and given no run
    state, because nothing grounds one (§ 5.0; the route kept this rule until
    W35 P2 moved it into the door).
    """
    from molbuilder import calcdirs
    from support.junction import run_dir
    run = run_dir(tmp_path)                       # the .fdf, nothing else
    assert not list(pathlib.Path(run).glob("*.out"))

    alone = parse_dir(run)
    assert alone.status is None and alone.record is None, alone.status

    calcdirs.write(run, role=calcdirs.RUN, root=tmp_path)
    got = parse_dir(run)

    assert got.status["state"] == "pending", got.status
    assert got.status["detail"] == "prepped, not launched (no run.json)", (
        got.status)
    assert got.status["active_source"] is None, (
        "nothing here speaks for a run that has not run")

    # ...and the predicate still discriminates, or the claim above is free:
    # an empty directory is nobody's run.
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(UnknownFormatError):
        parse_dir(empty)


def test_openable_is_not_active(tmp_path):
    """§ 5.1's two questions, separated by the case that separates them.

    A PySCF run in progress: the concluded stdout of an earlier attempt sits
    beside the molwatch log the current attempt is writing right now.  (On
    SIESTA the two coincide -- its stdout IS its live channel, § 5.5 -- so
    the separation shows on the engine whose stdout no parser claims.)

    * `active` -- *whose run-state is the status* -- considers RESULT files
      only, so the unconcluded log does not vote and the stdout answers.
    * `openable` -- *what a viewer should load* -- prefers the live log,
      because that is the run somebody opened the tab to watch.

    Answering one with the other is wrong in both directions: a viewer sent
    to the stdout watches a finished run while the live one scrolls past, and
    a status taken from the log reports "running" for ever on a directory
    whose result is already on disk (`running-a-job.md` § 4: a log with no
    conclusion footer is not a result).
    """
    run = tmp_path / "junction-run"
    run.mkdir()
    # a PySCF run directory: the deck the label is read from, the earlier
    # attempt's concluded stdout, the log being written now
    (run / "junction_01_coarse.py").write_text('JOB = "junction"\n', encoding="utf-8")
    done = run / "junction_01_coarse-run0.pyscf.log"
    done.write_text("PySCF stdout\nJob complete in 12.3 s\n", encoding="utf-8")
    log = _live_molwatch(run, "junction_01_coarse.molwatch.log")

    got = parse_dir(run)

    active = got.status["active_source"]
    assert active == done.name, (
        f"an unconcluded log voted for the status: {active}")
    assert pathlib.Path(got.openable).name == log.name, (
        f"the viewer was sent to a finished result: {got.openable}")
    assert active != pathlib.Path(got.openable).name


# ---- § 5.5: what should a viewer open -- the CALCULATION decides -------- #


def _spectrum_dir(tmp_path, *, say_calculation=True, shape="flat"):
    """A finished vibration run: the sidecar, and a molwatch log that is a stub.

    The stub is the real shape, not a simplification -- measured on a CO2
    spectrum run driven through the UI 2026-09-18: header, ONE
    `kind: initial_preview` block, `# concluded:` footer.  A spectrum has no
    geometry sequence to log, so the progress channel every run seeds says
    nothing about this one.

    ``shape`` puts the run where that shape puts it, and `task.json` where
    `project-layout.md` § 1.0 puts it -- in the PARENT, always.  Flat's run
    files sit in the calculation root, so the two land in one directory;
    hierarchical's sit two levels down, so they do not.  Returns the RUN
    directory, which is what a viewer is handed either way.
    """
    import json
    from molbuilder import calcdirs
    run = tmp_path
    if shape == "hierarchical":
        run = tmp_path / "01_freq" / "run-0"
        run.mkdir(parents=True)
        # STAMPED, because that is what prep produces (`project-layout.md`
        # § 1.4a, invariant 6b).  A hand-made tree with no records is a
        # different case with its own test below -- it is read ALONE.
        calcdirs.write(run.parent, role=calcdirs.CONTAINER, root=tmp_path)
        calcdirs.write(run, role=calcdirs.RUN, root=tmp_path)
    (run / "co2spec.spectra.json").write_text(json.dumps({
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
    _mw_log(run, "co2spec_01_freq.molwatch.log", concluded=True)
    if say_calculation:
        (tmp_path / "task.json").write_text(json.dumps({
            "schema": "molbuilder/task@1", "engine": {"name": "pyscf"},
            "shape": shape, "calculation": "vibration",
            "run": {"name": "co2spec", "id": "co2spec_CO2"},
            "structure": {"source": "co2spec.source.xyz",
                          "formula": "CO2", "atoms": 3},
            "stages": [{"name": "freq", "enabled": True, "overrides": {}}],
        }), encoding="utf-8")
    return run


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


@pytest.mark.parametrize("shape", ["flat", "hierarchical"])
def test_a_spectrum_run_opens_its_spectrum_not_its_molwatch_stub(
        tmp_path, shape):
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

    **BOTH SHAPES, because the description does not live with the run.**
    `project-layout.md` § 1.0 puts `task.json` in the PARENT -- *"only
    rendered files and copies go down to where the engine runs"* -- so in
    the hierarchical shape it is two levels above the run directory a viewer
    is handed.  The 2026-09-18 fix asked only the handed directory and was
    written against a flat fixture, so it passed while every hierarchical
    spectrum went on opening its stub; measured on a real Raman run
    2026-09-19 (`spectrum/bridge-hier/01_raman/run-0`), whose trail read
    `calculation: (not stated in task.json)`.  One rule, both shapes, or the
    rule is only true where the fixture happened to look.

    MUTATION THIS MUST FAIL AGAINST: put `.molwatch.log` first, stop asking
    `task.json` what calculation this is, or ask only the handed directory.
    """
    d = _spectrum_dir(tmp_path, shape=shape)
    got, attempts = openable_in(str(d))
    assert got is not None, attempts
    assert pathlib.Path(got).name == "co2spec.spectra.json", (
        f"got {pathlib.Path(got).name!r} -- the trail was:\n  "
        + "\n  ".join(attempts))


def test_an_unmarked_directory_is_read_alone(tmp_path):
    """No record ⇒ the directory answers for itself and claims nothing above.

    `project-layout.md` § 1.4a: *absence narrows the answer; it does not
    refuse the directory*.  A tree written before that rule, or an attempt
    copied out of its calculation, still reads — its files, which one to open
    — but it does not get to say which calculation it belongs to, because
    nothing here knows.

    The scenario is the one that matters: a `task.json` DOES sit above this
    directory, and the directory must not adopt it.  Proximity is not
    membership; the record is.  Adopting it would let a description anywhere
    up the tree decide what an unrelated folder opens, which is exactly what
    the walk this replaced could do.

    MUTATION THIS MUST FAIL AGAINST: search upward for a `task.json` instead
    of reading `calcdir.json`.  Then this directory adopts `vibration` and
    opens the spectrum.
    """
    loose = tmp_path / "01_freq" / "run-0"        # the shape, none of the record
    loose.mkdir(parents=True)
    _spectrum_dir(loose)                          # its own task.json, then:
    (loose / "task.json").unlink()                # ...say it only at the TOP
    _spectrum_dir(tmp_path, say_calculation=True)

    got, attempts = openable_in(str(loose))

    assert got is not None and pathlib.Path(got).name.endswith(
        ".molwatch.log"), (
        f"adopted {tmp_path}'s task.json without a record and opened "
        f"{pathlib.Path(got).name if got else None!r}; the trail was:\n  "
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


def test_a_python_file_that_names_no_job_is_nobodys_deck(
        isolated_projects_root):
    """`.py` is the PySCF deck's suffix, and a generic one: a person's own
    script beside a run is not a deck -- nor were the monitor's modules,
    fourteen `.py` files in every attempt until 2026-09-26.  Read by stem,
    each became a label: `job.py`, `runfiles.py`, ... runs of their own in
    the Results listing, and so many decks that a prepped PySCF attempt had
    no record at all.  A `.py` that names no `JOB` is nobody's deck.  Through
    the road: a PySCF relaxation described and prepped by `jobset` (no
    engine runs), the monitor's one file beside its deck, and a person's
    script beside both.

    MUTATION THIS MUST FAIL AGAINST: take a `.py` file's stem as a label."""
    from click.testing import CliRunner
    from molbuilder import describe as D
    from molbuilder.config.pyscf import PySCFConfig
    from molbuilder.jobset._cli import jobset_group
    from molbuilder.parse.dirs.rundir import labels_in
    from molbuilder.pyscf.stages import default_pyscf_stages
    from molbuilder.scheduler import Environment, Topology
    from molbuilder.structure import Structure
    from molbuilder.workingcopy_structure import StructureCodec

    root = isolated_projects_root
    struct = Structure.from_xyz("2\nh2\nH 0 0 0\nH 0 0 0.74\n")
    StructureCodec().write(struct, root / "in.xyz")
    dest = root / "t" / "calc"
    cfg, stages = PySCFConfig(job_name="H2"), default_pyscf_stages("publishable")
    D.write_description(
        D.build_description(struct, cfg, stages, engine="pyscf",
                            shape="hierarchical", name="H2",
                            source=str(root / "in.xyz")),
        dest, struct=struct)
    # The calculation's record: the machine it is prepped for, with how a
    # shell enters an environment there -- the activation the generator
    # reads (`configuration.md` § 4).
    (dest / "environment.json").write_text(
        Environment(scheduler="workstation",
                    topology=Topology(sockets=1, cores_per_socket=4),
                    env_init={"activation": "conda activate",
                                       "preamble": "true"}
                    ).to_json() + "\n")
    # ...and the run's threads, stated as every run's are
    # (`architecture.md` § 5.2).
    r = CliRunner().invoke(jobset_group, ["prep", "run", stages[0].name,
                                          "--bundle", str(dest),
                                          "--no-sbatch",
                                          "--cpus-per-task", "1"])
    assert r.exit_code == 0, r.output
    attempt = next(dest.glob("*_*/run-0"))
    token = attempt.parent.name
    assert (attempt / "mb_monitor.pyz").is_file()
    (attempt / "plot_energies.py").write_text(
        "import json\nprint(json.load(open('energies.json')))\n")
    assert labels_in(str(attempt)) == ["H2", f"H2_{token}"]
    got = parse_dir(str(attempt))
    assert got.record is not None, "a prepped PySCF attempt has a record"
    assert got.record["deck"]["path"] == f"H2_{token}.py", got.record["deck"]
    # ...and it has not been launched, which its launch record says
    assert got.status["state"] == "pending", got.status
