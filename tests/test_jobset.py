"""The ``jobset`` framework: model + persistence + validate + materialize
+ plan (docs/execution/job-system.md), and the SIESTA
stage-ladder producer."""

from __future__ import annotations

import os
import re
import time as _time
from pathlib import Path

import pytest

from molbuilder.jobset.model import Job, JobSet, Resources, SCHEMA, WarmFile


@pytest.fixture(autouse=True)
def _sandbox(tmp_path_factory, monkeypatch):
    """cwd isolation for EVERY test here (I6, 2026-08-13): the prep/submit
    tests read this machine's config and record, which conftest isolates;
    this keeps their working directory out of the checkout.  (It moved HOME
    too, and named a cwd molbuilder.json and ~/.molbuilder, all retired.)
    Tests that need their own cwd (monkeypatch.chdir) still win: their
    monkeypatching applies after this fixture's."""
    box = tmp_path_factory.mktemp("sandbox")
    monkeypatch.chdir(box)
from molbuilder.jobset.materialize import job_dir_name, materialize
from molbuilder.jobset.plan import render_plan
from molbuilder.jobset import submit as _submit
from molbuilder.jobset.submit import submit_jobset, SubmitError
from molbuilder.jobset.prep import prep_jobset
from molbuilder.jobset.runstatus import jobset_status, render_status


@pytest.fixture(autouse=True)
def _tmp_is_the_projects_tree(tmp_path, monkeypatch):
    """These tests build a calculation under ``tmp_path`` and hand its path
    to a verb.  ``--bundle`` is fenced to the projects tree
    (`job-contracts.md` § 2.5b), so the test says where its tree IS rather
    than handing over a path from outside one -- which is exactly what a
    user does when their calculations live on scratch: set
    ``paths.projects`` / ``$MOLBUILDER_PROJECTS`` and the fence follows.
    """
    from molbuilder.projects import PROJECTS_ROOT_ENV
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tmp_path))


class _CP:
    """Minimal stand-in for subprocess.CompletedProcess."""
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


# --------------------------------------------------------------------- #
#  model + persistence (job-set@1)                                       #
# --------------------------------------------------------------------- #

def _ladder() -> JobSet:
    return JobSet(
        name="demo", engine="siesta", kind="ladder",
        shared=["C.psml", "mb_monitor.py"],
        jobs=[
            Job(name="s1", script="demo_s1.fdf",
                resources=Resources(domain="htc", time="0-04:00:00",
                                    mem="8G")),
            Job(name="s2", script="demo_s2.fdf",
                resources=Resources(domain="public", exclusive=True,
                                    time="0-04:00:00", mem="8G"),
                # WHAT it would take from a run it is continued from -- never
                # WHICH job -- that is named by a person at prep, with
                # `--from` (project-layout.md 1.6).
                warm=[WarmFile("demo.XV"), WarmFile("demo.DM")]),
        ],
    )


def test_jobset_roundtrips_through_job_set_at_1():
    """A `JobSet` survives `to_dict` -> `from_dict` losslessly, nested records
    included.

    `job-set.json` is the file every later verb reads: `launch` and `status`
    work from the loaded object for what was prepped, never from the
    description that built it (`job-system.md` § 3.1). A field the codec drops is a resource
    request or a warm-file declaration that vanishes between `prep` and `launch`,
    with nothing to compare against. The two spot checks -- `warm[0].name` and
    `resources.exclusive` -- are the nested levels a shallow dict copy flattens.
    """
    js = _ladder()
    d = js.to_dict()
    assert d["schema"] == SCHEMA
    back = JobSet.from_dict(d)
    assert back.to_dict() == d           # lossless
    assert back.jobs[1].warm[0].name == "demo.XV"
    assert back.jobs[1].resources.exclusive is True


def test_from_dict_rejects_unknown_schema():
    """A `job-set.json` written under a different schema is REFUSED, not read
    best-effort.

    This file outlives the code that wrote it: it sits in a bundle on scratch for
    weeks. Reading a foreign schema field by field would apply whatever happens to
    match and drop the rest, which for this file means launching with the wrong
    resources on a machine that charges for them. `job-system.md` § 3.
    """
    d = _ladder().to_dict()
    d["schema"] = "job-set@99"
    with pytest.raises(ValueError, match="schema mismatch"):
        JobSet.from_dict(d)


def test_validate_passes_clean_ladder():
    """The ordinary ladder validates clean -- no findings.

    The negative control for the whole `validate` group: every other test here
    asserts that a specific defect IS reported, and a validator that reported
    something about every job-set would satisfy all of them while making
    `materialize` and `submit` -- both of which refuse on a non-empty result --
    reject every real bundle.
    """
    assert _ladder().validate() == []


def test_validate_catches_duplicate_names():
    """Two jobs sharing one name are reported.

    The job name is the identifier everything else resolves through: the
    directory (`bench-<name>`), the `-J` queue name, `--only`, and the status row
    (`job-contracts.md` § 6.3). Two jobs sharing it means one directory
    holding two jobs' files and a `--only` that silently picks one of them.
    """
    js = _ladder()
    js.jobs[1].name = "s1"
    assert any("duplicate" in e for e in js.validate())


def test_a_job_declares_exactly_these_eight_things():
    """A `Job` is a name, a script, resources, what it would warm-start from,
    the traits a warm-start condition is compared against -- and, for a
    TRIAL, its sweep coordinate as data (``point``, 2026-08-21,
    `generator.md` § 4.3a); for a job whose engine leaves no result, the
    bundle that finishes it (``finish``, 2026-09-28, `engines/vibration.md`
    § 5.5: a SIESTA force-constant run's modes); and whether a re-run of it
    continues from what the last one left (``resumes``, 2026-09-29,
    `job-contracts.md` § 4.2a: false for a force-constant run).

    Asserted as an EQUALITY on the field set, in both the dataclass and the
    wire form.  An equality is the whole rule in one line: any field added
    without a decision fails, including one that would let a job name another
    job -- and the test never has to spell out what such a field would be
    called, so a retired vocabulary stays out of the suite.  On the wire a
    RUNG carries no ``point`` and a job its engine finishes no ``finish``:
    absent, never empty (the same absent-vs-null reading as
    ``requires_same``).
    """
    import dataclasses
    assert {f.name for f in dataclasses.fields(Job)} == {
        "name", "script", "resources", "warm", "traits", "point", "finish",
        "resumes"}
    assert set(_ladder().to_dict()["jobs"][1]) == {
        "name", "script", "resources", "warm", "traits"}
    trial = Job("G1K4C6", "t.fdf", point={"G": 1, "K": 4, "C": 6})
    assert trial.to_dict()["point"] == {"G": 1, "K": 4, "C": 6}
    assert Job.from_dict(trial.to_dict()).point == {"G": 1, "K": 4, "C": 6}
    finished = Job("freq", "x.fdf", finish="mb_vibration.pyz")
    assert finished.to_dict()["finish"] == "mb_vibration.pyz"
    assert Job.from_dict(finished.to_dict()).finish == "mb_vibration.pyz"
    # a job that resumes says nothing; one that does not says false
    restarts = Job("freq", "x.fdf", resumes=False)
    assert restarts.to_dict()["resumes"] is False
    assert Job.from_dict(restarts.to_dict()).resumes is False
    assert "resumes" not in finished.to_dict()





def test_validate_catches_empty_and_bad_kind():
    """A job-set with no jobs, and one whose `kind` is not a known kind, are both
    reported.

    `kind` selects real behaviour downstream -- a ladder runs one stage at a time
    while a sweep's points are independent (`job-system.md` § 3) -- so an
    unknown value cannot be defaulted; there is nothing to default TO. And an
    empty job-set would prep and submit successfully while doing nothing, which
    reads to the person as a finished launch.
    """
    assert any("empty" in e for e in
               JobSet("n", "siesta", "ladder", jobs=[]).validate())
    assert any("kind" in e for e in
               JobSet("n", "siesta", "bogus",
                      jobs=[Job("a", "a.fdf")]).validate())


# --------------------------------------------------------------------- #
#  materialize engine                                                    #
# --------------------------------------------------------------------- #

def test_materialize_creates_dirs_and_copies(tmp_path):
    """L2 (roadmap 7.10, user 2026-08-24): a run directory holds REAL
    files.  This asserted relative symlinks until the layout repair --
    links up to root copies are the mechanism that put 50 rendered files
    at a ten-trial bundle's root."""
    js = _ladder()
    for f in js.shared + [j.script for j in js.jobs]:
        (tmp_path / f).write_text("x")
    dirs = materialize(js, tmp_path)
    assert [d.name for d in dirs] == ["bench-s1", "bench-s2"]
    got = tmp_path / "bench-s1" / "C.psml"
    assert got.is_file() and not got.is_symlink()
    assert got.read_text() == "x"


def test_materialize_lays_no_link_into_another_job(tmp_path):
    """The inverse of what this test used to assert, and that is the change.

    It read: the carry symlink exists, points at `../bench-s1/demo.XV`, and
    **dangles** -- *"dangling is fine (s1 hasn't run yet)"*.  It was fine only
    because a scheduler dependency stopped the consumer starting early and a
    run-time step localized the link before the engine could write through it.
    Decision 30 deleted all three (2026-08-10).

    So: a job's directory contains its own inputs and the shared package, and
    **nothing that reaches into a sibling**.  What a stage continues from is a
    real file copied by `prepare_attempt` from an attempt you name.
    """
    js = _ladder()
    for f in js.shared + [j.script for j in js.jobs]:
        (tmp_path / f).write_text("x")
    materialize(js, tmp_path)
    d = tmp_path / "bench-s2"
    assert not (d / "demo.XV").exists() and not (d / "demo.XV").is_symlink()
    strays = [e.name for e in d.iterdir()
              if e.is_symlink() and "bench-s1" in os.readlink(e)]
    assert not strays, f"links into another job's directory: {strays}"
    # ...and the legitimate contents are real copies of the shared
    # package.  (The deck is BORN in the directory by prep since the
    # layout repair; materialize no longer places it, so a bare
    # materialize of a hand-built set carries only the shared files.)
    assert sorted(e.name for e in d.iterdir()) == ["C.psml", "mb_monitor.py"]
    assert not any(e.is_symlink() for e in d.iterdir())


def test_materialize_is_idempotent(tmp_path):
    """Running `materialize` twice leaves the same REAL files, with no error and no
    duplication.

    `prep` is re-run routinely -- after editing a deck, after a failed attempt --
    so a second pass that raised on existing directories, or that turned the
    copies into links, would make the ordinary repair workflow the broken path.
    The `is_file() and not is_symlink()` pair is the same property the first-pass
    test asserts: a run directory holds real files, not links up to a root copy.
    """
    js = _ladder()
    for f in js.shared + [j.script for j in js.jobs]:
        (tmp_path / f).write_text("x")
    materialize(js, tmp_path)
    materialize(js, tmp_path)              # no exception, no duplication
    got = tmp_path / "bench-s2" / "C.psml"
    assert got.is_file() and not got.is_symlink()


def test_materialize_rejects_invalid_jobset(tmp_path):
    """`materialize` refuses an invalid job-set instead of laying out directories for
    it.

    `validate` is only useful if the verbs consult it. Without this gate a
    duplicate name reaches the filesystem as one directory holding two jobs'
    files -- and the refusal has to happen BEFORE anything is written, because a
    half-materialised bundle is worse than none.
    """
    js = _ladder()
    js.jobs[1].name = "s1"                 # duplicate
    with pytest.raises(ValueError, match="invalid JobSet"):
        materialize(js, tmp_path)


def test_job_dir_name():
    """A job's directory is `bench-<name>` -- one namer, no second speller.

    Every consumer derives the directory through this function: `materialize`
    creates it, `prep` renders into it, `submit` looks for the wrapper in it, and
    `status` reads the `.out` from it. A hand-built spelling anywhere else is a
    directory nothing else can find. `job-contracts.md` § 6.3 owns the
    identifier conventions.
    """
    assert job_dir_name("stage1") == "bench-stage1"


# --------------------------------------------------------------------- #
#  job_dir_names -- two kinds, two conventions (project-layout.md § 4.1) #
# --------------------------------------------------------------------- #


def _token_ladder(*scripts, optimizers=None):
    """A ladder shaped like the one the described producer emits.

    The warm declaration is part of that shape, not decoration: `prep` reads it
    to decide what `--from` copies, so a fixture without one stands in for a
    ladder no producer builds.  Mirrors the shipped rule -- the first stage is
    ``restart: clean`` and declares nothing; every later one declares SIESTA's
    group with ``.CG`` conditioned on the optimizer (`run-identity.md` § 4).

    ``optimizers`` gives per-stage traits so a test can make two stages
    disagree; by default they all match, which is the case that carries `.CG`.
    """
    from molbuilder.jobset.model import Job, JobSet, WarmFile
    from molbuilder.warmfiles import rules_for
    names = [s.split("_", 2)[2].rsplit(".", 1)[0] for s in scripts]
    opt = dict(optimizers or {})
    # DERIVED from the rules file, exactly as `siesta/stages.py`'s
    # ``_warm_declaration`` derives it.  It was a hard-coded
    # [.XV, .DM, .CG] until 2026-08-15, which made the docstring's claim
    # ("mirrors the shipped rule") false the moment the rules file grew --
    # and a test asserting the copy behaviour then tested the fixture's own
    # list rather than the system's.  § 4.2a's history is exactly this kind
    # of copy drifting; the fixture was the fourth one.
    declared = [WarmFile(f"JOB{r.suffix}", requires_same=r.requires_same)
                for r in rules_for("siesta", "optimization") if r.carry]
    jobs = []
    for i, (name, script) in enumerate(zip(names, scripts)):
        jobs.append(Job(
            name=name, script=script,
            traits={"optimizer": opt.get(name, "CG")},
            warm=([] if i == 0 else list(declared)),
            # STATED, as a described ladder's run card states it -- a run
            # script is not written for an unstated shape (2026-10-02).
            resources=Resources(mpi_np=2, cpus_per_task=1),
        ))
    return JobSet(name="JOB", engine="siesta", kind="ladder", jobs=jobs)


def test_job_dir_names_ladder_uses_the_decks_own_token():
    """A stage directory is ``<seq>_<name>``, and the seq is READ BACK off the
    deck rather than counted here -- counting would reintroduce the shifting
    number ``engines/stages.md`` R5 forbids."""
    from molbuilder.jobset.materialize import job_dir_names
    js = _token_ladder("JOB_01_coarse.fdf", "JOB_02_medium.fdf", "JOB_03_tight.fdf")
    assert job_dir_names(js) == {"coarse": "01_coarse",
                                 "medium": "02_medium",
                                 "tight": "03_tight"}


def test_job_dir_names_ladder_keeps_a_gap_a_gap():
    """Disabling stage 2 leaves 01 and 03 -- the directory does NOT renumber to
    01/02, because the seq belongs to the stage, not to its position."""
    from molbuilder.jobset.materialize import job_dir_names
    js = _token_ladder("JOB_01_coarse.fdf", "JOB_03_tight.fdf")
    assert job_dir_names(js) == {"coarse": "01_coarse", "tight": "03_tight"}


def test_job_dir_names_sweep_keeps_the_point_convention():
    """The benchmark is untouched by the ladder's rule -- and its tokenless
    trials live in the bare ``bench/`` container beside their own record
    (until 2026-08-13 they fell to the root while the record sat in
    ``bench/``, final review A-2).

    Only a HAND-BUILT set arrives here tokenless now: § 6.5 gave every
    description a ladder on 2026-08-16, so every described deck carries a
    token.  The naming authority still owes those an answer, which is why
    the row -- and this test -- stay."""
    from molbuilder.jobset.materialize import job_dir_names
    from molbuilder.jobset.model import Job, JobSet
    js = JobSet(name="JOB", engine="siesta", kind="sweep",
                jobs=[Job(name="np4", script="JOB.fdf"),
                      Job(name="np8", script="JOB.fdf")])
    assert job_dir_names(js) == {"np4": "bench/bench-np4",
                                 "np8": "bench/bench-np8"}


def test_job_dir_names_ladder_without_a_token_falls_back_rather_than_guessing():
    """A hand-written ladder whose deck carries no token gets ``bench-<name>``.
    Inventing a seq for it would be guessing at the one number § 4.2 says is
    assigned once and never reassigned."""
    from molbuilder.jobset.materialize import job_dir_names
    from molbuilder.jobset.model import Job, JobSet
    js = JobSet(name="JOB", engine="siesta", kind="ladder",
                jobs=[Job(name="only", script="JOB.fdf")])
    assert job_dir_names(js) == {"only": "bench-only"}


# --------------------------------------------------------------------- #
#  plan engine                                                          #
# --------------------------------------------------------------------- #

def test_render_plan_shows_warm_files_and_no_order():
    """The plan shows what each job would take, and never an order it waits on.

    It used to print a `depends on` column, a `carries` column and an
    `Order: s1 -> s2` line.  All three named another job; all three went with
    the edges.  What replaces them says the same useful thing -- *this stage
    continues from something* -- without claiming to know what.
    """
    txt = render_plan(_ladder())
    assert "JOB-SET PLAN -- demo (siesta, ladder)" in txt
    assert "C.psml" in txt                 # shared package
    assert "demo.XV" in txt                # the warm declaration
    assert "warm files" in txt             # ...under its own heading
    assert "afterok" not in txt            # no dependency kind survives
    assert "s1 -> s2" not in txt           # and no order it waits on
    # It still says how to run them, because a ladder IS ordered for a person.
    assert "ONE AT A TIME" in txt


def test_render_plan_sweep_says_independent():
    """A sweep's plan tells the reader its jobs are INDEPENDENT.

    The plan is what a person reads before launching, and the two kinds carry
    opposite instructions: a ladder is ordered, a sweep is not (`job-system.md`
    section 3). The wording matters because of the rule beside it -- a scheduler
    is handed one job at a time, so the PERSON is the one sequencing; telling them
    a sweep must be run in order would be a different, and wrong, instruction.
    """
    js = JobSet("sweep", "siesta", "sweep",
                jobs=[Job("a", "a.fdf"), Job("b", "b.fdf")])
    assert "independent" in render_plan(js)


# --------------------------------------------------------------------- #
#  SIESTA stage producer                                                #
# --------------------------------------------------------------------- #

# --------------------------------------------------------------------- #
#  The ``stages_to_jobset`` producer tests were RETIRED 2026-08-12       #
#  (step 6 u5) with the producer, which had no production caller.        #
#  Each property's live home:                                            #
#    * default ladder shape / schema-refused override -> the described   #
#      route (test_prep_calculation.py; resolve refuses by name);        #
#    * the .CG carry conditionals -> test_restart_group.py, repointed    #
#      at the live `_warm_declaration` seam the same day;                #
#    * no-edges / continue_retries on resources -> carried per element   #
#      (test_prep_calculation.py::test_the_allocation_reaches_...);      #
#    * invalid-ladder refusal -> task.py at read + resolve._stage_of.    #
# --------------------------------------------------------------------- #



# --------------------------------------------------------------------- #
#  prep engine (renders wrappers in root, links them into job dirs)     #
# --------------------------------------------------------------------- #

def _sweep() -> JobSet:
    # a SHARED-script sweep: both points use the SAME job-gpu.fdf but differ
    # in resources -- the case that broke the per-job-render model.
    return JobSet(
        name="sw", engine="siesta", kind="sweep",
        jobs=[Job(name="G1K1C4", script="job-gpu.fdf",
                  resources=Resources(mpi_np=1, cpus_per_task=4,
                                      gres="gpu:1", time="0-01:00:00",
                                      mem="8G")),
              Job(name="G1K2C4", script="job-gpu.fdf",
                  resources=Resources(mpi_np=2, cpus_per_task=4,
                                      gres="gpu:1", time="0-01:00:00",
                                      mem="8G"))])


def _write_fdf(path):
    # minimal .fdf so write_run_wrapper renders for real (bash -n validated).
    # The clean restart group rides along: since 2026-08-21 the submission
    # door verifies a trial's cold start against its deck, and a real deck
    # always carries the group written out (`siesta/input.py`, 2026-08-18).
    path.write_text("SystemName test\nSystemLabel test\nNumberOfAtoms 2\n"
                    "DM.UseSaveDM .false.\nMD.UseSaveXV .false.\n")


def test_prep_renders_real_wrappers_into_each_job_dir(tmp_path):
    """L1+L2 (roadmap 7.10, user 2026-08-24): every trial directory holds
    its own REAL deck and wrapper, and the bundle root holds NO rendered
    file.  This asserted the inverse until the layout repair -- one root
    render, symlinked into both dirs -- which is the mechanism that put
    50 rendered files at a real ten-trial bundle's root."""
    js = _sweep()
    _write_fdf(tmp_path / "job-gpu.fdf")
    prep_jobset(js, tmp_path, env="molbuilder-siesta-gpu", emit_sbatch=False)
    # the root-rendered input was ADOPTED into the first trial's dir --
    # nothing rendered remains at the root.
    assert not (tmp_path / "job-gpu.run.sh").exists()
    assert not (tmp_path / "job-gpu.fdf").exists()
    for name in ("bench-G1K1C4", "bench-G1K2C4"):
        d = tmp_path / "bench" / name
        for fn in ("job-gpu.fdf", "job-gpu.run.sh"):
            f = d / fn
            assert f.is_file() and not f.is_symlink(), (name, fn)


def test_prep_bakes_the_warm_retry_budget_into_the_wrapper(tmp_path):
    """**The whole road for `continue_retries`, end to end.**

    job-contracts.md § 6.2: the budget rides ``jobset.Resources`` -- the same
    road ``mpi_np`` and ``omp_threads`` ride -- but becomes no sbatch flag.
    It is baked into the wrapper's own retry loop at install time
    (running-a-job.md § 3.5).

    Asserted on the EMITTED TEXT rather than on a call argument, because the
    defect this closes was exactly a value that travelled correctly and was
    then dropped at the last hop: `job-system.md § 4.1` recorded the SIESTA
    ladder as never having implemented `continue`, and prep not passing the
    field was where it stopped (fixed 2026-08-07, P2 unit 3)."""
    js = JobSet(name="lad", engine="siesta", kind="ladder",
                jobs=[Job(name="tight", script="job.fdf",
                          resources=Resources(mpi_np=1, cpus_per_task=1,
                                            continue_retries=3))])
    _write_fdf(tmp_path / "job.fdf")
    prep_jobset(js, tmp_path, env="molbuilder-siesta", emit_sbatch=False)

    wrapper = (tmp_path / "bench-tight" / "job.run.sh").read_text()
    assert "_siesta_retry_max=3" in wrapper, wrapper
    # and the wrapper SAYS so to the person reading its banner
    assert "3" in wrapper and "etry" in wrapper


def test_prep_omits_the_retry_loop_when_no_budget_is_asked_for(tmp_path):
    """The other half: absent means absent.  A wrapper that always carried a
    retry loop would re-enter SIESTA for jobs nobody asked to retry."""
    js = JobSet(name="lad", engine="siesta", kind="ladder",
                jobs=[Job(name="tight", script="job.fdf",
                          resources=Resources(mpi_np=1, cpus_per_task=1))])
    _write_fdf(tmp_path / "job.fdf")
    prep_jobset(js, tmp_path, env="molbuilder-siesta", emit_sbatch=False)
    assert "_siesta_retry_max=" not in (tmp_path / "bench-tight" / "job.run.sh").read_text()


def test_prep_rejects_missing_script(tmp_path):
    """`prep` refuses when a job's deck is not in the bundle, and names what is
    missing.

    The alternative is a bundle that preps "successfully" and produces a wrapper
    pointing at a file that is not there -- discovered by SLURM hours later as a
    failed job with an unhelpful log, after the queue wait has been paid.
    `PrepError` at the door keeps the diagnosis at the moment of the mistake.
    """
    from molbuilder.jobset.prep import PrepError
    with pytest.raises(PrepError, match="not in"):
        prep_jobset(_sweep(), tmp_path, emit_sbatch=False)


def test_render_plan_surfaces_per_job_ranks_and_cores():
    """The plan shows the per-job `-n` / `-c` / `--gres` variation, because for a
    sweep that variation IS the experiment.

    A benchmark sweep exists to compare resource settings (`job-system.md`
    section 7). A plan printing one shared resource line -- or the first point's
    -- shows a table where every row looks the same, and the trial the person is
    about to launch is not the one they read.
    """
    # the plan MUST show the -n/-c variation -- that IS the sweep.
    txt = render_plan(_sweep())
    assert "n=1" in txt and "n=2" in txt
    assert "c=4" in txt and "gpu:1" in txt


# --------------------------------------------------------------------- #
#  submit engine                                                        #
# --------------------------------------------------------------------- #

def test_submit_dry_run_sweep_per_job_flags_vary(tmp_path):
    """The F2 fix: a SHARED-script sweep must still get per-job ``-n`` via the
    CLI flags, so one rendered ``.sbatch`` serves every point.

    Exercised one point per invocation (a named trial's lane; points also
    reach the scheduler as riders of their shelf's grouped job) — and the
    invariant is the same one: the flags are per-JOB, so two points of one
    sweep must come out different.
    """
    cmds = {}
    for name, want in (("G1K1C4", "1"), ("G1K2C4", "2")):
        res = submit_jobset(_sweep(), tmp_path, mode="submit", dry_run=True,
                            only=name)
        assert len(res) == 1
        cmds[name] = res[0].command
        assert not any(a.startswith("--dependency=") for a in res[0].command)
        assert "--gres=gpu:1" in res[0].command
        assert res[0].command[res[0].command.index("-n") + 1] == want
    assert cmds["G1K1C4"] != cmds["G1K2C4"], "per-job flags did not vary"


def test_submit_slurm_parses_the_id_and_records_the_launch(tmp_path,
                                                           monkeypatch):
    """The id comes back from ``sbatch`` stdout and lands on the result.

    Paired with the threaded-dependency assertion until 2026-08-10; that half
    went with batch submission, and this half is what `status` reads back.
    """
    js = _ladder()
    (tmp_path / "bench-s1").mkdir()
    (tmp_path / "bench-s1" / "demo_s1.sbatch").write_text("x")
    monkeypatch.setattr(_submit.subprocess, "run",
                        lambda *a, **k: _CP(stdout="Submitted batch job 111"))
    res = submit_jobset(js, tmp_path, mode="submit", only="s1")
    assert [(r.job_id, r.status) for r in res] == [("111", "submitted")]


def test_submit_slurm_raises_on_sbatch_failure(tmp_path, monkeypatch):
    """A non-zero `sbatch` exit is raised, carrying the scheduler's own stderr.

    The failure this prevents is the silent one: a submit path ignoring the return
    code reports the job as launched, records no id, and leaves a person watching
    a queue for something that was never queued. `sbatch` refuses for ordinary
    reasons -- a bad QOS, an over-limit time request (`asu-sol.md`) -- and its
    own message is the only thing that says which.
    """
    js = _ladder()
    (tmp_path / "bench-s1").mkdir()
    (tmp_path / "bench-s1" / "demo_s1.sbatch").write_text("x")
    monkeypatch.setattr(_submit.subprocess, "run",
                        lambda *a, **k: _CP(returncode=1, stderr="boom"))
    with pytest.raises(SubmitError, match="sbatch failed"):
        submit_jobset(js, tmp_path, mode="submit", only="s1")


# --------------------------------------------------------------------- #
#  A scheduler is handed ONE job at a time (user rule, 2026-08-10)       #
# --------------------------------------------------------------------- #

def test_a_scheduler_is_never_handed_more_than_one_job(tmp_path):
    """*"SLURM should never submit jobs in parallel.  Submission is manual and
    one by one.  It is a disaster to do parallel job submission on HPC."*

    `_submit_slurm` looped over every job, and its own docstring called the
    result intended: *"a sweep submits with no dependency, so its jobs queue in
    parallel."*  One command, N ``sbatch`` calls, all racing for the same
    nodes.  For a **benchmark** that is not merely antisocial — points running
    concurrently contend for the same cores and interconnect, so the sweep
    measures contention and the numbers are quietly wrong.
    """
    with pytest.raises(SubmitError) as e:
        submit_jobset(_sweep(), tmp_path, mode="submit", dry_run=True)
    msg = str(e.value)
    assert "G1K1C4" in msg and "G1K2C4" in msg      # WHICH jobs it refused
    assert "one at a time" in msg
    assert "--mode direct" in msg                   # ...and what still works


def test_the_refusal_holds_for_a_dry_run_too(tmp_path):
    """A dry run previews the real thing.  Printing the commands for a launch
    that would be refused is a preview of something that cannot happen."""
    with pytest.raises(SubmitError):
        submit_jobset(_sweep(), tmp_path, mode="submit", dry_run=True)


def test_direct_mode_is_untouched_because_it_is_not_submission(tmp_path,
                                                               monkeypatch):
    """The rule is about handing work to a SCHEDULER.  ``--mode direct`` runs
    each job here, in order, waiting for each — nothing queues, nothing races,
    and the user's 2026-08-10 directive keeps the flat shape runnable this
    way."""
    for d in ("bench/bench-G1K1C4", "bench/bench-G1K2C4"):
        (tmp_path / d).mkdir(parents=True)
        (tmp_path / d / "job-gpu.run.sh").write_text("x")
    class _Proc:
        def wait(self):
            return 0
    monkeypatch.setattr(_submit.subprocess, "Popen", lambda *a, **k: _Proc())
    res = submit_jobset(_sweep(), tmp_path, mode="direct")
    assert [r.status for r in res] == ["ran", "ran"]


#  A ladder is submitted one stage at a time, so the tests below cover one
#  launch each.  The scheduler rule that survives -- one job per invocation --
#  is `test_a_scheduler_is_never_handed_more_than_one_job`; the halt-on-failure
#  case is structural, see `test_a_ladder_refuses_to_act_on_all_of_itself`.
#  The earlier scheduler-chained design: docs/archive/2026-08-10-stage-chaining.md

def test_a_failure_skips_nothing_because_nothing_depends_on_anything(tmp_path,
                                                                    monkeypatch):
    """The inverse of what stood here, and it is not a weakening.

    This asserted the SLURM `afterok` meaning reproduced locally: s1 fails,
    so s2 is *skipped*.  That was right while `s2 --afterok--> s1` existed.
    With the edges deleted (2026-08-10) there is nothing to reproduce, and
    skipping would be the framework inventing an order nobody declared.

    **The protection did not go away, it moved up.** A ladder can no longer
    reach this loop with two jobs at all -- `_resolve_stage` refuses to act on
    one without a named stage -- so the case this test guarded (a second stage
    computing from a failed first) is now unreachable rather than handled.
    A `_sweep` is used here because it is the only kind that legitimately
    arrives with several jobs, and its points are independent by definition:
    one bad point says nothing about the next.
    """
    for d in ("bench/bench-G1K1C4", "bench/bench-G1K2C4"):
        (tmp_path / d).mkdir(parents=True)
        (tmp_path / d / "job-gpu.run.sh").write_text("x")
    class _Proc:
        def wait(self):
            return 2
    monkeypatch.setattr(_submit.subprocess, "Popen",
                        lambda *a, **k: _Proc())
    res = submit_jobset(_sweep(), tmp_path, mode="direct")
    assert [r.status for r in res] == ["failed", "failed"], (
        "a failed point must not skip the next -- sweep points are independent")
    assert all(r.returncode == 2 for r in res)


def test_submit_unknown_mode_and_invalid_jobset(tmp_path):
    """Two door refusals: an unknown `mode`, and a job-set that does not validate.

    Both raise `SubmitError`, which is what the CLI renders as a message instead
    of a traceback. The second is the same `validate` gate `materialize` carries,
    asserted again here because `submit` reads `job-set.json` from disk and is
    therefore the verb most likely to meet a file the current code did not write.
    """
    with pytest.raises(SubmitError, match="unknown mode"):
        submit_jobset(_sweep(), tmp_path, mode="bogus", dry_run=True)
    bad = _ladder()
    bad.jobs[1].name = "s1"                         # duplicate
    with pytest.raises(SubmitError, match="invalid JobSet"):
        submit_jobset(bad, tmp_path, mode="submit", dry_run=True)


def test_submit_exclusive_suppresses_mem(tmp_path):
    """`--exclusive` and `--mem` are never both emitted -- exclusive wins.

    A whole-node allocation already grants the node's memory; passing `--mem`
    alongside it caps the job BELOW what it was given, so a request meant to take
    a whole node quietly runs constrained and the benchmark measures the cap.
    `job-contracts.md` § 6.2 maps the config vocabulary onto scheduler
    flags; this is where two of them conflict.
    """
    js = JobSet("x", "siesta", "sweep",
                jobs=[Job("j", "j.fdf",
                          resources=Resources(exclusive=True, mem="120G",
                                              time="0-01:00:00"))])
    cmd = submit_jobset(js, tmp_path, mode="submit", dry_run=True)[0].command
    assert "--exclusive" in cmd
    assert not any(a.startswith("--mem") for a in cmd)   # exclusive wins


# --------------------------------------------------------------------- #
#  persistence (job-set.json write/load)                                #
# --------------------------------------------------------------------- #

def test_jobset_write_load_roundtrip(tmp_path):
    """`write` -> `load` through a real file on disk is lossless.

    The in-memory `to_dict` / `from_dict` round-trip is asserted above; this is
    the half that goes through JSON on disk, where a tuple comes back a list and a
    `Path` refuses to encode at all. Every verb after `prep` reads this file
    rather than the object that made it (`job-system.md` § 3.1).
    """
    js = _ladder()
    p = js.write(tmp_path / "job-set.json")
    assert p.is_file()
    assert JobSet.load(p).to_dict() == js.to_dict()      # lossless on disk


# --------------------------------------------------------------------- #
#  molbuilder jobset CLI (status / prep / submit over a bundle)         #
# --------------------------------------------------------------------- #

def _runner():
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    return CliRunner(), jobset_group


def test_cli_errors_without_jobset_json(tmp_path):
    """A directory that is not a bundle exits non-zero and says `no job-set.json`.

    Pointing `--bundle` at the wrong directory is the ordinary mistake. Exiting 0
    with an empty plan would read as "this bundle has no jobs" -- a different and
    much more alarming thing to be told about work you believe you prepared.
    """
    runner, grp = _runner()
    r = runner.invoke(grp, ["status", "--bundle", str(tmp_path)])
    assert r.exit_code != 0
    assert "no job-set.json" in r.output


def test_cli_prep_is_described_only(tmp_path):
    """U2/U4 (2026-08-12): the pre-made-bundle arm is RETIRED -- `describe`
    is floor 2's only writer, and the old bench bundles died in step 6 u5.
    A folder without the task.json + template pair is refused with the next
    step named; hand-built job-sets stay LAUNCHABLE (submit/status read
    job-set.json directly) -- prep is what they lose."""
    js = _sweep()
    js.write(tmp_path / "job-set.json")
    _write_fdf(tmp_path / "job-gpu.fdf")
    runner, grp = _runner()
    r = runner.invoke(grp, ["prep", "bench", "--bundle", str(tmp_path),
                            "--no-sbatch"])
    assert r.exit_code != 0
    assert "not a described calculation" in r.output
    assert "jobset init" in r.output
    # the library route for laying out a hand-built set is prep_jobset,
    # pinned by the prep_jobset tests above; nothing was laid out here
    assert not (tmp_path / "bench-G1K1C4").exists()


def test_submit_accepts_exactly_these_options(tmp_path):
    """`jobset launch` takes a kind, a stage, a trial and the options below --
    no more.  TRIAL names one benchmark point (§ 2.3.2, decided
    2026-08-12); `run` refuses it.

    An equality, and at the CLI because that is the surface a person types: an
    option added without a decision fails here whatever it is called.

    Three joined on 2026-08-23, and together they are **one question, one
    answer, one interface** (`jobset/ask.py`): ``time_text`` and ``mem_text``
    ask what the job needs — the person knows that better than any rule the
    framework could write, so it asks rather than deriving — and ``auto_yes``
    is how they say *I have decided to trust this*, its absence being no kind
    of permission.

    ``trial_timeout_min`` stays as the direct refinement — *no single trial
    may exceed this* — which is the same answer said the other way round, not
    a second knob for one thing.
    """
    from molbuilder.jobset._cli import submit_cmd
    assert {q.name for q in submit_cmd.params} == {
        "kind", "stage", "trial", "bundle", "mode", "domain", "dry_run",
        "time_text", "mem_text", "gpu_domain", "auto_yes",
        "trial_timeout_min", "only_side"}


def test_direct_launch_carries_the_launch_door_claim(tmp_path, monkeypatch):
    """`submit --mode direct` sets MB_LAUNCHED_BY in the child env — the
    claim the wrapper's launch-door gate checks (job-contracts.md § 2.6).
    Env inheritance survives forks, so a backgrounded local run launched
    through the verb never meets the gate's prompt."""
    import molbuilder.jobset.submit as sub
    seen = {}

    def fake_popen(cmd, **kw):
        seen["env"] = kw.get("env")
        class _Proc:
            def wait(self):
                return 0
        return _Proc()
    js = _sweep()
    _write_fdf(tmp_path / "job-gpu.fdf")
    prep_jobset(js, tmp_path, emit_sbatch=False)
    monkeypatch.setattr(sub.subprocess, "Popen", fake_popen)
    sub.submit_jobset(js, tmp_path, mode="direct", only=js.jobs[0].name)
    assert seen["env"]["MB_LAUNCHED_BY"] == "jobset-launch"


def _fake_status(states):
    """A `run_status` stand-in: dir-name -> state.

    These tests are about the LADDER -- first_incomplete, complete --
    not about how a state is read off a directory, so the read is
    stubbed.  It stubbed `decode_run_dir` until 2026-09-04, when the one
    field anyone used was split out of that eleven-field summary and the
    caller moved to `run_status`.
    """
    def fake(run_dir, match="*", *, launch=None):
        # A `RunStatus`, not a look-alike dict.  The stub is now bound by the
        # same constructor as the real thing, so a stubbed state outside
        # `RUN_STATES` raises here instead of flowing into the ladder as a
        # value production can never produce (2026-09-09).
        from molbuilder.parse.dirs.job import RunStatus
        s = states[Path(run_dir).name]
        return RunStatus(state=s, detail=s)
    return fake


def test_a_flat_rung_is_asked_about_by_name_not_by_directory(tmp_path):
    """ONE DIRECTORY, EVERY RUNG -- that is what the flat shape is
    (`project-layout.md` § 1), so "how is this directory doing" is not the
    question anyone means; "how is THIS RUNG doing" is.

    `run_status` bucketed the whole directory and picked the highest stage
    ordinal / newest mtime, so every rung of a flat calculation reported the
    newest rung's state.  The caller's existence check was already narrowed by
    `Shape.stage_glob` -- and said in a comment why -- but the filter stopped
    at the gate and never reached here.

    MEASURED 2026-09-08 on a built fixture: with the later rung's `.out`
    present a finished, day-old rung read ('running', 'running').
    """
    from molbuilder.parse.dirs.job import run_status
    old = tmp_path / "bdt_01_coarse-run0.out"
    new = tmp_path / "bdt_02_tight-run0.out"
    old.write_text("Job completed\n")
    new.write_text("still going\n")
    past = _time.time() - 86400
    os.utime(old, (past, past))

    coarse = run_status(tmp_path, "bdt_01_coarse*")
    tight = run_status(tmp_path, "bdt_02_tight*")
    assert coarse.active_source == old.name
    assert tight.active_source == new.name
    # The hierarchical answer is unchanged: the DIRECTORY selected the rung,
    # so no glob is passed and the newest file still speaks for it.
    assert run_status(tmp_path).active_source == new.name


def test_status_fresh_bundle_all_not_started(tmp_path):
    """A bundle where nothing has been prepped reports every stage `not-started`, and
    points at the first one.

    `not-started` has to stay distinguishable from `pending` (a directory exists,
    nothing has run) and from `finished`, because `first_incomplete` is what a
    person is told to resume from (`job-system.md` § 5.4) -- a fresh bundle
    reporting anything else sends them to the wrong stage.
    """
    st = jobset_status(_ladder(), tmp_path)        # nothing prepped
    assert [s.state for s in st.stages] == ["not-started", "not-started"]
    assert st.first_incomplete == "s1" and st.complete is False


def test_status_pending_and_warm_files(tmp_path):
    """A directory with warm files but no output reads `pending`, and the warm files
    are NAMED.

    "Prepared but not run" and "ran and produced nothing" are the two states a
    person most needs told apart, and both look like an absent `.out`. The warm
    list is what the next stage would continue from (`job-contracts.md` section
    4.2), so showing it is how someone checks a restart will actually be warm
    before spending the allocation to find out.
    """
    (tmp_path / "bench-s1").mkdir()
    (tmp_path / "bench-s1" / "demo.XV").write_text("x")   # label = jobset.name
    st = jobset_status(_ladder(), tmp_path)
    assert st.stages[0].state == "pending"                # dir, no .out
    assert "demo.XV" in st.stages[0].warm_files


def test_a_trials_warm_files_are_found_under_the_trials_own_label(tmp_path):
    """The warm list is read with the SAME label the state is read with.

    A sweep trial is relabelled — SIESTA finds its warm files by
    `SystemLabel`, so a trial carrying the real run's label could read or
    overwrite the real run's `.DM` and `.XV` (`project-layout.md` § 2.3.2)
    — and `_label_of` exists to recover that label off the deck.  It was
    applied to the `.out` and not to the warm files beside it, so the
    column asked for `<task.label>.XV` in a directory holding
    `<task.label>-<token>.XV`.

    MEASURED before the fix on this exact fixture: `warm_files == []` with
    `siesta-AuBDTAu-G0K20C1.XV` on disk — `jobset status` telling a person
    there is nothing to restart from, which is the answer that costs an
    allocation to disprove (`job-contracts.md` § 4.2).
    """
    js = JobSet(
        name="siesta-AuBDTAu", engine="siesta", kind="sweep",
        jobs=[Job(name="G0K20C1",
                  script="siesta-AuBDTAu-G0K20C1_01_coarse.fdf",
                  resources=Resources(mpi_np=1),
                  warm=[WarmFile("siesta-AuBDTAu-G0K20C1.XV")])])
    trial = tmp_path / "01_coarse/bench/bench-G0K20C1"
    trial.mkdir(parents=True)
    (trial / "siesta-AuBDTAu-G0K20C1.XV").write_text("x")

    st = jobset_status(js, tmp_path)
    assert st.stages[0].warm_files == ["siesta-AuBDTAu-G0K20C1.XV"]


def test_status_first_incomplete_advances(tmp_path, monkeypatch):
    """With the first stage finished and the second running, `first_incomplete` moves
    to the second and the set is not complete.

    `first_incomplete` is the resume pointer (`job-system.md` § 5.4) and it
    has to follow the stage STATES, not the row order: a pointer stuck at stage
    one sends a person to re-run finished work, and one that runs ahead skips a
    stage still going. That `running` counts as incomplete is the part a plain
    "is it finished?" test would not pin.
    """
    # PATCH THE DOOR, not the module behind it.  `runstatus` imports
    # `run_status` from `molbuilder.parse.dirs` -- the package that owns the
    # question (`model/parse.md` § 5.5, R-RO2) -- so patching
    # `parse.dirs.job` leaves the re-export pointing at the real function.
    import molbuilder.parse.dirs as jobmod
    for n in ("bench-s1", "bench-s2"):
        d = tmp_path / n; d.mkdir(); (d / "demo.out").write_text("x")
    monkeypatch.setattr(jobmod, "run_status",
                        _fake_status({"bench-s1": "finished",
                                       "bench-s2": "running"}))
    st = jobset_status(_ladder(), tmp_path)
    assert st.stages[0].state == "finished"
    assert st.first_incomplete == "s2" and st.complete is False


def test_status_complete_when_all_finished(tmp_path, monkeypatch):
    """Every stage finished means `complete` is True, `first_incomplete` is None, and
    the rendered status SAYS so.

    The terminal state has to be unambiguous in both the object and the text a
    person reads. `first_incomplete` left pointing at the last stage offers a
    resume for a ladder that is done -- an allocation spent to discover the work
    was already there.
    """
    # PATCH THE DOOR, not the module behind it.  `runstatus` imports
    # `run_status` from `molbuilder.parse.dirs` -- the package that owns the
    # question (`model/parse.md` § 5.5, R-RO2) -- so patching
    # `parse.dirs.job` leaves the re-export pointing at the real function.
    import molbuilder.parse.dirs as jobmod
    for n in ("bench-s1", "bench-s2"):
        d = tmp_path / n; d.mkdir(); (d / "demo.out").write_text("x")
    monkeypatch.setattr(jobmod, "run_status",
                        _fake_status({"bench-s1": "finished",
                                       "bench-s2": "finished"}))
    st = jobset_status(_ladder(), tmp_path)
    assert st.complete is True and st.first_incomplete is None
    assert "All stages finished" in render_status(st)


def test_render_status_shows_resume_pointer(tmp_path):
    """The rendered status names the calculation, the stage to resume from, and that
    nothing resumes on its own.

    `does NOT auto-resume` is the load-bearing line: nothing in the system
    advances a ladder -- the person launches each stage (`job-system.md` section
    5.4) -- so a status screen that only reported states would let someone leave a
    bundle sitting for days believing the next stage was queued.
    """
    txt = render_status(jobset_status(_ladder(), tmp_path))
    assert "JOB-SET STATUS -- demo" in txt
    assert "First incomplete stage: s1" in txt
    assert "does NOT auto-resume" in txt


def test_cli_status(tmp_path):
    """`jobset status --bundle <dir>` renders the status through the CLI.

    The path an operator on a login node actually uses: read `job-set.json` from
    the bundle, compute the status, print it. RECORDED DOUBT for the section 3b
    review -- the command is a thin caller of `jobset_status` + `render_status`,
    both tested directly above, so what this adds is that the subcommand is wired
    at all and exits 0 on a bundle where nothing has run.
    """
    _ladder().write(tmp_path / "job-set.json")
    runner, grp = _runner()
    r = runner.invoke(grp, ["status", "--bundle", str(tmp_path)])
    assert r.exit_code == 0, r.output
    assert "JOB-SET STATUS" in r.output and "First incomplete" in r.output


def test_status_finished_with_real_siesta_out(tmp_path):
    """The status reader reaches `finished` through the REAL parser on a real SIESTA
    output, with nothing monkeypatched.

    Its neighbours stub `run_status` to place a state; this one does not, which
    makes it the only test in the group that would notice the status dict's shape
    drifting between `parse.dirs.job` and `jobset.runstatus`. Under that drift
    every stubbed test still passes and every real bundle reports the wrong state.
    The fixture is a frozen finished run from `tests/watch/fixtures/siesta_frozen`.
    """
    # DEPTH: the real decode_run_dir -> "finished" path (not monkeypatched),
    # so a drift in the status-dict shape between decode + runstatus is caught.
    import shutil
    fix = (Path(__file__).parent / "watch" / "fixtures" / "siesta_frozen"
           / "hemeC-stage2-run3-finished-42fr.out")
    d = tmp_path / "bench-s1"; d.mkdir()
    shutil.copy(fix, d / "demo.out")            # label = jobset.name = "demo"
    st = jobset_status(_ladder(), tmp_path)
    assert st.stages[0].state == "finished"     # REAL parse of a finished run


# --------------------------------------------------------------------- #
#  carry-forward BEHAVIOR (the §4 isolation guarantee, end-result)       #
# --------------------------------------------------------------------- #

def test_a_wrapper_is_made_of_exactly_these_blocks(tmp_path):
    """The emitted wrapper contains exactly the blocks the CONTRACT lists.

    `job-contracts.md` § 2.6 tabulates them, and this reads that table rather
    than carrying its own copy — so the rule has one home. Adding a block is a
    contract change: each one is work happening on a compute node, which
    `running-a-job.md` § 2.2a keeps narrow on purpose (the wrapper activates and
    execs; anything that computes, decides or arranges files is Python's, on the
    host).

    An equality in both directions: a new block fails until it is documented,
    and a documented block that stops being emitted fails too.
    """
    import re as _re
    from molbuilder.runwrap import write_run_wrapper

    doc = (Path(__file__).resolve().parent.parent
           / "docs" / "execution" / "job-contracts.md").read_text(encoding="utf-8")
    start = doc.index("#### What a wrapper is made of")
    # stop at the sentence that closes the table -- the section continues with
    # OTHER tables, and running past this one silently harvested their rows.
    end = doc.index("**Adding a block is a contract change", start)
    rows = {m.group(1).strip(): m.group(2)
            for m in _re.finditer(r"^\| \*\*(.+?)\*\* \|([^|]*)\|",
                                  doc[start:end], _re.M)}
    documented = set(rows)
    conditional = {name for name, desc in rows.items() if "*(" in desc}
    assert documented, "§ 2.6's wrapper table could not be read — repoint this"

    # documented rows may carry a *(conditional: ...)* tag -- strip it the
    # same way headers strip their parentheticals
    documented = {d.split("*(")[0].strip() for d in documented}

    def _blocks(txt):
        return {h.split("(")[0].strip()
                for h in _re.findall(r"^# --- (.+?) -*$", txt, _re.M)}
    (tmp_path / "JOB.fdf").write_text(
        "SystemName j\nSystemLabel JOB\nNumberOfAtoms 100\n"
        "NumberOfSpecies 1\nMeshCutoff 300 Ry\nBasis.Size DZP\n")
    minimal = _blocks(write_run_wrapper(tmp_path / "JOB.fdf", resources=Resources(mpi_np=4, cpus_per_task=1), env="e").read_text())
    # The MAXIMAL wrapper (R9, 2026-08-12): a GPU deck with an estimable
    # size and a retry budget emits the four conditional blocks the table
    # omitted -- and this guard, rendering only the minimal wrapper,
    # could not see that its own "exhaustive" claim was false.
    (tmp_path / "GPU.fdf").write_text(
        "SystemName g\nSystemLabel GPU\nNumberOfAtoms 100\n"
        "NumberOfSpecies 1\nMeshCutoff 300 Ry\nBasis.Size DZP\n"
        "Diag.ELPA.GPU .true.\n")
    maximal = _blocks(write_run_wrapper(tmp_path / "GPU.fdf", resources=Resources(mpi_np=4, cpus_per_task=1, gres="gpu:1", continue_retries=2), env="e").read_text())
    # The ESTIMABLE CPU deck (D9 tightening, user decision 2026-08-13):
    # the Memory block renders only from a chemically parseable deck
    # (species + coordinates) -- which NEITHER fixture above carries, so
    # its row lived on the conditional escape and could stop rendering
    # without any test noticing.  (The comment claiming JOB.fdf was
    # "estimable" was false -- the G2 re-review's finding.)
    (tmp_path / "EST.fdf").write_text(
        "SystemName e\nSystemLabel EST\nNumberOfAtoms 2\n"
        "NumberOfSpecies 1\nMeshCutoff 300 Ry\nBasis.Size DZP\n"
        "%block ChemicalSpeciesLabel\n1 1 H\n"
        "%endblock ChemicalSpeciesLabel\n"
        "%block AtomicCoordinatesAndAtomicSpecies\n"
        "0.0 0.0 0.0 1\n0.0 0.0 0.74 1\n"
        "%endblock AtomicCoordinatesAndAtomicSpecies\n")
    estimable = _blocks(write_run_wrapper(tmp_path / "EST.fdf", resources=Resources(mpi_np=2, cpus_per_task=1), env="e").read_text())
    # A JOB WITH A FINISH (a SIESTA force-constant stage, `engines/
    # vibration.md` § 5.5): the check before the engine and the finish after
    # it render only with one -- which no fixture above carried, so both
    # blocks were emitted and unguarded (the review of 51590fa6).
    (tmp_path / "FIN.fdf").write_text((tmp_path / "EST.fdf").read_text()
                                      .replace("SystemLabel EST",
                                               "SystemLabel FIN"))
    finishing = _blocks(write_run_wrapper(tmp_path / "FIN.fdf", resources=Resources(mpi_np=2, cpus_per_task=1), env="e", finish="mb_vibration.pyz").read_text())
    # PySCF (D9: the guard never rendered one, so its parsing header
    # matched no row and its anatomy was unguarded entirely)
    (tmp_path / "PY.py").write_text('JOB = "PY"\nimport pyscf\n')
    pyscf = _blocks(write_run_wrapper(tmp_path / "PY.py", resources=Resources(cpus_per_task=1), env="e").read_text())
    union = minimal | maximal | estimable | pyscf | finishing
    assert union <= documented, (
        "the wrapper emits blocks job-contracts.md § 2.6 does not list:\n"
        f"  {sorted(union - documented)}")
    # EVERY documented row must render in at least one fixture -- the
    # conditional tag describes WHEN a block appears, it is not an
    # exemption from proof.  Before this (D9), any conditional-tagged
    # row could silently stop being emitted and the guard stayed green.
    unrendered = documented - union
    assert not unrendered, (
        "§ 2.6 lists blocks no fixture renders -- extend the fixture set "
        f"or retire the rows:\n  {sorted(unrendered)}")
    assert conditional, "the conditional tags vanished from the table"


def test_prep_writes_stage_plan_md(tmp_path):
    """J1 (D3): prep emits STAGE-PLAN.md into the bundle (bench parity)."""
    js = _sweep()
    _write_fdf(tmp_path / "job-gpu.fdf")
    prep_jobset(js, tmp_path, emit_sbatch=False)
    plan = tmp_path / "STAGE-PLAN.md"
    assert plan.is_file()
    assert "JOB-SET PLAN" in plan.read_text()


# --------------------------------------------------------------------- #
#  StageRef — one resolver, six callers (plan § 8f, decision 28)         #
# --------------------------------------------------------------------- #


def test_stage_refs_reads_the_seq_off_the_deck_not_the_row():
    """The after-produce half: seq comes from each deck's token, so a disabled
    stage leaves 01/03 rather than being renumbered to the rows 0/1."""
    from molbuilder.jobset.materialize import stage_refs
    js = _token_ladder("JOB_01_coarse.fdf", "JOB_03_tight.fdf")
    refs = stage_refs(js)
    assert (refs["coarse"].seq, refs["tight"].seq) == (1, 3)
    assert refs["tight"].token == "03_tight"


def test_stage_refs_gives_a_tokenless_job_no_seq_rather_than_inventing_one():
    """§ 4.2's number is assigned once and never guessed -- so a deck with no
    token yields ``seq=None``, and no token to name a directory from."""
    from molbuilder.jobset.materialize import stage_refs
    from molbuilder.jobset.model import Job, JobSet
    js = JobSet(name="JOB", engine="siesta", kind="ladder",
                jobs=[Job(name="only", script="JOB.fdf")])
    ref = stage_refs(js)["only"]
    assert (ref.seq, ref.token, ref.label) == (None, None, "only")


def test_stage_refs_is_total_so_no_caller_has_to_ask_whether_a_job_is_in_it():
    """Every job gets a ref, both kinds.  Omission was what pushed *what if
    there is no ordinal?* out to four callers, who answered it four ways."""
    from molbuilder.jobset.materialize import stage_refs
    from molbuilder.jobset.model import Job, JobSet
    sweep = JobSet(name="JOB", engine="siesta", kind="sweep",
                   jobs=[Job(name="p1", script="JOB_p1.fdf"),
                         Job(name="p2", script="JOB_p2.fdf")])
    refs = stage_refs(sweep)
    assert set(refs) == {"p1", "p2"}
    assert [r.seq for r in refs.values()] == [None, None]
    assert refs["p1"].token is None          # a point has no token to be named from


def test_stage_refs_carries_the_jobs_name_not_the_tokens():
    """The CLI resolves to a name and then looks the JOB up by it, so a ref must
    never hand back a string this JobSet does not have."""
    from molbuilder.jobset.materialize import stage_refs
    from molbuilder.jobset.model import Job, JobSet
    js = JobSet(name="JOB", engine="siesta", kind="ladder",
                jobs=[Job(name="tight", script="JOB_03_renamed.fdf")])
    ref = stage_refs(js)["tight"]
    assert (ref.seq, ref.name) == (3, "tight")


def test_job_dir_names_sweep_is_unchanged_by_the_total_refs():
    """A sweep point's directory is `bench/bench-<name>` -- nested, unlike a ladder
    rung's.

    The two kinds use two conventions (`project-layout.md` § 4.1): a ladder
    is flat, one directory per calculation, and a sweep's trials live under a
    `bench/` of their own. This pins the sweep half against the change that made
    `stage_refs` total -- giving every job a ref, points with no ordinal included
    -- because the risk of that change was precisely that the namer would start
    treating a point like a rung.
    """
    from molbuilder.jobset.materialize import job_dir_names
    from molbuilder.jobset.model import Job, JobSet
    js = JobSet(name="JOB", engine="siesta", kind="sweep",
                jobs=[Job(name="p1", script="JOB_p1.fdf")])
    assert job_dir_names(js) == {"p1": "bench/bench-p1"}


def test_a_sweep_point_prints_a_dash_not_its_row_under_seq(tmp_path):
    """The rename made `#` mean `seq`; falling back to the row for a kind that
    has no ordinal is the same defect wearing the new column's name."""
    from molbuilder.jobset.model import Job, JobSet
    js = JobSet(name="JOB", engine="siesta", kind="sweep",
                jobs=[Job(name="p1", script="JOB_p1.fdf"),
                      Job(name="p2", script="JOB_p2.fdf")])
    body = [l for l in render_plan(js).splitlines() if "p2" in l]
    assert body[0].split()[0] == "-"          # NOT "1", which the row would be
    # tmp_path, never ".": status READS the filesystem, and a repo that
    # happened to hold a `bench-p2/` would decide this test's outcome.
    out = render_status(jobset_status(js, tmp_path))
    assert [l.split()[0] for l in out.splitlines() if "p2" in l] == ["-"]


def test_prepare_attempt_takes_the_same_two_spellings_as_every_surface():
    """One vocabulary everywhere (2026-08-10's fix, re-ruled 2026-08-21):
    a stage is its NAME, or `#N` its assigned number.  The bare number and
    the token were retired the same day -- both are legal stage NAMES
    ([A-Za-z0-9_]+), so `2` was ambiguous with an ordinal; `#` cannot
    appear in a name."""
    import tempfile
    import pytest as _pt
    from molbuilder.jobset.materialize import prepare_attempt
    js = _token_ladder("JOB_01_coarse.fdf", "JOB_03_tight.fdf")
    with tempfile.TemporaryDirectory() as td:
        for spelling in ("tight", "#3"):
            rep = prepare_attempt(js, td, spelling)
            assert rep.stage == "tight"           # the NAME, always
            assert rep.dir.parent.name == "03_tight"
        for retired in ("3", "03", "03_tight"):
            with _pt.raises(ValueError, match="coarse \\('#1'\\)"):
                prepare_attempt(js, td, retired)


def test_prepare_attempt_refuses_with_the_one_listing_that_carries_ordinals():
    """decision 28's gap verbatim: the refusal listed 'coarse, medium, tight'
    with no order, at the one moment you are choosing which stage to run.
    The listing offers the TYPEABLE spellings -- name and #N (user-settled
    2026-08-21) -- not the on-disk token nobody can type any more."""
    import tempfile
    from molbuilder.jobset.materialize import prepare_attempt
    js = _token_ladder("JOB_01_coarse.fdf", "JOB_03_tight.fdf")
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(ValueError) as e:
            prepare_attempt(js, td, "bogus")
    assert "coarse ('#1'), tight ('#3')" in str(e.value)


def test_an_unlaunched_attempt_is_reused_and_a_launched_one_is_never_touched():
    """§ 1.5: an attempt is immutable once it has run, so a re-run is a NEW
    directory.  Before it has run there is nothing to preserve, and minting
    run-1 beside an empty run-0 would just litter."""
    import tempfile
    from pathlib import Path
    from molbuilder.jobset.materialize import prepare_attempt, write_run_launch
    js = _token_ladder("JOB_03_tight.fdf")
    with tempfile.TemporaryDirectory() as td:
        first = prepare_attempt(js, td, "tight").dir
        assert first.name == "run-0"
        assert prepare_attempt(js, td, "tight").dir == first   # reused
        write_run_launch(first, mode="direct", command=["bash", "x.sh"])
        second = prepare_attempt(js, td, "tight").dir
        assert second.name == "run-1"                             # never reused
        assert (Path(first) / "run.json").is_file()               # left intact


def test_prep_says_reused_only_of_an_attempt_an_earlier_prep_opened(
        isolated_projects_root):
    """The report tells a new attempt from one an earlier prep left behind.

    GOAL: every `prep run` of the 2026-09-25 transport ladder said "(reused --
    not launched yet)" of a directory it had just made -- `prep_calculation`
    opens the attempt, and the CLI opened it again for its report and found it
    unlaunched.  CONTRACT: `Attempt.fresh` (`jobset/materialize.py`) is False
    only when an unlaunched attempt was REUSED rather than opened
    (`project-layout.md` § 1.6).  (Its second half -- a re-prep saying
    "reused" -- retired 2026-10-02 with re-preps: a prepped stage is refused
    now, `job-system.md` § 5.0.)
    """
    tree = isolated_projects_root
    (tree / "P" / "structure").mkdir(parents=True)
    (tree / "P" / "structure" / "h2.xyz").write_text(
        "2\nh2\nH 0 0 0\nH 0 0 0.74\n")
    runner, grp = _runner()
    init = runner.invoke(grp, ["init", "--structure", "P/structure/h2.xyz",
                               "--bundle", "P/optimization/h",
                               "--shape", "hierarchical", "--engine", "pyscf"])
    assert init.exit_code == 0, init.output
    prep = ["prep", "run", "coarse", "--bundle", "P/optimization/h",
            "--no-sbatch", "--cpus-per-task", "1"]
    first = runner.invoke(grp, prep)
    assert first.exit_code == 0, first.output
    assert "prepared coarse: " in first.output, first.output
    assert "(reused" not in first.output, first.output


# --------------------------------------------------------------------- #
#  The observe layer vs the attempt layer (project-layout.md § 1.5, 1.6) #
# --------------------------------------------------------------------- #


def test_status_reads_the_attempt_because_that_is_where_the_run_happened(tmp_path):
    """`project-layout.md` § 1.5, *"Where a run happens: inside the attempt
    directory"* -- so a stage whose output is in run-0 has RUN, and status that
    globs the container reports it as never launched, forever."""
    from molbuilder.jobset.materialize import prepare_attempt
    js = _token_ladder("JOB_03_tight.fdf")
    attempt = prepare_attempt(js, tmp_path, "tight").dir
    (attempt / "JOB_03_tight.out").write_text("Job completed\n")

    st = jobset_status(js, tmp_path).stages[0]
    assert st.attempt == "run-0"                 # says WHICH attempt it read
    # A POSITIVE claim: the decoder was reached and returned one of its own
    # verdicts.  `!= "pending"` would also pass for "unknown", which is what
    # this reports when the decoder THROWS -- a broken decoder would look like
    # a working fix.
    assert st.state in ("running", "finished", "failed")
    assert "not launched" not in st.detail


def test_every_table_column_gets_a_rule_segment(tmp_path):
    """The widths and the rule were two hand-written column counts, and adding
    `attempt` desynchronised them at once: six headings over a five-segment
    rule.  Both are driven off the header now, so this cannot recur."""
    import re
    js = _token_ladder("JOB_01_coarse.fdf", "JOB_03_tight.fdf")
    for out in (render_plan(js), render_status(jobset_status(js, tmp_path))):
        lines = out.splitlines()
        # Find the rule rather than index it: the two tables do not start at
        # the same offset, which is how this test's own first draft made the
        # very mistake it exists to catch.
        rule = next(l for l in lines if l.strip() and set(l.strip()) <= {"-", " "})
        header = lines[lines.index(rule) - 1]
        assert len(rule.split()) == len(re.split(r"\s{2,}", header.strip()))


def test_warm_files_are_read_from_the_attempt_not_the_container(tmp_path):
    """Same sentence, other half: what a run WRITES is created in place, so the
    restart files a user is deciding on are in the attempt."""
    from molbuilder.jobset.materialize import prepare_attempt
    js = _token_ladder("JOB_03_tight.fdf")
    attempt = prepare_attempt(js, tmp_path, "tight").dir
    (attempt / "JOB.XV").write_text("")
    (attempt / "JOB_03_tight.out").write_text("Job completed\n")

    assert jobset_status(js, tmp_path).stages[0].warm_files == ["JOB.XV"]


def test_a_launched_attempt_with_no_output_is_queued_not_not_started(tmp_path):
    """`project-layout.md` § 1.6: *"a queued cluster job has produced nothing
    yet, so 'no output' and 'not started' look identical"* -- run.json is what
    tells them apart, and status *"can say queued as job 481923 instead of
    guessing from an absence"*."""
    from molbuilder.jobset.materialize import prepare_attempt, write_run_launch
    js = _token_ladder("JOB_03_tight.fdf")
    attempt = prepare_attempt(js, tmp_path, "tight").dir

    before = jobset_status(js, tmp_path).stages[0]
    assert before.state == "pending"             # prepped, genuinely not launched

    write_run_launch(attempt, mode="submit", command=["sbatch", "x"],
                     job_id="481923")
    after = jobset_status(js, tmp_path).stages[0]
    assert after.state == "queued"
    assert "481923" in after.detail              # the contract's own sentence


def test_a_continue_carries_the_accumulative_records_too(tmp_path):
    """§ 2.3.4's last row: the layout must not change the data.

    SIESTA opens .MD.nc / .MD / .MDE / .ANI and APPENDS.  In `flat` every
    attempt shares one directory, so those files end up holding the whole
    calculation.  In `hierarchical` each attempt is its own directory -- so
    unless they are carried, a continued stage starts with empty records and
    the earlier frames survive only in the previous attempt.  Same run, same
    continue, different record depending on a layout flag.

    .MD.nc is the one molbuilder READS (the trajectory source), so a
    truncated one silently shortens a continued stage's history."""
    from molbuilder.jobset.materialize import prepare_attempt
    js = _token_ladder("JOB_01_coarse.fdf", "JOB_03_tight.fdf")
    coarse = prepare_attempt(js, tmp_path, "coarse").dir
    (coarse / "JOB.XV").write_text("geometry from coarse\n")
    (coarse / "JOB.MD.nc").write_bytes(b"CDF\x01 pretend netcdf")
    (coarse / "JOB.MDE").write_text("# Step T E_KS\n     0  0.0  -1.0\n")
    (coarse / "JOB.ANI").write_text("2\nframe 0\nH 0 0 0\nH 0 0 0.74\n")

    warm = prepare_attempt(js, tmp_path, "tight",
                           continue_from="01_coarse/run-0")
    for name in ("JOB.XV", "JOB.MD.nc", "JOB.MDE", "JOB.ANI"):
        assert (warm.dir / name).is_file(), f"{name} was not carried"
        assert name in warm.copied
    # A real copy, never a link -- the engine appends to it, and appending
    # through a link would rewrite the attempt we decided to build on.
    assert not (warm.dir / "JOB.MD.nc").is_symlink()
    assert (warm.dir / "JOB.MD.nc").read_bytes() == b"CDF\x01 pretend netcdf"


def test_an_absent_accumulative_record_is_not_an_error(tmp_path):
    """write_md_history / write_md_xmol off means the files were never
    written.  Declaring them must not make a continue fail for a run that
    legitimately has none of them."""
    from molbuilder.jobset.materialize import prepare_attempt
    js = _token_ladder("JOB_01_coarse.fdf", "JOB_03_tight.fdf")
    coarse = prepare_attempt(js, tmp_path, "coarse").dir
    (coarse / "JOB.XV").write_text("geometry from coarse\n")   # and nothing else

    warm = prepare_attempt(js, tmp_path, "tight",
                           continue_from="01_coarse/run-0")
    assert warm.copied == ["JOB.XV"]
    assert not (warm.dir / "JOB.MD.nc").exists()


def test_re_prepping_cold_removes_what_the_previous_prep_carried_in(tmp_path):
    """§ 1.6 makes re-prep *"changing your mind about the setup"*.  A mind
    changed from `--from A` to `--cold` that leaves A's .XV in the directory has
    changed nothing: the engine finds it and warm-starts anyway.  That is the
    *"present but not honoured"* failure inverted, and it is silent."""
    from molbuilder.jobset.materialize import prepare_attempt
    js = _token_ladder("JOB_01_coarse.fdf", "JOB_03_tight.fdf")
    coarse = prepare_attempt(js, tmp_path, "coarse").dir
    (coarse / "JOB.XV").write_text("geometry from coarse\n")

    warm = prepare_attempt(js, tmp_path, "tight",
                           continue_from="01_coarse/run-0")
    attempt = warm.dir
    assert warm.copied == ["JOB.XV"]
    assert (attempt / "JOB.XV").is_file()

    cold = prepare_attempt(js, tmp_path, "tight", cold=True)
    assert cold.dir == attempt                # the same unlaunched attempt
    assert not (attempt / "JOB.XV").exists()     # and it is actually cold now
    assert not (attempt / ".continued-from").exists()


def test_status_takes_a_stage_and_answers_the_other_question(tmp_path):
    """`job-system.md` § 5.3 reserves a per-stage form and marked it unbuilt.

    The table answers *where is this calculation up to*; this answers *what
    happened to this stage*, which is what you ask before deciding to run it
    again.  It is only answerable because a try is a directory and a launch is
    a record (§ 1.5, § 1.6) -- so it prints the attempt, the launch and the
    provenance, not just the row.
    """
    from molbuilder.jobset.materialize import prepare_attempt, write_run_launch
    from molbuilder.jobset.runstatus import render_stage_status
    js = _token_ladder("JOB_01_coarse.fdf", "JOB_03_tight.fdf")
    coarse = prepare_attempt(js, tmp_path, "coarse").dir
    (coarse / "JOB.XV").write_text("COARSE-GEOM")
    tight = prepare_attempt(js, tmp_path, "tight",
                            continue_from="01_coarse/run-0").dir
    write_run_launch(tight, mode="submit", command=["sbatch", "x.sbatch"],
                     job_id="481923", continued_from="01_coarse/run-0")

    out = render_stage_status(jobset_status(js, tmp_path), "tight")
    assert out.splitlines()[0].startswith("STAGE 03_tight")
    assert "run-0" in out
    assert "481923" in out                       # the launch record, not a guess
    assert "01_coarse/run-0" in out              # where this geometry came from
    assert "03_tight/run-0" in out               # and where to go look


def test_a_never_launched_stage_says_so_instead_of_showing_a_blank_record(tmp_path):
    """Prepared but not started is its own state, and it is what `run.json`'s
    absence means (§ 1.6)."""
    from molbuilder.jobset.materialize import prepare_attempt
    from molbuilder.jobset.runstatus import render_stage_status
    js = _token_ladder("JOB_03_tight.fdf")
    prepare_attempt(js, tmp_path, "tight")

    out = render_stage_status(jobset_status(js, tmp_path), "tight")
    assert "no run.json" in out
    assert "continued from" not in out


def test_a_cold_run_prints_no_provenance_line_at_all(tmp_path):
    """`continued_from` is ABSENT, not null, when a run starts from the
    structure (checkpointing.md S3) -- and the view must not turn that absence
    into *"continued from: nothing"*, which is a different claim.

    This is the LAUNCHED-but-cold case.  Testing it on a never-launched stage
    proves nothing: that path stops before provenance is ever considered, so a
    view that printed a blank line for every cold run would still pass.
    """
    from molbuilder.jobset.materialize import prepare_attempt, write_run_launch
    from molbuilder.jobset.runstatus import render_stage_status
    js = _token_ladder("JOB_03_tight.fdf")
    attempt = prepare_attempt(js, tmp_path, "tight", cold=True).dir
    write_run_launch(attempt, mode="direct", command=["bash", "x.sh"])

    out = render_stage_status(jobset_status(js, tmp_path), "tight")
    assert "launched" in out and "direct" in out      # it DID start
    assert "continued from" not in out                # from the structure


def test_every_label_in_the_per_stage_view_is_padded_off_the_longest(tmp_path):
    """The pad was hand-written as 14 -- exactly the width of `continued from`,
    so the one row with provenance to report ran its value into its own name.
    Same defect as the table's two column counts, one screen over."""
    from molbuilder.jobset.materialize import prepare_attempt, write_run_launch
    from molbuilder.jobset.runstatus import render_stage_status
    js = _token_ladder("JOB_01_coarse.fdf", "JOB_03_tight.fdf")
    coarse = prepare_attempt(js, tmp_path, "coarse").dir
    (coarse / "JOB.XV").write_text("x")
    tight = prepare_attempt(js, tmp_path, "tight",
                            continue_from="01_coarse/run-0").dir
    write_run_launch(tight, mode="direct", command=["bash", "x.sh"],
                     continued_from="01_coarse/run-0")

    body = [l for l in render_stage_status(jobset_status(js, tmp_path),
                                           "tight").splitlines()
            if l.startswith("  ")]
    # every indented row separates its label from its value by real whitespace
    assert body, "no rows rendered"
    for line in body:
        assert re.match(r"^ {2}\S.*?\s{2,}\S", line), f"label runs into value: {line!r}"


def test_status_takes_the_bundle_the_way_every_verb_does(tmp_path):
    """One word cannot mean the folder on two verbs and the stage on two others.
    `jobset status tight` answered *"Directory 'tight' does not exist"* -- a
    complaint about a path the user never meant to type (§ 5.3)."""
    _token_ladder("JOB_01_coarse.fdf", "JOB_03_tight.fdf").write(
        tmp_path / "job-set.json")
    runner, grp = _runner()
    r = runner.invoke(grp, ["status", "--bundle", str(tmp_path)])
    assert r.exit_code == 0, r.output
    assert "coarse" in r.output
    # ...and the positional is a STAGE, resolved the way every other verb
    # resolves one.  A NUMBER, deliberately: an exact name would pass even if
    # the command took the string verbatim and never reached the resolver.
    r = runner.invoke(grp, ["status", "#3", "--bundle", str(tmp_path)])
    assert r.exit_code == 0, r.output
    # The CONTENT property (#3 resolved to 03_tight), not its line position --
    # pinning splitlines()[0] made any banner a false failure (2026-08-12).
    assert "STAGE 03_tight" in r.output


def test_a_ladder_refuses_to_act_on_all_of_itself(tmp_path):
    """`project-layout.md` § 1.6 -- *"Each stage is prepped and submitted on
    its own"* -- and the reason is cost, not tidiness: *"a chain that continues
    on its own can spend a week refining a geometry you would have rejected in
    a minute."*

    **The refusal now has no escape hatch**, and that is the change of
    **The refusal is the end of the road**, so its message offers only the
    one-stage form.  It is also why a ladder can never reach `_run_direct` with
    more than one job, and therefore why no second stage can continue from a
    failed first.
    """
    _token_ladder("JOB_01_coarse.fdf", "JOB_03_tight.fdf").write(
        tmp_path / "job-set.json")
    runner, grp = _runner()

    r = runner.invoke(grp, ["launch", "run", "--bundle", str(tmp_path),
                            "--mode", "direct", "--dry-run", "--yes"])
    assert r.exit_code != 0
    assert "acts on ONE stage" in r.output
    # what you can TYPE, ordinals beside -- the token is refused when typed
    # back (`job-system.md` § 5.3, plan § 5w K12)
    assert "coarse ('#1'), tight ('#3')" in r.output
    # Every piece of advice it prints must be a command that works -- the
    # option set is pinned by test_submit_accepts_exactly_these_options.
    # Scanned over the whole output, not per line: option tokens are single
    # words, so line-splitting added an assumption without adding a check.
    for word in re.findall(r"--[a-z][a-z-]*", r.output):
        assert word in {"--bundle", "--mode", "--domain", "--dry-run"}, (
            f"the refusal advertises {word}, which submit does not accept")


def test_what_a_run_continues_from_is_copied_never_linked(tmp_path):
    """§ 1.6: *"they are **copied, never linked** -- the engine writes to those
    very filenames, and writing through a link would destroy the result you
    started from."*

    ``is_file()`` is true for a symlink that resolves, so the only honest check
    is to WRITE, the way the engine will, and look at what the producer still
    holds afterwards. This is the difference between carrying a geometry
    forward and overwriting the one you chose it from.
    """
    from molbuilder.jobset.materialize import prepare_attempt
    js = _token_ladder("JOB_01_coarse.fdf", "JOB_03_tight.fdf")
    coarse = prepare_attempt(js, tmp_path, "coarse").dir
    (coarse / "JOB.XV").write_text("COARSE-GEOM")

    tight = prepare_attempt(js, tmp_path, "tight",
                            continue_from="01_coarse/run-0").dir
    carried = tight / "JOB.XV"
    assert not carried.is_symlink(), "carried warm state is a LINK back to it"
    assert carried.read_text() == "COARSE-GEOM"

    carried.write_text("TIGHT-GEOM")            # what the engine does, step 1
    assert (coarse / "JOB.XV").read_text() == "COARSE-GEOM"


# --------------------------------------------------------------------- #
#  What a run continues from is decided by the PAIR (P6 unit 3)          #
#                                                                        #
#  `project-layout.md` § 2.3.4 states it as three rows, and only the     #
#  third needs two stages:                                               #
#    .XV  always | .DM  when the description says | .CG  ONLY if both    #
#    stages use the same algorithm                                       #
# --------------------------------------------------------------------- #

def _shipped_ladder(coarse_restart=None):
    """The ladder a user actually gets — coarse (CG), medium + tight (Broyden).

    Built through the real producer rather than by hand, because the claim
    under test is about the SHIPPED science: the tiers really do change
    optimizer between rung one and rung two, which is what makes `.CG` a live
    question rather than a hypothetical one.
    """
    # Built the way the LIVE path builds it (`prep._job_for`, u5): each
    # enabled stage resolved through the one seam, warm + traits from the
    # engine's own declarations.
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.identity import stage_token
    from molbuilder.resolve import effective_config
    from molbuilder.siesta.stages import (_traits, _warm_declaration,
                                          default_siesta_stages)
    label = "bdt"
    jobs = []
    for i, st in enumerate(default_siesta_stages("vib-quality"), start=1):
        if not st.enabled:
            continue
        overrides = dict(getattr(st, "overrides", None) or {})
        if coarse_restart is not None and i == 1:
            overrides["restart"] = coarse_restart
        eff = effective_config(SiestaConfig(system_label=label),
                               overrides,
                               where=f"stage {st.name!r}")
        jobs.append(Job(name=st.name,
                        script=f"{label}_{stage_token(i, st.name)}.fdf",
                        resources=Resources(),
                        warm=_warm_declaration(label, eff),
                        traits=_traits(eff)))
    js = JobSet(name=label, engine="siesta", kind="ladder", jobs=jobs)
    # lowercase since R11: the trait normalizes at its ONE producer so
    # "Broyden" vs "broyden" (one optimizer, two hands) compare equal
    assert [j.traits["optimizer"] for j in js.jobs] == \
        ["cg", "broyden", "broyden"], "the fixture's premise moved"
    return js


def _finished(base, stage_dir, label="bdt"):
    """An attempt that has run: its three warm files, with tellable contents."""
    d = base / stage_dir / "run-0"
    d.mkdir(parents=True)
    for ext in ("XV", "DM", "CG"):
        (d / f"{label}.{ext}").write_text(f"{stage_dir}:{ext}")
    return d


def test_the_optimizer_rule_is_asked_of_the_pair_not_of_the_ladders_neighbour(
        tmp_path):
    """`job-system.md` § 4.1: `.CG` is carried *"only when consecutive stages
    use the same relaxation method — a CG state is meaningless to a Broyden
    stage, so blindly carrying it would corrupt the restart."*

    **The stage you continue from is not always the one before you.** `--from`
    names any finished attempt (§ 1.6), so continuing `tight` from `01_coarse`
    skips `medium` entirely — and coarse relaxes with **CG** while tight uses
    **Broyden**.

    Until 2026-08-10 the set came off `Job.carry`, whose `from_job` is the
    immediate predecessor and is fixed at produce time. `tight.carry` compares
    tight against *medium* (Broyden vs Broyden → carry it), so this prep copied
    a CG optimizer history into a Broyden stage on the strength of a comparison
    with a stage that never ran.
    """
    from molbuilder.jobset.materialize import prepare_attempt
    js = _shipped_ladder()
    _finished(tmp_path, "01_coarse")

    rep = prepare_attempt(js, tmp_path, "tight",
                          continue_from="01_coarse/run-0")
    assert rep.copied == ["bdt.XV", "bdt.DM"]
    assert not (rep.dir / "bdt.CG").exists(), (
        "a CG history reached a Broyden stage -- the restart it corrupts "
        "still reports success")


def test_and_the_same_prep_does_carry_it_when_the_two_agree(tmp_path):
    """The other half, and it is not decoration: a system that carried `.CG`
    **never** would pass the test above while quietly throwing away the
    optimizer history every real continuation depends on.

    `medium` and `tight` are both Broyden, so this is the case the rule
    permits — and it is the ordinary one, since a ladder normally continues
    from the rung below it.
    """
    from molbuilder.jobset.materialize import prepare_attempt
    js = _shipped_ladder()
    _finished(tmp_path, "02_medium")

    rep = prepare_attempt(js, tmp_path, "tight",
                          continue_from="02_medium/run-0")
    assert rep.copied == ["bdt.XV", "bdt.DM", "bdt.CG"]
    assert (rep.dir / "bdt.CG").read_text() == "02_medium:CG"


def test_a_redo_of_one_stage_agrees_with_itself(tmp_path):
    """§ 2.3.4: *"A redo is the same instruction."* `--from` inside the SAME
    stage re-runs it from where the last attempt reached, and a stage always
    matches its own optimizer — so the full group comes across, including the
    history the rule exists to protect."""
    from molbuilder.jobset.materialize import prepare_attempt, write_run_launch
    js = _shipped_ladder()
    first = _finished(tmp_path, "03_tight")
    write_run_launch(first, mode="direct", command=["bash", "x"])

    rep = prepare_attempt(js, tmp_path, "tight",
                          continue_from="03_tight/run-0")
    assert rep.dir.name == "run-1"            # the launched one is immutable
    assert rep.copied == ["bdt.XV", "bdt.DM", "bdt.CG"]


def test_a_source_this_jobset_cannot_place_withholds_the_conditional_file(
        tmp_path):
    """Unverified is not the same as satisfied, and the mistake is not
    symmetric: a `.CG` wrongly withheld costs some optimizer steps, while one
    wrongly carried corrupts the restart and the run still reports success.

    So a `--from` naming a directory this JobSet has no job for — a hand-made
    path, a stage disabled since the bundle was produced — keeps the
    unconditional files and drops the conditional one.
    """
    from molbuilder.jobset.materialize import prepare_attempt
    js = _shipped_ladder()
    d = tmp_path / "99_elsewhere" / "run-0"
    d.mkdir(parents=True)
    for ext in ("XV", "DM", "CG"):
        (d / f"bdt.{ext}").write_text(ext)

    rep = prepare_attempt(js, tmp_path, "tight",
                          continue_from="99_elsewhere/run-0")
    assert rep.copied == ["bdt.XV", "bdt.DM"]


def test_a_clean_stage_refuses_from_instead_of_copying_nothing(tmp_path):
    """`run-identity.md` § 4's silent pair — *present but not honoured*: the
    files are right there, the parameter is off, and the stage starts from
    scratch looking like it continued.

    A `restart: clean` stage's deck writes `MD.UseSaveXV` / `DM.UseSaveDM` /
    `MD.UseSaveCG` as `.false.` (the same `restart` field decides the deck and
    the declaration), so anything copied in would sit unread. Copying it
    anyway and reporting success is the failure; the refusal is the fix.

    *(This said the deck OMITS them until 2026-08-18. It did, and that was the
    bug: SIESTA reads the files when they are present unless a deck says
    `.false.`, so "clean" said nothing and the stage continued. The refusal
    tested here was right either way -- it reads the WARM DECLARATION, which
    has always been empty for a clean stage.)*
    """
    from molbuilder.jobset.materialize import prepare_attempt
    # The rung is made clean EXPLICITLY, which is the only way a stage is
    # clean since 2026-08-18: `continue` is the default, and `clean` is a
    # person overriding it (`run-identity.md` § 4 rule 3).  The shipped ladder
    # used to splice `clean` into rung one positionally, which is what this
    # test used to lean on.
    js = _shipped_ladder(coarse_restart="clean")
    _finished(tmp_path, "01_coarse")

    with pytest.raises(ValueError) as e:
        prepare_attempt(js, tmp_path, "coarse",
                        continue_from="01_coarse/run-0")
    assert "declares no warm-restart files" in str(e.value)
    assert "restart" in str(e.value)             # and what to change


def test_the_declaration_is_now_the_only_rendering_of_the_rule():
    """The warm declaration is the one rendering of the warm-start rule.
    -- the plan's item 12c's *"two lists that agree today and
    nothing keeps them agreeing"*, held in step by derivation.

    P7 unit 2 deleted the second rendering, which is the better end state: a
    guard against drift is only needed while there are two things to drift.
    What is left to assert is that the projection really is gone, so nobody
    reintroduces it as a convenience.
    """
    js = _shipped_ladder()
    assert all(j.warm for j in js.jobs[1:])      # the rule travels as `warm`


def test_validate_refuses_a_condition_the_job_could_never_meet():
    """A `requires_same` naming a trait the job does not declare can never be
    satisfied, so the file would simply never be carried — the wrong kind of
    silence, because "fail safe" here means starting cold while everything
    reports success. The comparison stays fail-safe; the DECLARATION is a
    producer bug and is refused by name."""
    from molbuilder.jobset.model import Job, JobSet, WarmFile
    js = JobSet(name="JOB", engine="siesta", kind="ladder",
                jobs=[Job(name="s1", script="JOB_01_s1.fdf",
                          warm=[WarmFile("JOB.CG", requires_same="optimizer")])])
    errs = js.validate()
    assert len(errs) == 1
    assert "optimizer" in errs[0] and "traits" in errs[0]


def test_warm_and_traits_survive_job_set_at_1():
    """`prep` runs on the target, from the persisted bundle — a declaration
    that does not round-trip is one the machine that runs the job never sees."""
    from molbuilder.jobset.model import JobSet
    js = _shipped_ladder()
    back = JobSet.from_dict(js.to_dict())
    for a, b in zip(js.jobs, back.jobs):
        assert a.traits == b.traits
        assert [(w.name, w.requires_same) for w in a.warm] == \
               [(w.name, w.requires_same) for w in b.warm]
    # ABSENT, not null, for an unconditional file (checkpointing.md S3).
    xv = js.to_dict()["jobs"][1]["warm"][0]
    assert xv == {"name": "bdt.XV"}


def test_re_prep_sweeps_the_whole_declared_set_not_the_pair_filtered_one(
        tmp_path):
    """Changing your mind from `--from 02_medium` to `--from 01_coarse` must
    not leave medium's `.CG` behind: coarse would not have carried it, but the
    file is there and SIESTA reads what it finds.

    So the undo sweeps everything this stage DECLARES, not what this prep would
    have copied — the previous prep may have named a different source.
    """
    from molbuilder.jobset.materialize import prepare_attempt
    js = _shipped_ladder()
    _finished(tmp_path, "01_coarse")
    _finished(tmp_path, "02_medium")

    warm = prepare_attempt(js, tmp_path, "tight",
                           continue_from="02_medium/run-0")
    assert (warm.dir / "bdt.CG").is_file()

    again = prepare_attempt(js, tmp_path, "tight",
                            continue_from="01_coarse/run-0")
    assert again.dir == warm.dir           # same unlaunched attempt
    assert not (again.dir / "bdt.CG").exists()
    assert (again.dir / "bdt.XV").read_text() == "01_coarse:XV"


def test_attempts_are_ordered_as_numbers_not_as_names(tmp_path):
    """`run-10` comes after `run-9`, and lexically it does not.

    Nothing reads these back as strings today, and that is the point of pinning
    it: sorting by name makes `resolve_attempt` hand out `run-3` when `run-10`
    already exists, so the next prep writes into a directory that has already
    run -- § 1.5's one prohibition, reached by a sort order.
    """
    from molbuilder.jobset.materialize import (attempts, latest_attempt,
                                               resolve_attempt)
    d = tmp_path / "03_tight"
    for n in (0, 1, 2, 9, 10):
        (d / f"run-{n}").mkdir(parents=True)
        (d / f"run-{n}" / "run.json").write_text("{}")   # all launched
    (d / "notes.txt").write_text("")                     # not an attempt
    (d / "run-x").mkdir()                                # nor is this

    assert attempts(d) == [0, 1, 2, 9, 10]
    assert latest_attempt(d).name == "run-10"
    assert resolve_attempt(d) == (d / "run-11", True)


def test_prepare_links_resolve_from_two_levels_down(tmp_path):
    """The deck and the package arrive in ``<stage>/run-<n>/`` as REAL
    COPIES (L2, roadmap 7.10; they were relative links until the layout
    repair, and a synced-back bundle's links dangled on the other
    machine)."""
    from molbuilder.jobset.materialize import prepare_attempt
    from molbuilder.jobset.model import Job, JobSet
    js = JobSet(name="JOB", engine="siesta", kind="ladder",
                shared=["C.psml"], jobs=[Job(name="tight",
                                             script="JOB_03_tight.fdf")])
    from molbuilder.runwrap import MONITOR_BUNDLE
    for f in ("JOB_03_tight.fdf", "C.psml", MONITOR_BUNDLE,
              "JOB_03_tight.run.sh"):
        (tmp_path / f).write_text("x")

    rep = prepare_attempt(js, tmp_path, "tight")
    attempt = rep.dir
    # THE MONITOR'S ONE FILE, by runwrap's own name for it.  `config_dir.py`
    # was once named in the wrapper writer and not here, so it travelled with
    # bench trials (rendered in place) and not with run attempts (linked)
    # -- and every production run's monitor died at import, silently
    # (2026-08-28).  One file cannot be half-brought.
    assert set(rep.brought) == {"JOB_03_tight.fdf", "C.psml", MONITOR_BUNDLE,
                                  "JOB_03_tight.run.sh"}
    for name in rep.brought:
        link = attempt / name
        assert link.is_file() and not link.is_symlink(), (
            f"{name}: a run directory holds real files (L2, roadmap "
            f"7.10) -- this asserted symlinks until the layout repair")
        assert link.read_text() == (tmp_path / name).read_text(), (
            f"{name}: the copy differs from its source")


def test_the_grammar_is_unambiguous_even_for_a_stage_named_3(tmp_path):
    """Stage names are ``[A-Za-z0-9_]+``, so a stage may legitimately be
    named ``3`` -- which is exactly why the bare-number spelling died
    (user-settled 2026-08-21): ``3`` is ONLY ever the name, ``#3`` is ONLY
    ever the assigned number, and the two can never collide because ``#``
    cannot appear in a name."""
    import pytest as _pt
    from molbuilder.identity import StageRef, resolve_stage_ref
    refs = [StageRef(1, "3"), StageRef(3, "tight")]
    assert resolve_stage_ref(refs, "3").name == "3"      # the NAME, seq 1
    assert resolve_stage_ref(refs, "#3").name == "tight"  # the NUMBER
    assert resolve_stage_ref(refs, "#1").name == "3"
    assert resolve_stage_ref(refs, "tight").seq == 3
    with _pt.raises(ValueError, match="no stage named"):
        resolve_stage_ref(refs, "03_tight")              # tokens retired


def test_run_launch_omits_continued_from_rather_than_writing_null(tmp_path):
    """`checkpointing.md` S3 words its check as *"names a directory that exists
    **or is absent**"*, and absent is not `null`: a reader that tests for the
    key sees a starting-from-the-structure run as one that continued from
    nothing-in-particular.  Two different claims, one of them false."""
    import json
    from molbuilder.jobset.materialize import (RUN_LAUNCH_SCHEMA,
                                               write_run_launch)
    p = write_run_launch(tmp_path, mode="direct", command=["bash", "x.sh"])
    body = json.loads(p.read_text())
    assert body["schema"] == RUN_LAUNCH_SCHEMA
    assert "continued_from" not in body          # ABSENT, not None

    p = write_run_launch(tmp_path, mode="direct", command=["bash", "x.sh"],
                         continued_from="01_coarse/run-0")
    assert json.loads(p.read_text())["continued_from"] == "01_coarse/run-0"


def test_the_provenance_survives_the_prep_to_submit_handover(tmp_path):
    """§ 1.6, *"How `continued_from` reaches it"*: prep is what knows which
    attempt this one continues from, submit is what writes `run.json`, and a
    private marker carries it across.  That seam has no other reader, so if it
    breaks nothing complains -- the record just quietly says a run started from
    the structure when it started from a geometry you chose."""
    import json
    from molbuilder.jobset.materialize import prepare_attempt
    from molbuilder.jobset.submit import submit_jobset
    js = _token_ladder("JOB_01_coarse.fdf", "JOB_03_tight.fdf")
    coarse = prepare_attempt(js, tmp_path, "coarse").dir
    (coarse / "JOB.XV").write_text("geometry from coarse\n")

    tight = prepare_attempt(js, tmp_path, "tight",
                            continue_from="01_coarse/run-0").dir
    # the wrapper prep would have linked in; submit only launches
    (tight / "JOB_03_tight.run.sh").write_text("#!/bin/bash\nexit 0\n")
    submit_jobset(js, tmp_path, mode="direct", only="tight")

    body = json.loads((tight / "run.json").read_text())
    assert body["continued_from"] == "01_coarse/run-0"
    assert body["mode"] == "direct"


def test_prepare_attempt_refuses_a_from_that_has_not_run(tmp_path):
    """*"Did it run?"* -- an attempt directory that exists but holds none of the
    warm files is a live mistake (naming the attempt you are ABOUT to run, or a
    stage that failed before writing).  Copying nothing and reporting success
    would start it cold while the user believed it continued."""
    from molbuilder.jobset.materialize import prepare_attempt
    js = _token_ladder("JOB_01_coarse.fdf", "JOB_03_tight.fdf")
    prepare_attempt(js, tmp_path, "coarse")      # exists, but produced nothing

    with pytest.raises(ValueError) as e:
        prepare_attempt(js, tmp_path, "tight", continue_from="01_coarse/run-0")
    assert "Did it run?" in str(e.value)

    with pytest.raises(ValueError) as e:
        prepare_attempt(js, tmp_path, "tight", continue_from="01_coarse/run-9")
    assert "no such attempt" in str(e.value)


def test_a_corrupt_run_json_still_reads_as_launched(tmp_path):
    """The file's PRESENCE is the answer to *has this been launched?* (§ 1.6).
    Its contents are extra, so a truncated write must not demote the stage to
    'never started' -- which would invite a submit on top of a running job."""
    from molbuilder.jobset.materialize import prepare_attempt
    js = _token_ladder("JOB_03_tight.fdf")
    attempt = prepare_attempt(js, tmp_path, "tight").dir
    (attempt / "run.json").write_text('{"schema": "molbuilder/run-la')

    st = jobset_status(js, tmp_path).stages[0]
    assert st.state == "queued"                  # launched, details lost


def test_submit_only_takes_the_same_two_spellings_as_every_surface(tmp_path):
    """`only` is a library entry point, and it had its own lookup and its own
    listing -- the same defect prepare_attempt had (§ 8f).  The grammar is
    the one resolver's: name, or #N (user-settled 2026-08-21)."""
    from molbuilder.jobset.submit import submit_jobset, SubmitError
    js = _token_ladder("JOB_01_coarse.fdf", "JOB_03_tight.fdf")
    res = submit_jobset(js, tmp_path, mode="direct", dry_run=True, only="#3")
    assert [r.name for r in res] == ["tight"]

    with pytest.raises(SubmitError) as e:
        submit_jobset(js, tmp_path, mode="direct", dry_run=True, only="bogus")
    assert "coarse ('#1'), tight ('#3')" in str(e.value)


def test_a_number_resolves_to_the_seq_and_never_to_the_row():
    """R5's whole point, and the two differ the moment a stage is disabled:
    the ladder is 01/03, so `#1` is coarse (row 1 would be tight) and `#3`
    is tight (there is no row 3 at all)."""
    from molbuilder.identity import StageRef, resolve_stage_ref
    refs = [StageRef(1, "coarse"), StageRef(3, "tight")]
    assert resolve_stage_ref(refs, "#1").name == "coarse"
    assert resolve_stage_ref(refs, "#3").name == "tight"
    assert resolve_stage_ref(refs, "tight").name == "tight"     # its identity
    with pytest.raises(ValueError):
        resolve_stage_ref(refs, "#2")           # the row of tight; not its seq


def test_resolver_refuses_a_number_a_sweep_cannot_have():
    """One resolver serves both kinds, so the refusal must stop offering
    ordinals to a job-set that has none."""
    from molbuilder.identity import StageRef, resolve_stage_ref
    refs = [StageRef(None, "p1"), StageRef(None, "p2")]
    assert resolve_stage_ref(refs, "p2").name == "p2"
    with pytest.raises(ValueError) as e:
        resolve_stage_ref(refs, "2")
    assert "p1, p2" in str(e.value) and "number" not in str(e.value)


def test_plan_prints_the_seq_not_the_row():
    """The `#` column was `enumerate()` -- a POSITION where a reader reads the
    ordinal, which is the number R5 forbids as an identifier."""
    js = _token_ladder("JOB_01_coarse.fdf", "JOB_03_tight.fdf")
    out = render_plan(js)
    assert "seq" in out.splitlines()[3]
    body = [l for l in out.splitlines() if "coarse" in l or "tight" in l]
    assert body[0].split()[0] == "1"
    assert body[1].split()[0] == "3"          # NOT "1", which the row would be


def test_status_prints_the_seq_not_the_row(tmp_path):
    """The status table's number column prints the stage's ASSIGNED seq, not its row
    position.

    The sibling of `test_plan_prints_the_seq_not_the_row`, in the surface a person
    reads while a run is going. A ladder whose stages are 01 and 03 must show 1
    and 3; `enumerate()` shows 1 and 2, and the reader then names stage "2" to
    `--only` -- which is either a different stage or nothing at all.
    `project-layout.md` § 4.2: the number is assigned once and never
    guessed.
    """
    js = _token_ladder("JOB_01_coarse.fdf", "JOB_03_tight.fdf")
    st = jobset_status(js, tmp_path)
    assert [s.seq for s in st.stages] == [1, 3]
    out = render_status(st)
    assert "seq" in out
    assert [l.split()[0] for l in out.splitlines() if "tight" in l] == ["3"]


def test_a_launched_sweep_trial_reports_queued_not_pending(tmp_path):
    """`project-layout.md` § 1.6: `run.json` is the honest answer to "has
    this been launched?" -- a queued job has produced nothing, so "no
    output" and "never started" look identical without it.  A sweep trial
    is ITS OWN attempt (submit.py's rule), so its record sits at the
    trial's top, not in a `run-<n>`/.  Until 2026-08-20 the status reader
    looked only inside attempts, and a grouped-submitted trial answered
    the exact false line § 1.6 forbids."""
    from molbuilder.jobset.materialize import (job_dir_names, shape_of,
                                               write_run_launch)
    from molbuilder.jobset.model import Job, JobSet
    js = JobSet(name="JOB", engine="siesta", kind="sweep",
                jobs=[Job(name="p1", script="JOB_p1.fdf"),
                      Job(name="p2", script="JOB_p2.fdf")])
    dirs = job_dir_names(js, shape_of(js, tmp_path))
    for name in dirs.values():
        (tmp_path / name).mkdir(parents=True)
    # p1 rides a grouped submission; p2 was never launched.
    write_run_launch(tmp_path / dirs["p1"], mode="submit",
                     command=["sbatch", "bench-group.sbatch"],
                     job_id="48213")
    st = jobset_status(js, tmp_path)
    by = {s_.ref.name: s_ for s_ in st.stages}
    assert by["p1"].state == "queued", by["p1"]
    assert "queued as job 48213" in by["p1"].detail
    assert by["p1"].launch and by["p1"].launch.get("job_id") == "48213"
    assert by["p2"].state == "pending"


def test_status_seq_is_none_for_a_sweep_point():
    """A sweep point has no order, so it has no seq -- and says so rather than
    borrowing a row number's authority."""
    from molbuilder.jobset.model import Job, JobSet
    js = JobSet(name="JOB", engine="siesta", kind="sweep",
                jobs=[Job(name="p1", script="JOB.fdf")])
    # tmp-independent: the set is hand-built and never touches disk,
    # but "." as a base path reads whatever cwd the RUNNER happens to
    # be in -- an accidental dependence on found state (G2 I-list,
    # 2026-08-12).  A path that cannot exist keeps the call honest.
    assert jobset_status(js, "/nonexistent-base").stages[0].seq is None


# --------------------------------------------------------------------- #
#  Shape — where a stage's files live (§ 9's object, P5)                 #
# --------------------------------------------------------------------- #


def test_shape_refuses_anything_that_is_not_one_of_the_two():
    """`engines/stages.md` § 6.7: required, never inferred.  A constructor that
    accepted a third word would let a typo become a layout."""
    from molbuilder.paths import Shape
    assert Shape.named("flat").name == "flat"
    assert Shape.named("hierarchical").name == "hierarchical"
    with pytest.raises(ValueError) as e:
        Shape.named("nested")
    assert "flat" in str(e.value) and "hierarchical" in str(e.value)


def test_hierarchical_tells_stages_apart_by_PATH_and_flat_by_NAME():
    """`project-layout.md` § 1, the one difference everything else follows
    from.  Hierarchical gives each stage a directory, so anything inside it
    belongs to it.  Flat is DEPTH 1 -- one directory holds every stage, and the
    deck's token in each filename is what selects one.

    A layer that only asks *which directory* is right in the hierarchy and
    silently wrong in flat: it answers about every stage at once.
    """
    from molbuilder.paths import Shape
    hier, flat = Shape.named("hierarchical"), Shape.named("flat")

    assert hier.stage_dir("03_tight") == "03_tight"
    assert hier.stage_glob("03_tight", "JOB") == "*"      # the dir already chose

    assert flat.stage_dir("03_tight") == "."             # a joinable path, not None
    assert flat.stage_glob("03_tight", "JOB") == "JOB_03_tight*"


def test_only_the_hierarchy_keeps_attempts_as_directories():
    """§ 1: flat separates attempts by an OUTPUT INDEX (`-run0.out`) the
    wrapper writes, so there is no attempt directory to open and `--from` has
    nothing to name -- *"continuing: free, the next stage finds them lying
    there."*"""
    from molbuilder.paths import Shape
    assert Shape.named("hierarchical").keeps_attempts_as_directories is True
    assert Shape.named("flat").keeps_attempts_as_directories is False


def test_a_flat_ladder_lays_every_stage_out_in_the_bundle_root():
    """Depth 1.  This is what makes a flat `job-set.json` safe to emit: without
    it the bundle's own prep would build `01_coarse/`, `02_medium/` inside a
    calculation whose description says flat."""
    from molbuilder.jobset.materialize import job_dir_names
    from molbuilder.paths import Shape
    js = _token_ladder("JOB_01_coarse.fdf", "JOB_03_tight.fdf")

    assert job_dir_names(js, Shape.named("flat")) == {"coarse": ".",
                                                      "tight": "."}
    assert job_dir_names(js, Shape.named("hierarchical")) == {
        "coarse": "01_coarse", "tight": "03_tight"}


def test_a_sweep_is_laid_out_the_same_way_in_either_shape():
    """``bench/bench-<name>`` is the benchmark's own convention and says
    nothing about flat or hierarchical -- a TOKENLESS trial has no stage
    directory to nest in, so the bare container is the same in both shapes
    (which is why a bench bundle needs no description to be laid out)."""
    from molbuilder.jobset.materialize import job_dir_names
    from molbuilder.jobset.model import Job, JobSet
    from molbuilder.paths import Shape
    js = JobSet(name="JOB", engine="siesta", kind="sweep",
                jobs=[Job(name="p1", script="JOB_p1.fdf")])
    assert (job_dir_names(js, Shape.named("flat"))
            == job_dir_names(js, Shape.named("hierarchical"))
            == {"p1": "bench/bench-p1"})


def _describe(base, shape, names=("coarse", "tight")):
    """Write a real `task.json` beside a bundle, through the one codec."""
    from molbuilder.task import (FILENAME, Stage, StructureRef, Task,
                                 derive_run, write_task)
    stages = tuple(Stage(name=n, overrides={"mesh_cutoff": 200}) for n in names)
    write_task(Path(base) / FILENAME, Task(
        engine="siesta", shape=shape,
        run=derive_run("JOB", "H2", stage_names=names),
        structure=StructureRef(source="h2.xyz", formula="H2", atoms=2),
        varies=("mesh_cutoff",), stages=stages))


def test_the_surfaces_read_the_shape_from_the_description(tmp_path):
    """`engines/stages.md` § 6.7: *"`prep` **reads** it; it does not decide
    it."*  `shape_of` is the one place a surface asks, and the layers below
    take the answer as an argument rather than going looking a second time.

    Pinned through `job_dir_names` on a REAL bundle, because the object being
    correct proves nothing if nobody hands it the description's answer -- which
    is what a mutation found: `shape_of` returning a constant left every test
    green.
    """
    from molbuilder.jobset.materialize import job_dir_names, shape_of
    js = _token_ladder("JOB_01_coarse.fdf", "JOB_03_tight.fdf")

    _describe(tmp_path, "flat")
    assert shape_of(js, tmp_path).name == "flat"
    assert job_dir_names(js, shape_of(js, tmp_path)) == {"coarse": ".",
                                                          "tight": "."}

    _describe(tmp_path, "hierarchical")
    assert shape_of(js, tmp_path).name == "hierarchical"
    assert job_dir_names(js, shape_of(js, tmp_path)) == {
        "coarse": "01_coarse", "tight": "03_tight"}


def test_a_sweep_asks_no_description_for_its_shape(tmp_path):
    """`bench-<name>` is the benchmark's convention in either layout, which is
    why a bench bundle carries no `task.json` and needs none."""
    from molbuilder.jobset.materialize import shape_of
    from molbuilder.jobset.model import Job, JobSet
    js = JobSet(name="JOB", engine="siesta", kind="sweep",
                jobs=[Job(name="p1", script="JOB_p1.fdf")])
    assert shape_of(js, tmp_path) is None          # no description, no problem


def test_prepare_attempt_refuses_a_flat_calculation(tmp_path):
    """Flat has no attempt directories (`project-layout.md` § 1): attempts are
    the wrapper's output index, the warm files are one shared set, and
    continuing is free.  So there is nothing to open and `--from` has nothing
    to name -- and opening `run-0/` in the bundle root would invent a layer the
    description did not ask for."""
    from molbuilder.jobset.materialize import prepare_attempt
    js = _token_ladder("JOB_01_coarse.fdf", "JOB_03_tight.fdf")
    _describe(tmp_path, "flat")

    with pytest.raises(ValueError) as e:
        prepare_attempt(js, tmp_path, "tight")
    assert "flat" in str(e.value) and "output index" in str(e.value)
    assert not (tmp_path / "03_tight").exists()    # and nothing was created

    _describe(tmp_path, "hierarchical")            # ...the hierarchy still does
    assert prepare_attempt(js, tmp_path, "tight").dir.name == "run-0"


@pytest.mark.parametrize("shape", ["flat", "hierarchical"])
def test_prep_leaves_every_job_a_readable_deck_and_wrapper(tmp_path, shape):
    """`job-system.md` decision #2: *"Each job in a JobSet is launched by
    exactly the `.run.sh` / `.sbatch` wrapper … built by the same function."*
    A job whose deck or wrapper is a **dangling symlink** is launched by
    nothing.

    This is M5 pass 1's finding, and it was severe: in the flat shape a job's
    directory IS the bundle root, so `relink(d, "../<name>", …)` unlinked the
    real file and pointed at the bundle's PARENT.  A flat prep destroyed its
    own decks, its wrappers and the monitor — every one of them.

    It was invisible to the check I ran at the time, which asked *"does prep
    make the right directories?"* (flat: none, correct) and never asked whether
    the files survived.  So the assertion here is about **what a job can
    actually open**, in both shapes, which is the obligation rather than the
    mechanism.
    """
    from molbuilder.jobset.materialize import job_dir_names, shape_of
    from molbuilder.jobset.prep import prep_jobset
    js = _token_ladder("JOB_01_coarse.fdf", "JOB_03_tight.fdf")
    _describe(tmp_path, shape, names=("coarse", "tight"))
    for deck in ("JOB_01_coarse.fdf", "JOB_03_tight.fdf"):
        (tmp_path / deck).write_text("SystemLabel JOB\n")
    (tmp_path / "mb_monitor.py").write_text("# monitor\n")

    prep_jobset(js, tmp_path, emit_sbatch=False)

    dir_of = job_dir_names(js, shape_of(js, tmp_path))
    for job in js.jobs:
        d = tmp_path / dir_of[job.name]
        deck = d / job.script
        assert deck.exists(), f"{shape}: {job.name}'s deck does not resolve"
        assert deck.read_text().strip(), f"{shape}: {job.name}'s deck is empty"
        wrapper = d / (Path(job.script).stem + ".run.sh")
        assert wrapper.exists(), f"{shape}: {job.name} has no runnable wrapper"

    # ...and nothing anywhere points outside the bundle.  The ONLY dangling
    # links a correct tree may hold are the hierarchy's carry-forwards, which
    # are meant to dangle until the producer runs (job-system.md D1).
    stray = [str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*")
             if p.is_symlink() and not p.exists()
             and not p.name.startswith("JOB.")]
    assert stray == [], f"{shape}: dangling links that are not carry-forwards: {stray}"


# --------------------------------------------------------------------- #
#  P6 unit 2 -- the deck and its launch must agree                       #
# --------------------------------------------------------------------- #


def _deck_rendered_for(path, mpi_np):
    """A real deck, rendered through the shipped renderer at a given rank
    count -- so the BENCH-MARKS block is the emitter's, not a fixture's."""
    import numpy as np
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.siesta.input import render_fdf
    from molbuilder.structure import Structure
    s = Structure(elements=["H"] * 20,
                  positions=np.linspace(0, 6, 60).reshape(20, 3))
    s.vacuum = (9.0, 9.0, 9.0)
    Path(path).write_text(
        render_fdf(s, SiestaConfig(system_label="JOB", mpi_np=mpi_np)))


def _one_stage_bundle(base, mpi_np_deck, mpi_np_launch):
    from molbuilder.jobset.model import Job, JobSet, Resources
    # FLAT, so the stage runs in the bundle root and the deck sits where the
    # launch looks for it -- a described bundle, which is what a real one is.
    _describe(base, "flat", names=("coarse",))
    _deck_rendered_for(Path(base) / "JOB_01_coarse.fdf", mpi_np_deck)
    (Path(base) / "JOB_01_coarse.run.sh").write_text("#!/bin/bash\nexit 0\n")
    return JobSet(name="JOB", engine="siesta", kind="ladder",
                  jobs=[Job(name="coarse", script="JOB_01_coarse.fdf",
                            resources=Resources(mpi_np=mpi_np_launch))])


def test_a_deck_rendered_for_no_rank_count_refuses_an_explicit_one(tmp_path):
    """THE LIVE FAILURE OF 2026-08-10, caught before the engine.

    `project-layout.md § 2.3.1`: *"a parameter that depends on the launch
    cannot be decided before the launch is known"* -- step 3 cannot precede
    step 1.  A deck rendered with no rank count derived its `BlockSize` from
    the system size alone; launching it at 14 ranks made SIESTA refuse at
    startup with *"You have too many processors for the system size"*.

    P4 unit 5 put `mpi_np` INTO the deck, which is why the failure was
    diagnosable.  Recording is not agreeing -- this is the agreement.
    """
    from molbuilder.jobset.submit import submit_jobset, SubmitError
    js = _one_stage_bundle(tmp_path, mpi_np_deck=None, mpi_np_launch=14)

    with pytest.raises(SubmitError) as e:
        submit_jobset(js, tmp_path, mode="direct", dry_run=True)
    msg = str(e.value)
    assert "auto" in msg and "14" in msg          # BOTH numbers named
    assert "BlockSize" in msg                     # ...and what depends on it


def test_a_deck_and_a_launch_that_agree_are_not_refused(tmp_path):
    """Both spellings of agreement: an explicit match, and both deferring to
    the wrapper.  A check that refused these would make every ordinary bundle
    unlaunchable."""
    from molbuilder.jobset.submit import submit_jobset
    for deck, launch in ((8, 8), (None, None)):
        d = tmp_path / f"{deck}-{launch}"; d.mkdir()
        js = _one_stage_bundle(d, mpi_np_deck=deck, mpi_np_launch=launch)
        res = submit_jobset(js, d, mode="direct", dry_run=True)
        assert [r.status for r in res] == ["planned"]


def test_two_explicit_rank_counts_that_differ_are_refused(tmp_path):
    """A deck rendered for 8 ranks, launched at 32, is refused before the engine sees
    it -- and BOTH numbers are named.

    The general form of the 2026-08-10 live failure recorded in
    `test_a_deck_rendered_for_no_rank_count_refuses_an_explicit_one`: a
    `BlockSize` derived from one rank count and a launch at another makes SIESTA
    refuse at startup with a message about processor count that says nothing about
    which two statements disagreed. `project-layout.md` § 2.3.1 -- a
    parameter that depends on the launch cannot be decided before the launch is
    known.
    """
    from molbuilder.jobset.submit import submit_jobset, SubmitError
    js = _one_stage_bundle(tmp_path, mpi_np_deck=8, mpi_np_launch=32)
    with pytest.raises(SubmitError) as e:
        submit_jobset(js, tmp_path, mode="direct", dry_run=True)
    assert "8" in str(e.value) and "32" in str(e.value)


def test_a_deck_with_no_bench_marks_says_nothing_and_is_not_refused(tmp_path):
    """A deck that never recorded its launch cannot disagree with one.  The
    check is an agreement between two statements, not a demand that every deck
    make one."""
    from molbuilder.jobset.model import Job, JobSet, Resources
    from molbuilder.jobset.submit import submit_jobset
    _describe(tmp_path, "flat", names=("coarse",))
    (tmp_path / "JOB_01_coarse.fdf").write_text("SystemLabel JOB\n")
    (tmp_path / "JOB_01_coarse.run.sh").write_text("#!/bin/bash\nexit 0\n")
    js = JobSet(name="JOB", engine="siesta", kind="ladder",
                jobs=[Job(name="coarse", script="JOB_01_coarse.fdf",
                          resources=Resources(mpi_np=99))])
    assert submit_jobset(js, tmp_path, mode="direct", dry_run=True)


# --------------------------------------------------------------------- #
#  P6 unit 6 -- prep prints what it resolved, so submit is a plain yes    #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("deck,launch", [
    (None, None),        # both defer to the wrapper
    (8, 8),              # the same explicit count
    (None, 14),          # THE live failure: a count imposed on an auto deck
    (8, 32),             # two explicit counts that differ
    (8, None),           # a deck rendered for 8, launched without saying so
])
def test_the_prep_warning_and_the_submit_refusal_cannot_disagree(
        tmp_path, deck, launch):
    """One comparison, two surfaces — the guard that keeps them one.

    `project-layout.md` § 2.3.3: *"Printing what it resolved is what makes `submit`
    a plain yes."*  That only holds if what `prep` says and what `submit` does
    are the same answer. Two implementations of *"do these agree?"* would drift
    the way the plan's item 12c describes — agreeing today, with
    nothing keeping them agreeing — and the drift is silent in the worst
    direction: a prep that reports no problem before a submit that refuses.

    So this asserts the equivalence directly, across every shape the question
    has, rather than checking each surface's wording in isolation.
    """
    from molbuilder.jobset.agreement import launch_agreement
    from molbuilder.jobset.submit import submit_jobset, SubmitError

    js = _one_stage_bundle(tmp_path, mpi_np_deck=deck, mpi_np_launch=launch)
    warns = launch_agreement(tmp_path, js.jobs[0]).verdict == "differs"
    try:
        submit_jobset(js, tmp_path, mode="direct", dry_run=True)
        refuses = False
    except SubmitError:
        refuses = True
    assert warns == refuses, (
        f"deck={deck} launch={launch}: prep "
        f"{'warns' if warns else 'is quiet'} and submit "
        f"{'refuses' if refuses else 'proceeds'}")


def _report(tmp_path, job):
    """The prep report printed from the answer the one entry builds
    (`_echo_prep_answer`; `prep.prep_stage`'s step 8: `launch_agreement`,
    kept unless the deck makes no claim).

    API-level on purpose: a deck rendered for ANOTHER launch is what the road
    cannot make -- prep renders the deck for the very launch it prepares --
    so the report's wording for it is pinned here, and the ledger half rides
    the road tests (`test_prep_from_the_browser.py`).  The contract is unchanged:
    the reporter and submit's refusal both read the ONE comparison,
    `agreement.launch_agreement`."""
    import contextlib, io
    from molbuilder.jobset._cli import _echo_prep_answer
    from molbuilder.jobset.agreement import launch_agreement
    from molbuilder.jobset.materialize import Attempt
    from molbuilder.jobset.prep import PrepAnswer

    g = launch_agreement(tmp_path, job)
    ans = PrepAnswer(
        "run", "coarse",
        attempt=Attempt("coarse", tmp_path, True, [], [], None, False),
        resources={"mpi_np": job.resources.mpi_np, "cpus_per_task": None,
                   "continue_retries": 0},
        deck=job.script, agreement=g if g.verdict != "silent" else None)
    # stdout and stderr are one stream here, as at a terminal -- the warning
    # goes to stderr on purpose, so a report piped to a file still carries
    # it to the screen.
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        _echo_prep_answer(ans, tmp_path)
    return out.getvalue() + err.getvalue()


def _prep_output(tmp_path, mpi_np_deck, mpi_np_launch):
    from molbuilder.jobset.model import Job, Resources
    _deck_rendered_for(tmp_path / "JOB_01_coarse.fdf", mpi_np_deck)
    return _report(tmp_path, Job(name="coarse", script="JOB_01_coarse.fdf",
                                 resources=Resources(mpi_np=mpi_np_launch)))


def test_prep_names_both_numbers_and_says_submit_will_refuse(tmp_path):
    """The gap P6 unit 2 opened: `submit` refuses correctly and at the last
    honest moment, but a refusal that first appears when you are committing
    cluster time is exactly the surprise `prep` exists to prevent.

    So the warning arrives while it is still cheap to change your mind, and it
    says what will happen rather than only what is wrong.
    """
    out = _prep_output(tmp_path, mpi_np_deck=None, mpi_np_launch=14)
    assert "auto" in out and "14" in out          # BOTH numbers, as at submit
    assert "REFUSE" in out                        # ...and what comes next
    assert "BlockSize" in out                     # ...and why it matters


def test_prep_says_the_deck_agrees_when_it_does(tmp_path):
    """Not decoration: a report that mentioned the deck only on disagreement
    would leave a reader unable to tell *checked and fine* from *not checked*,
    which is the whole difference between a review step and a quiet one."""
    out = _prep_output(tmp_path, mpi_np_deck=8, mpi_np_launch=8)
    assert "agrees with this launch" in out
    assert "REFUSE" not in out


def test_prep_stays_quiet_about_a_deck_that_makes_no_claim(tmp_path):
    """A deck with no BENCH-MARKS block has said nothing about its launch, so
    there is nothing to report — and reporting *"agrees"* would be a claim
    nobody made."""
    from molbuilder.jobset.model import Job, Resources

    (tmp_path / "JOB_01_coarse.fdf").write_text("SystemLabel JOB\n")
    out = _report(tmp_path, Job(name="coarse", script="JOB_01_coarse.fdf",
                                resources=Resources(mpi_np=99)))
    assert "rendered for" not in out
    assert "resources:" in out                    # the rest of the report stays


def test_prep_reports_the_resources_this_stage_will_be_launched_with(tmp_path):
    """The second of § 2.3.3's three — *"the measured numbers, the chosen
    geometry and the rendered deck appear together"*. `auto` is printed as a
    word rather than omitted, because a blank line and *"the wrapper decides"*
    are different claims."""
    out = _prep_output(tmp_path, mpi_np_deck=8, mpi_np_launch=8)
    assert "resources: mpi_np 8" in out
    assert "omp auto" in out


def test_a_flat_stage_that_never_ran_does_not_borrow_a_siblings_state(tmp_path):
    """`project-layout.md` § 1: flat is **depth 1** — every stage shares one
    directory and they are told apart by the deck's token in each filename.

    The observe layer asked *"is there a `.out` in this stage's directory?"*,
    which is right in the hierarchy (the directory already chose the stage) and
    silently wrong in flat: `coarse` finishing made `tight` claim to be running
    too, because the glob matched `coarse`'s file.

    `Shape.stage_glob` is what answers *which files are this stage's*, and this
    is the caller it was built for.
    """
    from molbuilder.jobset.materialize import prepare_attempt
    js = _token_ladder("JOB_01_coarse.fdf", "JOB_03_tight.fdf")
    _describe(tmp_path, "flat")
    # coarse ran; tight has not been touched
    (tmp_path / "JOB_01_coarse-run0.out").write_text("Job completed\n")

    by_name = {s.name: s for s in jobset_status(js, tmp_path).stages}
    assert by_name["coarse"].state not in ("pending", "not-started")
    assert by_name["tight"].state == "pending", (
        "tight borrowed coarse's output: they share a directory in flat")
    assert "not launched" in by_name["tight"].detail
    # Deliberately NOT asserting `first_incomplete` here: that depends on the
    # DECODER's verdict for coarse ("Job completed" is not necessarily
    # `finished`), which is a different contract.  This test is about which
    # files a stage owns, and asserting past that would make it fail for a
    # reason it does not name.


def test_two_flat_rungs_with_real_output_each_report_their_own(tmp_path):
    """The SECOND half of the same rule, and the half that had no test.

    The one above proves a stage with NO output does not borrow a sibling's —
    that is caught by the existence gate, which was already shape-aware. This
    proves the case the gate lets through: **both** rungs have real output, and
    they must still be told apart. The gate narrows with `Shape.stage_glob` and
    then hands the directory to `run_status`; until 2026-09-08 it handed over
    no glob at all, so `run_status` picked "highest ordinal, newest mtime"
    across the whole directory and every row showed the newest rung's state.

    Reverting only that one argument passes every other test in this suite —
    which is how it shipped. This is the test that fails.
    """
    js = _token_ladder("JOB_01_coarse.fdf", "JOB_03_tight.fdf")
    _describe(tmp_path, "flat")
    old = tmp_path / "JOB_01_coarse-run0.out"
    new_ = tmp_path / "JOB_03_tight-run0.out"
    old.write_text("Siesta Version: 5.4.2\nsiesta: iscf\n>> End of run:  1-JAN-2026\n")
    new_.write_text("Siesta Version: 5.4.2\nsiesta: iscf\nscf:  1  -100.0\n")
    past = _time.time() - 86400
    os.utime(old, (past, past))
    for a in (tmp_path / "run.json",):
        a.write_text('{"mode": "direct"}')

    by_name = {s.name: s for s in jobset_status(js, tmp_path).stages}
    assert (by_name["coarse"].state, by_name["tight"].state) == (
        "finished", "running"), (
        "the rungs did not each report their own file -- the newest one spoke "
        "for the whole directory, which is the flat-shape bug this narrows: "
        f"{by_name['coarse']} / {by_name['tight']}")


def test_the_hierarchy_is_unaffected_because_its_directory_already_chose(tmp_path):
    """`stage_glob` is `*` there, so the behaviour is identical to before —
    which is the point of one object answering for both."""
    from molbuilder.jobset.materialize import prepare_attempt
    js = _token_ladder("JOB_01_coarse.fdf", "JOB_03_tight.fdf")
    _describe(tmp_path, "hierarchical")
    a = prepare_attempt(js, tmp_path, "coarse").dir
    (a / "JOB_01_coarse.out").write_text("Job completed\n")

    by_name = {s.name: s for s in jobset_status(js, tmp_path).stages}
    assert by_name["coarse"].state not in ("pending", "not-started")
    assert by_name["tight"].state == "not-started"      # no directory at all


# --------------------------------------------------------------------- #
#  P6 unit 1 -- step ONE of the five, lifted out of the benchmark        #
# --------------------------------------------------------------------- #

def test_prep_resolves_the_machine_before_anything_else(tmp_path):
    """`project-layout.md` § 2.3.1 step 1: *"Resolve the machine — detect
    cores, GPUs, scheduler, conda → environment.json"*, and the order is
    forced, not chosen.

    **The general `prep` did not do this at all until 2026-08-10.**
    `bench/prep.py` did; `prep_jobset` went straight to rendering wrappers on
    a machine nobody had asked about. § 2.3.1a says how to read that: the
    benchmark is *"the one place this framework is already built"*, built
    there because that is where the need appeared first — so **the general
    part is lifted out of it**, not borrowed from it.
    """
    from molbuilder.jobset.prep import prep_jobset
    js = _sweep()
    _write_fdf(tmp_path / "job-gpu.fdf")
    assert not (tmp_path / "environment.json").exists()

    prep_jobset(js, tmp_path, emit_sbatch=False)

    import json
    env = json.loads((tmp_path / "environment.json").read_text())
    assert env["schema"] == "molbuilder/environment@2"
    assert env["topology"]["cores_per_socket"] >= 1     # a real probe ran




def test_the_machine_probe_is_molbuilders_not_the_benchmarks():
    """§ 2.3.1a: *"stating it the other way round would make the general case
    look like a special case of the special case."*

    The module moved out of `bench/` on 2026-08-10. Its persisted artifact was
    **already** registered under the `molbuilder/environment@N` name
    (`job-contracts.md` § 6.1) — the schema saying it was never the
    benchmark's to own. What is asserted is the NAME; the major moved to @2 on
    2026-08-17 (N2) when the record gained the reachable domains, and pinning
    the major here would make this test fail for a reason it is not about.
    """
    import importlib
    from molbuilder.scheduler import SCHEMA
    from molbuilder.persist import schema_name
    assert schema_name(SCHEMA) == "molbuilder/environment"
    assert importlib.util.find_spec("molbuilder.bench.environment") is None


def test_a_machine_WITHOUT_A_RECORD_stops_the_prep(tmp_path, monkeypatch):
    """**Reversed 2026-09-02.**  It read *"a machine that will not probe does
    not stop the prep"* -- best-effort, on the reasoning that `prep` has four
    other steps and the deck/launch agreement is what refuses a wrong launch.

    That was true and is now the wrong trade.  What a best-effort step 1
    produced was a wrapper whose numbers came from **whichever box happened
    to run prep**, which for a bundle described at a desk and run on a
    cluster is the wrong machine -- and the number looks exactly like a right
    one, so nothing downstream can tell.  A missing *description* becoming a
    missing *calculation* is the cheaper failure by far: it costs one command
    *(user, 2026-09-02: "all environments have to be explicitly probed and
    stored. no environment json, error")*.

    So the probe is gone entirely -- there is nothing left to be
    best-effort ABOUT -- and a record-less machine is refused, by name, with
    the command.
    """
    from molbuilder.jobset.prep import PrepError, prep_jobset
    from molbuilder.scheduler import machine_scope_path

    # NOTHING is probed: the record the suite writes for every test is gone.
    Path(machine_scope_path()).unlink(missing_ok=True)

    js = _sweep()
    _write_fdf(tmp_path / "job-gpu.fdf")
    with pytest.raises(PrepError) as exc:
        prep_jobset(js, tmp_path, emit_sbatch=False)

    said = str(exc.value)
    assert "no machine record" in said, said
    assert "jobset probe --write" in said, (
        "the refusal does not name the command that fixes it: " + said)
    assert not (tmp_path / "environment.json").exists(), (
        "a refused prep left a record behind -- it probed after all")


# --------------------------------------------------------------------- #
#  P12 unit 3 -- the two environments, pinned on the claim that matters  #
# --------------------------------------------------------------------- #

def _prep_bundle(base, *, scheduler: bool, monkeypatch):
    """Prep the same two-stage flat calculation, with and without a cluster.

    A CLUSTER IS ITS RECORD: a scheduler and the queues it lists.  And a run
    on it STATES its queue, wall and memory (`architecture.md` § 5.2) -- a
    `scheduler` block in `molbuilder.json` stood in for both until
    2026-10-02.
    """
    import dataclasses
    from molbuilder.jobset.prep import prep_jobset
    base.mkdir(parents=True, exist_ok=True)
    _describe(base, "flat", names=("coarse", "tight"))
    js = _token_ladder("JOB_01_coarse.fdf", "JOB_02_tight.fdf")
    for j in js.jobs:
        (base / j.script).write_text("SystemLabel JOB\n")
        if scheduler:
            j.resources = dataclasses.replace(
                j.resources, domain="public", time="0-04:00:00", mem="8G")
    from molbuilder.scheduler import Domain, Topology
    monkeypatch.chdir(base)
    # THE MACHINE'S RECORD, in a config directory of its own: the bundle is
    # where prep SNAPSHOTS it, so a record written there is the copy, not the
    # machine.  Until 2026-10-02 the cluster record was written into the
    # bundle, the config root pointed there too, and the conftest default
    # then overwrote it -- both arms prepped a workstation, and the claim
    # below compared a workstation with itself.
    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR",
                       str(base.parent / f"{base.name}-config"))
    from conftest import write_machine_record
    write_machine_record(
        scheduler="slurm" if scheduler else "workstation",
        topology=Topology(sockets=2, cores_per_socket=64),
        domains=([Domain(name="public", partition="public",
                         qos="public", max_time="1-00:00:00")]
                 if scheduler else []),
        env_init={"activation": "conda activate",
                  "preamble": "source /x/conda.sh"})
    prep_jobset(js, base, env="molbuilder-siesta")
    return base


def test_the_inner_wrapper_is_byte_identical_on_both(tmp_path, monkeypatch):
    """**The claim the whole two-layer split rests on**, and it was asserted
    nowhere until 2026-08-10.

    `architecture.md` § 9: the outer `.sbatch` is a header whose body calls the
    inner `.run.sh`, and the inner one owns activation and launch.  If that is
    true then the SAME `.run.sh` runs in both places -- so a run you debugged
    on your laptop is the run the cluster performs.  If it ever stops being
    true, the laptop stops being a rehearsal and this test is how you find out.
    """
    ws = _prep_bundle(tmp_path / "ws", scheduler=False, monkeypatch=monkeypatch)
    hpc = _prep_bundle(tmp_path / "hpc", scheduler=True, monkeypatch=monkeypatch)
    def _logic(text: str) -> str:
        """The script without its PROVENANCE timestamp.

        ``generated-at`` is wall clock at seconds precision, and the two preps
        above run in sequence -- so a byte-for-byte comparison fails whenever
        they straddle a second boundary, which has nothing to do with
        workstation versus cluster and everything to do with how busy the
        process was. The claim being tested is about the script's LOGIC; a
        generation timestamp is provenance by definition, so it is excluded
        rather than the claim being weakened.
        """
        return "\n".join(l for l in text.splitlines()
                          if not l.lstrip("# ").startswith("generated-at"))

    for name in ("JOB_01_coarse.run.sh", "JOB_02_tight.run.sh"):
        a = _logic((ws / name).read_text())
        b = _logic((hpc / name).read_text())
        assert a == b, (
            f"{name} differs between a workstation and a cluster -- the inner "
            "wrapper is supposed to be the same file, so a laptop run is a "
            "rehearsal of the cluster run")


# --------------------------------------------------------------------- #
#  P12 unit 4 -- the five steps run in their order                       #
# --------------------------------------------------------------------- #

def test_prep_resolves_the_machine_before_it_writes_anything(tmp_path,
                                                             monkeypatch):
    """`project-layout.md` § 2.3.1: the order is forced, not chosen.

    **Step 3 cannot precede step 1.** A deck carries values that depend on how
    it will be launched -- a block size derived from the rank count, an
    eigensolver that also picks which environment the wrapper activates -- so a
    deck written before the machine is known has guessed at them.

    The outcome alone cannot show this: a prep that resolved the machine LAST
    leaves exactly the same files behind.  So the order is observed directly --
    the two steps' functions are wrapped and the call order recorded.  Step 1
    is `resolve_target`, step 4 is `write_run_wrapper`; if the wrapper is
    written first, the deck it accompanies was rendered against nothing.
    """
    from molbuilder.jobset import prep as _prep

    base = tmp_path / "b"
    base.mkdir()
    _describe(base, "flat", names=("coarse", "tight"))
    js = _token_ladder("JOB_01_coarse.fdf", "JOB_02_tight.fdf")
    for j in js.jobs:
        (base / j.script).write_text("SystemLabel JOB\n")
    monkeypatch.chdir(base)
    # THE SANDBOX IS THE CONFIG ROOT.  This config was read through the
    # working-directory step, which is gone (configuration.md § 2.1a) --
    # without naming the directory the write lands in a file nothing
    # opens, and the test passes having configured nothing.
    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(base))
    # The record follows the config root: this env var moves the
    # machine scope, and prep refuses without a record there.
    from conftest import write_machine_record
    write_machine_record()

    from molbuilder import runwrap as _rw

    order: list = []
    real_target, real_wrap = _prep.resolve_target, _rw.write_run_wrapper

    def spy_target(b):
        order.append("1 machine")
        return real_target(b)

    def spy_wrap(*a, **k):
        order.append("4 wrapper")
        return real_wrap(*a, **k)

    # `prep_jobset` imports the wrapper writer inside its own body, so the
    # patch goes on the SOURCE module -- patching `prep` would miss it.
    monkeypatch.setattr(_prep, "resolve_target", spy_target)
    monkeypatch.setattr(_rw, "write_run_wrapper", spy_wrap)
    _prep.prep_jobset(js, base, env="molbuilder-siesta")

    assert order[0] == "1 machine", (
        f"prep wrote a wrapper before resolving the machine: {order}")
    assert "4 wrapper" in order, "no wrapper was written at all"
    # ...and the machine is resolved ONCE per bundle, not once per stage.
    assert order.count("1 machine") == 1, order


def test_the_library_itself_refuses_a_whole_ladder(tmp_path):
    """U5: the no-chain rule lives at the SEAM, not only in the CLI's
    stage resolution -- a library caller handing submit_jobset a two-stage
    ladder with no `only` is refused in EVERY mode, because direct-running
    stages in order would be local chaining (project-layout.md § 1.6)."""
    import pytest as _pytest
    from molbuilder.jobset.model import Job, JobSet, Resources
    from molbuilder.jobset.submit import SubmitError, submit_jobset
    js = JobSet(name="JOB", engine="siesta", kind="ladder",
                jobs=[Job(name="coarse", script="JOB_01_coarse.fdf",
                          resources=Resources()),
                      Job(name="tight", script="JOB_02_tight.fdf",
                          resources=Resources())])
    for mode in ("direct", "submit"):
        with _pytest.raises(SubmitError, match="ONE stage at a time"):
            submit_jobset(js, tmp_path, mode=mode, dry_run=True)
    # named, it proceeds to a single planned launch
    res = submit_jobset(js, tmp_path, mode="direct", dry_run=True,
                        only="coarse")
    assert [r.name for r in res] == ["coarse"]


def test_resources_fields_equal_the_contracts_list_exactly():
    """job-contracts § 6.2's OWN SENTENCE and the dataclass, an equality
    in BOTH directions (U19, 2026-08-12; re-pinned I1, 2026-08-13).  The
    first version held its own nine-name literal, so editing the doc's
    list back to seven reddened NOTHING -- a doc-equality test that never
    read the doc.  This parses the sentence's backticked field names, so
    a field added without a § 6.2 decision fails here, a doc row deleted
    fails here, and the doc drifting from the class fails here."""
    import dataclasses
    import re as _re
    from molbuilder.jobset.model import Resources
    doc = (Path(__file__).resolve().parent.parent
           / "docs" / "execution" / "job-contracts.md").read_text(
               encoding="utf-8")
    start = doc.index("dataclass holds exactly")
    end = doc.index("This sentence said", start)
    names = {m for m in _re.findall(r"`([a-z_]+)`", doc[start:end])}
    assert len(names) >= 7, (
        "§ 6.2's field sentence could not be parsed -- repoint this "
        f"(found {sorted(names)})")
    assert {f.name for f in dataclasses.fields(Resources)} == names


def _write_domains(where, rows):
    """A probed `environment.json` carrying the reachable domains."""
    from molbuilder.scheduler import (FILENAME, Domain, Environment,
                                        Topology, write_environment)
    return write_environment(
        Environment(scheduler="slurm", topology=Topology(cores_per_socket=64),
                    domains=[Domain(name=n, partition=p, qos=q, max_time=t)
                             for n, p, q, t in rows]),
        Path(where) / FILENAME)


# --------------------------------------------------------------------- #
#  G7 — the GPU answer travels; the deck is not re-read for it          #
# --------------------------------------------------------------------- #

def test_every_directory_prep_makes_says_what_it_is(tmp_path):
    """Invariant 6b, and the drift half of it.

    `project-layout.md` § 1.4a gives § 1.4's container-or-run rule a
    mechanism: the code that makes a directory stamps it, because that code
    is the only one that knows.  Two things have to hold or the mechanism is
    decoration --

    1. **every directory prep made answers.**  One that does not is read
       ALONE, which for a directory inside a live calculation is a silently
       partial answer.
    2. **the stamp agrees with the naming authority.**  `role` and the
       directory's name are two statements of one fact; `job_dir_names` is
       the authority for the second, so a container must be a directory the
       authority maps a job to, and the run must sit under it.

    Without (2) this test would pass on a writer that stamped everything
    ``container`` -- the same shape as `_run_ending.py`'s
    ``set(READERS) == set(run_output_roles())``, which is how two lists in
    this codebase are made unable to drift.

    MUTATION THIS MUST FAIL AGAINST: stamp only the attempt, or stamp the
    stage directory ``run``.
    """
    from molbuilder import calcdirs
    from molbuilder.jobset.materialize import (job_dir_names, prepare_attempt,
                                               shape_of)

    js = _token_ladder("JOB_01_coarse.fdf", "JOB_03_tight.fdf")
    for job in js.jobs:
        (tmp_path / job.script).write_text("x")
    rep = prepare_attempt(js, tmp_path, "coarse")

    attempt = rep.dir
    stage_dir = attempt.parent

    said_run = calcdirs.read(attempt)
    said_container = calcdirs.read(stage_dir)
    assert said_run is not None, f"{attempt} carries no calcdir record"
    assert said_container is not None, (
        f"{stage_dir} carries no calcdir record -- the attempt was stamped "
        f"and its container was not, so the level a viewer lands on when it "
        f"clicks the stage is the one that cannot answer")
    assert said_run.role == calcdirs.RUN
    assert said_container.role == calcdirs.CONTAINER

    # (2) the stamp and the name are the same fact, said twice.
    authority = set(job_dir_names(js, shape_of(js, tmp_path)).values())
    assert stage_dir.relative_to(tmp_path).as_posix() in authority, (
        f"stamped {stage_dir.name!r} a container, but the naming authority "
        f"maps no job to it: {sorted(authority)}")

    # ...and `of` leads back, from either level.
    for d in (attempt, stage_dir):
        assert (d / calcdirs.read(d).of).resolve() == tmp_path.resolve()


def test_the_progress_channel_ends_up_in_the_run_and_nowhere_else(
        tmp_path, monkeypatch):
    """One home for the live log, and the home is the run directory.

    `project-layout.md` § 1.0: a run directory "holds everything it
    produces".  The progress channel is a product -- the deck names it, the
    engine writes it -- so two copies is one too many, and which copy a
    reader lands on then depends on which directory they clicked.

    MEASURED, 2026-09-19, on a finished Raman run: `prep` seeds the log
    beside the deck it renders, which in the hierarchy is the stage
    CONTAINER.  The run then wrote its own inside the attempt.  Result: a
    971-byte stub with no `# concluded:` footer in `01_raman/`, and the real
    1763-byte concluded log in `01_raman/run-0/` -- different inodes, one
    name.  An unconcluded log is how every reader tells a run is still
    going, so the stage directory reported a FINISHED calculation as
    *Running*, permanently, and offered the stub to open.

    Driven through `jobset init` -> `prep run` (`support.road`).

    MUTATION THIS MUST FAIL AGAINST: copy instead of move, or leave the seed
    where it was rendered.
    """
    from support.road import describe_h2, jobset
    bundle = describe_h2(tmp_path, monkeypatch)
    r = jobset("prep", "run", "coarse", "--bundle", bundle,
               "--target", "this", "--np", "2", "--cpus-per-task", "1")
    assert r.exit_code == 0, r.output
    log = "H2_01_coarse.molwatch.log"
    attempt = bundle / "01_coarse" / "run-0"
    assert (attempt / log).is_file(), (
        f"the seeded progress log did not reach the run directory; "
        f"{attempt} holds {sorted(p.name for p in attempt.iterdir())}")
    assert not (bundle / "01_coarse" / log).exists(), (
        f"the seed is still in the stage container as well -- two homes for "
        f"one log is the state that reported a finished run as 'Running'")
