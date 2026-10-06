"""The ``jobset`` framework: model + persistence + validate + materialize
+ plan (docs/execution/job-system.md), and the SIESTA
stage-ladder producer."""

from __future__ import annotations

import re
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
from molbuilder.jobset.materialize import materialize
from molbuilder.paths import trial_name
from molbuilder.jobset.plan import render_plan
from molbuilder.jobset.submit import SubmitError, plan_launch


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

# Retired 2026-10-06 (MEMORY gate 5: a failing test fed a hand-built job set
# is retired, never updated): 24 tests built a JobSet by hand -- in a folder
# with no task.json, with decks naming no stage, or asking the naming
# authority with no shape.  A folder no description names and a job whose
# deck names no stage are refused now (`materialize.shape_of`,
# `job_dir_names`); the road's tables prep real calculations instead.


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
    assert trial_name("stage1") == "bench-stage1"


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
    from molbuilder.warmfiles import warm_list
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
                for r in warm_list("siesta", "optimization").rules if r.carry]
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
    assert "restart files it declares" in txt  # ...headed as declared
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
#      route (tests/data/prep_protocol.toml; resolve refuses by name);   #
#    * the .CG carry conditionals -> test_restart_group.py, repointed    #
#      at the live `_warm_declaration` seam the same day;                #
#    * no-edges / continue_retries on resources -> carried per element   #
#      (tests/data/launch_values.toml, tests/data/prep_protocol.toml);   #
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
                                      use_gpu=True, gres="gpu:1",
                                      time="0-01:00:00", mem="8G")),
              Job(name="G1K2C4", script="job-gpu.fdf",
                  resources=Resources(mpi_np=2, cpus_per_task=4,
                                      use_gpu=True, gres="gpu:1",
                                      time="0-01:00:00", mem="8G"))])


def _write_fdf(path):
    # minimal .fdf so write_run_wrapper renders for real (bash -n validated).
    # The clean restart group rides along: since 2026-08-21 the submission
    # door verifies a trial's cold start against its deck, and a real deck
    # always carries the group written out (`siesta/input.py`, 2026-08-18).
    path.write_text("SystemName test\nSystemLabel test\nNumberOfAtoms 2\n"
                    "DM.UseSaveDM .false.\nMD.UseSaveXV .false.\n")


# Retired 2026-10-06 with `prep_jobset` as a door of its own -- it takes
# the prep entry's answer now (`job-system.md` § 5.0) -- five tests that
# handed it a hand-built job set and decks written by hand.  Each rule is
# held on the road: every file a stage's prep writes, real and in its own
# folder (`tests/data/the_catalogue.toml`); the retry budget baked into the
# run script, and none when none is asked (`tests/data/prep_protocol.toml`);
# `STAGE-PLAN.md` (`tests/data/hand_overs.toml`'s `plan` rows).  A job
# whose deck is missing cannot reach the wrapper step: the entry writes the
# decks it wraps.


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

# Retired 2026-10-05 with the doors they called (`submit_jobset`'s
# dry run and send; `job-system.md` § 6.0, one entry): the job id
# read off `sbatch` and recorded (`test_launch_door.py`, through the
# road), the scheduler's refusal carried in its own words (the
# shelf test in `test_prep_bench_fold.py`, the one sender's), the
# per-trial flags (every benchmark row's line), `--exclusive` over
# `--mem` (`test_one_emitter.py`), and the one-job-at-a-time refusal
# -- a sweep sent to a scheduler goes one job per shelf, by the
# entry's own dispatch, so no door is left to refuse it.  Each stood
# on a hand-built job set with no files the plan reads.

# Retired 2026-10-04 (user: "any fucking faking tests should be retired"):
# 13 tests here stood on restart files or outputs written by hand,
# or on run endings a mocked process returned (`process/testing.md` § 6).


#  A ladder is submitted one stage at a time, so the tests below cover one
#  launch each; the halt-on-failure case is structural, see
#  `test_a_ladder_refuses_to_act_on_all_of_itself`.  The earlier
#  scheduler-chained design: docs/archive/2026-08-10-stage-chaining.md


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
    # nothing lays out a hand-built set: prep is the described route's
    # alone (its library door, `prep_jobset` called directly, went
    # 2026-10-06), and nothing was laid out here
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


# `test_direct_launch_carries_the_launch_door_claim` retired
# 2026-10-05: a run launched here without the claim is refused by
# its own run script (exit 2, `job-contracts.md` § 2.6), so every
# road row that launches here and builds on the run asserts it.


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
    maximal = _blocks(write_run_wrapper(tmp_path / "GPU.fdf", resources=Resources(mpi_np=4, cpus_per_task=1, use_gpu=True, gres="gpu:1", continue_retries=2), env="e").read_text())
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
                               "--shape", "hierarchical", "--engine", "pyscf",
                               "--calculation", "optimization"])
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


# `test_submit_only_takes_the_same_two_spellings_as_every_surface`
# retired 2026-10-05: `launch` takes `#N` through the road now
# (`test_stage_names.py`), and the one resolver's listing is
# asserted above.


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
    assert flat.stage_dir("03_tight") == "."             # a joinable path, not None
    # *(`Shape.run_basename` -- the stem in flat, ``None`` in the hierarchy --
    # was asserted here until 2026-10-04: a run is asked about by its stem
    # in either shape, `runfiles.stem`.)*


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


def _describe(base, shape, names=("coarse", "tight")):
    """Write a real `task.json` beside a bundle, through the one codec."""
    from molbuilder.task import (FILENAME, Stage, StructureRef, Task,
                                 derive_run, write_task)
    stages = tuple(Stage(name=n, overrides={"mesh_cutoff": 200}) for n in names)
    write_task(Path(base) / FILENAME, Task(
        engine="siesta", shape=shape, calculation="optimization",
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


# `test_prep_leaves_every_job_a_readable_deck_and_wrapper` retired
# 2026-10-06 (a hand-built job set handed to `prep_jobset`): every file a
# stage's prep writes is there as the card names it, in both shapes, and a
# launch runs it -- `tests/data/the_catalogue.toml`.


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


# --------------------------------------------------------------------- #
#  P6 unit 6 -- prep prints what it resolved, so submit is a plain yes    #
# --------------------------------------------------------------------- #


def _report(tmp_path, job):
    """The prep report printed from the answer the one entry builds
    (`_echo_prep_answer`; `prep.prep_stage`'s step 8: `launch_agreement`,
    kept unless the deck makes no claim).

    API-level because no road row reads the printed report: what it says of
    a deck that agrees with its launch, and the resources it states.  *(The
    tests of a deck rendered for ANOTHER launch -- a state only a hand-
    edited file makes, prep rendering the deck for the launch it prepares
    -- were retired 2026-10-05; the refusal stays, explicit, untested.)*"""
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


def test_prep_says_the_deck_agrees_when_it_does(tmp_path):
    """Not decoration: a report that mentioned the deck only on disagreement
    would leave a reader unable to tell *checked and fine* from *not checked*,
    which is the whole difference between a review step and a quiet one."""
    out = _prep_output(tmp_path, mpi_np_deck=8, mpi_np_launch=8)
    assert "agrees with this launch" in out
    assert "REFUSE" not in out


def test_prep_reports_the_resources_this_stage_will_be_launched_with(tmp_path):
    """The second of § 2.3.3's three — *"the measured numbers, the chosen
    geometry and the rendered deck appear together"*. What is stated is
    printed, and nothing else: no launch value is left for the wrapper to
    decide (`architecture.md` § 5.2), so `omp auto` claimed a decision
    nobody makes — and a PySCF run has no rank count to call `auto`."""
    out = _prep_output(tmp_path, mpi_np_deck=8, mpi_np_launch=8)
    assert "resources: mpi_np 8" in out
    assert "auto" not in out


    # Deliberately NOT asserting `first_incomplete` here: that depends on the
    # DECODER's verdict for coarse ("Job completed" is not necessarily
    # `finished`), which is a different contract.  This test is about which
    # files a stage owns, and asserting past that would make it fail for a
    # reason it does not name.


# --------------------------------------------------------------------- #
#  P6 unit 1 -- step ONE of the five, lifted out of the benchmark        #
# --------------------------------------------------------------------- #

# `test_prep_resolves_the_machine_before_anything_else` retired 2026-10-06
# (a hand-built job set handed to `prep_jobset`): the calculation's copy of
# its machine's record is what every verb after prep reads, and a refused
# prep leaves none (`tests/data/prep_protocol.toml`).


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


# `test_a_machine_WITHOUT_A_RECORD_stops_the_prep` is a row since
# 2026-10-06: a machine never probed is refused at prep, naming the probe
# that records it (`tests/data/launch_values.toml`).


# --------------------------------------------------------------------- #
#  P12 unit 3 -- the two environments, pinned on the claim that matters  #
# --------------------------------------------------------------------- #

def test_the_inner_wrapper_is_byte_identical_on_both(tmp_path, monkeypatch):
    """**The claim the whole two-layer split rests on**, and it was asserted
    nowhere until 2026-08-10.

    `architecture.md` § 9: the outer `.sbatch` is a header whose body calls the
    inner `.run.sh`, and the inner one owns activation and launch.  If that is
    true then the SAME `.run.sh` runs in both places -- so a run you debugged
    on your laptop is the run the cluster performs.  If it ever stops being
    true, the laptop stops being a rehearsal and this test is how you find out.

    ON THE ROAD, two calculations -- so a function, not a row, which is one:
    one description by `jobset init`, copied before anything was prepped, and
    each prepped by `jobset prep` -- for this machine, a workstation, and for
    `sol`, a cluster whose record names a queue, with the same way into an
    environment (a hand-built job set handed to `prep_jobset` stood here
    until 2026-10-06).
    """
    import json
    import shutil
    from conftest import write_machine_record
    from molbuilder.config_dir import ensure_private_dir
    from molbuilder.scheduler import (Domain, Environment, Topology,
                                      environments_dir,
                                      named_environment_path,
                                      write_environment)
    from support.road import describe_h2, jobset
    enters = {"activation": "conda activate",
              "preamble": "source /x/conda.sh"}
    write_machine_record(scheduler="workstation", env_init=enters)
    ensure_private_dir(environments_dir())
    write_environment(Environment(
        scheduler="slurm", topology=Topology(sockets=2, cores_per_socket=64),
        domains=[Domain.from_row({
            "name": "cpu", "partition": "cpu", "qos": "public",
            "max_time": "1-00:00:00", "max_cores": 128,
            "node_types": [{"cores": 128, "nodes": 4}]})],
        env_init=enters), named_environment_path("sol"))
    ws = describe_h2(tmp_path, monkeypatch, shape="flat")
    hpc = ws.parent / "H2-on-sol"
    shutil.copytree(ws, hpc)
    # A RUN ON A CLUSTER STATES ITS QUEUE, WALL AND MEMORY
    # (`architecture.md` § 5.2) -- in the description, as a person states it.
    task = json.loads((hpc / "task.json").read_text())
    task["allocation"] = {"domain": "cpu", "time": "04:00:00", "mem": "8G"}
    (hpc / "task.json").write_text(json.dumps(task, indent=2))
    for bundle, target in ((ws, "this"), (hpc, "sol")):
        r = jobset("prep", "run", "coarse", "--bundle", bundle,
                   "--target", target)
        assert r.exit_code == 0, r.output
    assert (hpc / "H2_01_coarse.sbatch").is_file(), "the cluster has no header"

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

    name = "H2_01_coarse.run.sh"
    a = _logic((ws / name).read_text())
    b = _logic((hpc / name).read_text())
    assert a == b, (
        f"{name} differs between a workstation and a cluster -- the inner "
        "wrapper is supposed to be the same file, so a laptop run is a "
        "rehearsal of the cluster run")


# --------------------------------------------------------------------- #
#  P12 unit 4 -- the five steps run in their order                       #
# --------------------------------------------------------------------- #

# `test_prep_resolves_the_machine_before_it_writes_anything` retired
# 2026-10-06: it spied on the order of two calls below the entry.  The
# entry reads the record at its checkpoint 4 and resolves the stage with it
# as an argument, so no deck can be written against no machine; and a
# refused prep writes nothing (`tests/data/prep_protocol.toml`).


def test_the_library_itself_refuses_a_whole_ladder(tmp_path):
    """U5: the no-chain rule lives at the SEAM, not only in the CLI's
    stage resolution -- a library caller handing the launch entry a
    two-stage ladder with no `only` is refused in EVERY mode, because
    direct-running stages in order would be local chaining
    (project-layout.md § 1.6).  API-LEVEL because the road cannot reach it:
    `launch run` always names one stage."""
    import pytest as _pytest
    from molbuilder.jobset.model import Job, JobSet, Resources
    js = JobSet(name="JOB", engine="siesta", kind="ladder",
                jobs=[Job(name="coarse", script="JOB_01_coarse.fdf",
                          resources=Resources()),
                      Job(name="tight", script="JOB_02_tight.fdf",
                          resources=Resources())])
    for mode in ("direct", "submit"):
        with _pytest.raises(SubmitError, match="ONE stage at a time"):
            plan_launch(js, tmp_path, mode=mode,
                        told=dict(kind="run", stage=None, trial=None,
                                  mode=mode, flags=[]))


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


# --------------------------------------------------------------------- #
#  G7 — the GPU answer travels; the deck is not re-read for it          #
# --------------------------------------------------------------------- #

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
