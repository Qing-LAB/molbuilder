"""The ``jobset`` framework: model + validate + materialize
+ plan (docs/execution/job-system.md), and the SIESTA
stage-ladder producer."""

from __future__ import annotations

from pathlib import Path

import pytest

from molbuilder.jobset.model import Job, JobSet, Resources
from molbuilder.paths import trial_name
from molbuilder.jobset.plan import render_plan


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


@pytest.fixture(autouse=True)
def _sandbox(tmp_path_factory, monkeypatch):
    """cwd isolation for EVERY test here (I6, 2026-08-13): the prep/submit
    tests read this machine's config and record, which conftest isolates;
    this keeps their working directory out of the checkout.
    Tests that need their own cwd (monkeypatch.chdir) still win: their
    monkeypatching applies after this fixture's."""
    box = tmp_path_factory.mktemp("sandbox")
    monkeypatch.chdir(box)


# --------------------------------------------------------------------- #
#  model (job-set@1)                                                     #
# --------------------------------------------------------------------- #


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
    # ``_warm_declaration`` derives it, so a test asserting the copy
    # behaviour tests the system's list rather than the fixture's own.
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
    # The clean restart group rides along: the submission
    # door verifies a trial's cold start against its deck, and a real deck
    # always carries the group written out (`siesta/input.py`, 2026-08-18).
    path.write_text("SystemName test\nSystemLabel test\nNumberOfAtoms 2\n"
                    "DM.UseSaveDM .false.\nMD.UseSaveXV .false.\n")


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
    """`describe` is floor 2's only writer: a folder without the task.json +
    template pair is refused with the next
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
    # alone, and nothing was laid out here
    assert not (tmp_path / "bench-G1K1C4").exists()


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
    (`project-layout.md` § 1.6).
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
    # stage resolved through the one seam, warm + traits from the
    # engine's own declarations.
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.identity import stage_token
    from molbuilder.resolve import effective_config
    from molbuilder.siesta.stages import (_traits, _warm_declaration,
                                          default_siesta_stages)
    label = "bdt"
    jobs = []
    for i, st in enumerate(default_siesta_stages("vib-quality"), start=1):
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
    # lowercase: the trait normalizes at its ONE producer so
    # "Broyden" vs "broyden" (one optimizer, two hands) compare equal
    assert [j.traits["optimizer"] for j in js.jobs] == \
        ["cg", "broyden", "broyden"], "the fixture's premise moved"
    return js


def test_the_declaration_is_now_the_only_rendering_of_the_rule():
    """The warm declaration is the one rendering of the warm-start rule.
    -- the plan's item 12c's *"two lists that agree today and
    nothing keeps them agreeing"*, held in step by derivation.
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


# --------------------------------------------------------------------- #
#  P6 unit 1 -- step ONE of the five, lifted out of the benchmark        #
# --------------------------------------------------------------------- #


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
    environment.
    """
    import json
    import shutil
    from conftest import write_machine_record
    from molbuilder.config_dir import ensure_private_dir
    from molbuilder.scheduler import (Domain, Environment, Topology,
                                      environments_dir,
                                      named_environment_path,
                                      write_environment)
    from support.road import describe_calculation, jobset
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
    ws = describe_calculation(tmp_path, monkeypatch, shape="flat")
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


def test_resources_fields_equal_the_contracts_list_exactly():
    """job-contracts § 6.2's OWN SENTENCE and the dataclass, an equality
    in BOTH directions.  This parses the sentence's backticked field names, so
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
    from support.road import describe_calculation, jobset
    bundle = describe_calculation(tmp_path, monkeypatch)
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
