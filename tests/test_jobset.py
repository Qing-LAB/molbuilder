"""The ``jobset`` framework: model + persistence + validate + materialize
+ plan (docs/execution/job-system.md), and the SIESTA
stage-ladder producer."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from molbuilder.jobset.model import (Carry, Job, JobSet, Resources,
                                     SCHEMA)
from molbuilder.jobset.materialize import job_dir_name, materialize
from molbuilder.jobset.plan import render_plan
from molbuilder.jobset import submit as _submit
from molbuilder.jobset.submit import submit_jobset, SubmitError
from molbuilder.jobset.prep import prep_jobset
from molbuilder.jobset.runstatus import jobset_status, render_status


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
                resources=Resources(domain="htc", time="0-04:00:00")),
            Job(name="s2", script="demo_s2.fdf",
                resources=Resources(domain="public", exclusive=True),
                depends_on="s1", dep_kind="afterok",
                carry=[Carry("demo.XV", "s1"), Carry("demo.DM", "s1")]),
        ],
    )


def test_jobset_roundtrips_through_job_set_at_1():
    js = _ladder()
    d = js.to_dict()
    assert d["schema"] == SCHEMA
    back = JobSet.from_dict(d)
    assert back.to_dict() == d           # lossless
    assert back.jobs[1].carry[0].pattern == "demo.XV"
    assert back.jobs[1].resources.exclusive is True


def test_from_dict_rejects_unknown_schema():
    d = _ladder().to_dict()
    d["schema"] = "job-set@99"
    with pytest.raises(ValueError, match="schema mismatch"):
        JobSet.from_dict(d)


def test_validate_passes_clean_ladder():
    assert _ladder().validate() == []


def test_validate_catches_duplicate_names():
    js = _ladder()
    js.jobs[1].name = "s1"
    assert any("duplicate" in e for e in js.validate())


def test_validate_catches_forward_dependency():
    js = _ladder()
    js.jobs[0].depends_on = "s2"          # references a LATER job
    assert any("PRIOR job" in e for e in js.validate())


def test_validate_catches_bad_dep_kind_and_carry():
    js = _ladder()
    js.jobs[1].dep_kind = "afterbogus"
    js.jobs[1].carry = [Carry("x.XV", "nope")]
    errs = js.validate()
    assert any("dep_kind" in e for e in errs)
    assert any("prior job" in e for e in errs)


def test_validate_catches_empty_and_bad_kind():
    assert any("empty" in e for e in
               JobSet("n", "siesta", "ladder", jobs=[]).validate())
    assert any("kind" in e for e in
               JobSet("n", "siesta", "bogus",
                      jobs=[Job("a", "a.fdf")]).validate())


# --------------------------------------------------------------------- #
#  materialize engine                                                    #
# --------------------------------------------------------------------- #

def test_materialize_creates_dirs_and_symlinks(tmp_path):
    js = _ladder()
    # lay the shared package + the per-job scripts in the bundle root.
    for f in js.shared + [j.script for j in js.jobs]:
        (tmp_path / f).write_text("x")
    dirs = materialize(js, tmp_path)
    assert [d.name for d in dirs] == ["point-s1", "point-s2"]
    # shared package symlinked into each job dir (relative).
    link = tmp_path / "point-s1" / "C.psml"
    assert link.is_symlink() and os.readlink(link) == os.path.join("..", "C.psml")
    # the job's own input symlinked in.
    assert (tmp_path / "point-s2" / "demo_s2.fdf").is_symlink()


def test_materialize_carry_symlink_points_at_producer_dir(tmp_path):
    js = _ladder()
    for f in js.shared + [j.script for j in js.jobs]:
        (tmp_path / f).write_text("x")
    materialize(js, tmp_path)
    xv = tmp_path / "point-s2" / "demo.XV"
    # carry symlink exists and targets the producer (s1) dir -- dangling
    # is fine (s1 hasn't "run" yet).
    assert xv.is_symlink()
    assert os.readlink(xv) == os.path.join("..", "point-s1", "demo.XV")
    assert not xv.exists()                 # dangling until s1 produces it


def test_materialize_is_idempotent(tmp_path):
    js = _ladder()
    for f in js.shared + [j.script for j in js.jobs]:
        (tmp_path / f).write_text("x")
    materialize(js, tmp_path)
    materialize(js, tmp_path)              # no exception, no duplication
    assert (tmp_path / "point-s2" / "demo.XV").is_symlink()


def test_materialize_rejects_invalid_jobset(tmp_path):
    js = _ladder()
    js.jobs[1].name = "s1"                 # duplicate
    with pytest.raises(ValueError, match="invalid JobSet"):
        materialize(js, tmp_path)


def test_job_dir_name():
    assert job_dir_name("stage1") == "point-stage1"


# --------------------------------------------------------------------- #
#  job_dir_names -- two kinds, two conventions (project-layout.md § 4.1) #
# --------------------------------------------------------------------- #


def _token_ladder(*scripts):
    from molbuilder.jobset.model import Job, JobSet
    return JobSet(name="JOB", engine="siesta", kind="ladder",
                  jobs=[Job(name=s.split("_", 2)[2].rsplit(".", 1)[0],
                            script=s) for s in scripts])


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
    """The benchmark is untouched by the ladder's rule."""
    from molbuilder.jobset.materialize import job_dir_names
    from molbuilder.jobset.model import Job, JobSet
    js = JobSet(name="JOB", engine="siesta", kind="sweep",
                jobs=[Job(name="np4", script="JOB.fdf"),
                      Job(name="np8", script="JOB.fdf")])
    assert job_dir_names(js) == {"np4": "point-np4", "np8": "point-np8"}


def test_job_dir_names_ladder_without_a_token_falls_back_rather_than_guessing():
    """A hand-written ladder whose deck carries no token gets ``point-<name>``.
    Inventing a seq for it would be guessing at the one number § 4.2 says is
    assigned once and never reassigned."""
    from molbuilder.jobset.materialize import job_dir_names
    from molbuilder.jobset.model import Job, JobSet
    js = JobSet(name="JOB", engine="siesta", kind="ladder",
                jobs=[Job(name="only", script="JOB.fdf")])
    assert job_dir_names(js) == {"only": "point-only"}


# --------------------------------------------------------------------- #
#  plan engine                                                          #
# --------------------------------------------------------------------- #

def test_render_plan_shows_deps_carries_and_order():
    txt = render_plan(_ladder())
    assert "JOB-SET PLAN -- demo (siesta, ladder)" in txt
    assert "C.psml" in txt                 # shared package
    assert "s1 (afterok)" in txt           # dependency + kind
    assert "demo.XV" in txt                # carry
    assert "Order: s1 -> s2" in txt        # chain


def test_render_plan_sweep_says_independent():
    js = JobSet("sweep", "siesta", "sweep",
                jobs=[Job("a", "a.fdf"), Job("b", "b.fdf")])
    assert "independent" in render_plan(js)


# --------------------------------------------------------------------- #
#  SIESTA stage producer                                                #
# --------------------------------------------------------------------- #

def test_stages_to_jobset_default_ladder():
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.siesta.stages import (DEFAULT_NONCONVERGENCE,
                                          default_siesta_stages,
                                          stages_to_jobset)

    cfg = SiestaConfig()                   # the template -- no ladder in it
    stages = default_siesta_stages()       # coarse+medium on, tight off
    js = stages_to_jobset(cfg, stages, shared=["C.psml"],
                          on_nonconvergence=DEFAULT_NONCONVERGENCE)
    assert js.kind == "ladder" and js.engine == "siesta"
    # The JOB keeps the stage's NAME; its SCRIPT carries the artifact
    # token, because that is the file the renderer wrote (decision 27).
    assert [j.name for j in js.jobs] == ["coarse", "medium"]
    assert js.jobs[0].script == "siesta_01_coarse.fdf"
    assert js.jobs[1].script == "siesta_02_medium.fdf"
    # medium chains off coarse; coarse policy "proceed" -> afterany edge.
    assert js.jobs[1].depends_on == "coarse"
    assert js.jobs[1].dep_kind == "afterany"
    # carry: .XV always + .DM (use_save_dm default) ; NO .CG (coarse CG vs
    # medium Broyden -- different optimizer, history not carried).
    patterns = [c.pattern for c in js.jobs[1].carry]
    assert "siesta.XV" in patterns and "siesta.DM" in patterns
    assert "siesta.CG" not in patterns
    assert js.validate() == []             # framework-valid


def test_stages_to_jobset_carries_cg_when_same_relax_type():
    """The comparison is over the RESOLVED optimizer, not a stage field:
    a stage that does not override ``relax_type`` has the template's."""
    import dataclasses
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.siesta.stages import (default_siesta_stages,
                                          stages_to_jobset)

    stages = default_siesta_stages()
    # make medium use CG too (same as coarse) -> .CG should carry forward.
    stages[1] = dataclasses.replace(
        stages[1], overrides={**stages[1].overrides, "relax_type": "CG"})
    js = stages_to_jobset(SiestaConfig(), stages)
    assert "siesta.CG" in [c.pattern for c in js.jobs[1].carry]


def test_stages_to_jobset_carries_cg_when_neither_stage_overrides_it():
    """Both stages inherit the template's optimizer, so they match -- a
    case the old field-comparison could not even express, since every
    stage carried its own ``relax_type`` whether or not it meant to."""
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.siesta.stages import stages_to_jobset
    from molbuilder.task import Stage

    # `b` must SAY it continues.  Before P3 unit 4 a bare stage carried state
    # anyway, because the carry keyed on the template's use_save_dm, which
    # defaulted True -- so this fixture used to pass without stating the one
    # thing it depends on.  The optimizer inheritance under test is unchanged:
    # neither stage overrides relax_type.
    js = stages_to_jobset(SiestaConfig(relax_type="Broyden"),
                          [Stage(name="a"),
                           Stage(name="b", overrides={"restart": "continue"})])
    assert "siesta.CG" in [c.pattern for c in js.jobs[1].carry]


def test_stages_to_jobset_halt_policy_gives_afterok():
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.siesta.stages import (default_siesta_stages,
                                          stages_to_jobset)

    js = stages_to_jobset(SiestaConfig(), default_siesta_stages(),
                          on_nonconvergence={"coarse": "halt"})
    assert js.jobs[1].dep_kind == "afterok"


def test_stages_to_jobset_defaults_an_unnamed_stage_to_afterok():
    """Silence in the policy input means halt, and halt means the next
    stage runs only on success -- the safe reading."""
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.siesta.stages import (default_siesta_stages,
                                          stages_to_jobset)

    js = stages_to_jobset(SiestaConfig(), default_siesta_stages())
    assert js.jobs[1].dep_kind == "afterok"


def test_stages_to_jobset_resources_injection():
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.jobset.model import Resources
    from molbuilder.siesta.stages import stages_to_jobset

    from molbuilder.siesta.stages import default_siesta_stages

    overrides = {"medium": Resources(domain="public", time="7-00:00:00",
                                     exclusive=True)}
    js = stages_to_jobset(SiestaConfig(), default_siesta_stages(),
                          resources_for=overrides.get)
    assert js.jobs[1].resources.domain == "public"
    assert js.jobs[1].resources.exclusive is True
    # coarse (no override) inherits job-level defaults.
    assert js.jobs[0].resources.domain is None


def test_stages_to_jobset_rejects_invalid_ladder():
    import dataclasses
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.siesta.stages import (default_siesta_stages,
                                          stages_to_jobset)

    stages = default_siesta_stages()
    stages[1] = dataclasses.replace(stages[1], name="coarse")  # duplicate
    with pytest.raises(ValueError, match="collide|silently"):
        stages_to_jobset(SiestaConfig(), stages)


def test_stages_to_jobset_rejects_an_override_the_schema_has_no_field_for():
    """The refusal arrives BEFORE any Job is built, and names the field --
    the preflight rule applied at the producer (stages.md § 6.6)."""
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.siesta.stages import stages_to_jobset
    from molbuilder.task import Stage

    with pytest.raises(ValueError, match="mesh_cutof"):
        stages_to_jobset(SiestaConfig(),
                         [Stage(name="a", overrides={"mesh_cutof": 300})])


def test_stages_to_jobset_carries_continue_retries_into_resources():
    """job-contracts.md § 6.2: the warm-retry budget rides Resources under
    its own name.  It is the one field there that becomes no SLURM flag --
    the wrapper bakes it in (running-a-job.md § 3.5)."""
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.siesta.stages import (default_siesta_stages,
                                          stages_to_jobset)

    js = stages_to_jobset(SiestaConfig(continue_retries=3),
                          default_siesta_stages())
    assert [j.resources.continue_retries for j in js.jobs] == [3, 3]


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
                                      gres="gpu:a100:1")),
              Job(name="G1K2C4", script="job-gpu.fdf",
                  resources=Resources(mpi_np=2, cpus_per_task=4,
                                      gres="gpu:a100:1"))])


def _write_fdf(path):
    # minimal .fdf so write_run_wrapper renders for real (bash -n validated).
    path.write_text("SystemName test\nSystemLabel test\nNumberOfAtoms 2\n")


def _write_config(root):
    # a bundle carries the script_generation block the wrapper needs.
    (root / ".molbuilder.json").write_text(
        '{"script_generation": {"preamble": "module load mamba", '
        '"activation": "source activate"}}')


def test_prep_renders_real_wrappers_into_each_job_dir(tmp_path):
    # THE integration seam the old stubbed tests never exercised (F1): the
    # script is symlinked into the job dir, but the wrapper must NOT end up at
    # the resolved bundle-root target -- it must land in the job dir.
    js = _sweep()
    _write_config(tmp_path)
    _write_fdf(tmp_path / "job-gpu.fdf")
    prep_jobset(js, tmp_path, env="molbuilder-siesta-gpu", emit_sbatch=False)
    # rendered ONCE in the bundle root, from the real file.
    assert (tmp_path / "job-gpu.run.sh").is_file()
    # symlinked into BOTH job dirs (one shared wrapper, the bench model).
    for name in ("point-G1K1C4", "point-G1K2C4"):
        link = tmp_path / name / "job-gpu.run.sh"
        assert link.is_symlink() and os.readlink(link) == "../job-gpu.run.sh"
        assert link.resolve() == (tmp_path / "job-gpu.run.sh").resolve()


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
                          resources=Resources(mpi_np=1, continue_retries=3))])
    _write_config(tmp_path)
    _write_fdf(tmp_path / "job.fdf")
    prep_jobset(js, tmp_path, env="molbuilder-siesta", emit_sbatch=False)

    wrapper = (tmp_path / "job.run.sh").read_text()
    assert "_siesta_retry_max=3" in wrapper, wrapper
    # and the wrapper SAYS so to the person reading its banner
    assert "3" in wrapper and "etry" in wrapper


def test_prep_omits_the_retry_loop_when_no_budget_is_asked_for(tmp_path):
    """The other half: absent means absent.  A wrapper that always carried a
    retry loop would re-enter SIESTA for jobs nobody asked to retry."""
    js = JobSet(name="lad", engine="siesta", kind="ladder",
                jobs=[Job(name="tight", script="job.fdf",
                          resources=Resources(mpi_np=1))])
    _write_config(tmp_path)
    _write_fdf(tmp_path / "job.fdf")
    prep_jobset(js, tmp_path, env="molbuilder-siesta", emit_sbatch=False)
    assert "_siesta_retry_max=" not in (tmp_path / "job.run.sh").read_text()


def test_prep_rejects_missing_script(tmp_path):
    from molbuilder.jobset.prep import PrepError
    with pytest.raises(PrepError, match="not in bundle root"):
        prep_jobset(_sweep(), tmp_path, emit_sbatch=False)


def test_render_plan_surfaces_per_job_ranks_and_cores():
    # the plan MUST show the -n/-c variation -- that IS the sweep.
    txt = render_plan(_sweep())
    assert "n=1" in txt and "n=2" in txt
    assert "c=4" in txt and "gpu:a100:1" in txt


# --------------------------------------------------------------------- #
#  submit engine                                                        #
# --------------------------------------------------------------------- #

def test_submit_dry_run_threads_dependency_and_emits_J(tmp_path):
    # ladder, SLURM, dry-run: threaded --dependency + per-job -J, no files.
    res = submit_jobset(_ladder(), tmp_path, mode="submit", dry_run=True)
    assert [r.status for r in res] == ["planned", "planned"]
    assert res[0].command[0] == "sbatch"
    assert res[0].command[res[0].command.index("-J") + 1] == "s1"
    dep = [a for a in res[1].command if a.startswith("--dependency=")]
    assert dep == ["--dependency=afterok:<s1>"]
    assert list(tmp_path.iterdir()) == []          # wrote nothing


def test_submit_dry_run_sweep_per_job_flags_vary(tmp_path):
    # the F2 fix: a SHARED-script sweep must still get per-job -n via CLI.
    res = submit_jobset(_sweep(), tmp_path, mode="submit", dry_run=True)
    for r in res:
        assert not any(a.startswith("--dependency=") for a in r.command)
        assert "--gres=gpu:a100:1" in r.command
    assert res[0].command[res[0].command.index("-n") + 1] == "1"
    assert res[1].command[res[1].command.index("-n") + 1] == "2"   # varies


def test_submit_slurm_parses_ids_and_threads_real_dep(tmp_path, monkeypatch):
    js = _ladder()
    for d in (tmp_path / "point-s1", tmp_path / "point-s2"):
        d.mkdir()
    (tmp_path / "point-s1" / "demo_s1.sbatch").write_text("x")
    (tmp_path / "point-s2" / "demo_s2.sbatch").write_text("x")
    ids = iter(["111", "222"])
    monkeypatch.setattr(_submit.subprocess, "run",
                        lambda *a, **k: _CP(stdout=f"Submitted batch job {next(ids)}"))
    res = submit_jobset(js, tmp_path, mode="submit")
    assert res[0].job_id == "111" and res[0].status == "submitted"
    # the second sbatch threads the REAL producer id, not the symbolic ref.
    assert "--dependency=afterok:111" in res[1].command
    assert res[1].job_id == "222"


def test_submit_slurm_errors_when_not_prepped(tmp_path):
    # real run (not dry): a missing wrapper is a friendly error, not a crash.
    (tmp_path / "point-s1").mkdir()
    with pytest.raises(SubmitError, match="prep first"):
        submit_jobset(_ladder(), tmp_path, mode="submit")


def test_submit_slurm_raises_on_sbatch_failure(tmp_path, monkeypatch):
    js = _ladder()
    (tmp_path / "point-s1").mkdir()
    (tmp_path / "point-s1" / "demo_s1.sbatch").write_text("x")
    monkeypatch.setattr(_submit.subprocess, "run",
                        lambda *a, **k: _CP(returncode=1, stderr="boom"))
    with pytest.raises(SubmitError, match="sbatch failed"):
        submit_jobset(js, tmp_path, mode="submit")


def test_run_direct_afterok_skips_dependent_after_failure(tmp_path, monkeypatch):
    js = _ladder()                                 # s2 --afterok--> s1
    for d in (tmp_path / "point-s1", tmp_path / "point-s2"):
        d.mkdir()
    (tmp_path / "point-s1" / "demo_s1.run.sh").write_text("x")
    (tmp_path / "point-s2" / "demo_s2.run.sh").write_text("x")
    # s1 fails -> afterok edge means s2 must be SKIPPED, never executed.
    monkeypatch.setattr(_submit.subprocess, "run",
                        lambda *a, **k: _CP(returncode=2))
    res = submit_jobset(js, tmp_path, mode="direct")
    assert res[0].status == "failed"
    assert res[1].status == "skipped" and res[1].returncode is None


def test_submit_direct_dry_run_passes_np_omp(tmp_path):
    res = submit_jobset(_sweep(), tmp_path, mode="direct", dry_run=True)
    assert res[0].command[0] == "bash"
    assert "-np" in res[0].command and "-omp" in res[0].command


def test_submit_direct_rejects_domain(tmp_path):
    with pytest.raises(SubmitError, match="no meaning in 'direct'"):
        submit_jobset(_sweep(), tmp_path, mode="direct", domain="htc",
                      dry_run=True)


def test_submit_unknown_mode_and_invalid_jobset(tmp_path):
    with pytest.raises(SubmitError, match="unknown mode"):
        submit_jobset(_sweep(), tmp_path, mode="bogus", dry_run=True)
    bad = _ladder()
    bad.jobs[1].name = "s1"                         # duplicate
    with pytest.raises(SubmitError, match="invalid JobSet"):
        submit_jobset(bad, tmp_path, mode="submit", dry_run=True)


def test_submit_exclusive_suppresses_mem(tmp_path):
    js = JobSet("x", "siesta", "sweep",
                jobs=[Job("j", "j.fdf",
                          resources=Resources(exclusive=True, mem="120G"))])
    cmd = submit_jobset(js, tmp_path, mode="submit", dry_run=True)[0].command
    assert "--exclusive" in cmd
    assert not any(a.startswith("--mem") for a in cmd)   # exclusive wins


# --------------------------------------------------------------------- #
#  persistence (job-set.json write/load)                                #
# --------------------------------------------------------------------- #

def test_jobset_write_load_roundtrip(tmp_path):
    js = _ladder()
    p = js.write(tmp_path / "job-set.json")
    assert p.is_file()
    assert JobSet.load(p).to_dict() == js.to_dict()      # lossless on disk


# --------------------------------------------------------------------- #
#  molbuilder jobset CLI (plan / prep / submit over a bundle)           #
# --------------------------------------------------------------------- #

def _runner():
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    return CliRunner(), jobset_group


def test_cli_plan_reads_jobset_json(tmp_path):
    _ladder().write(tmp_path / "job-set.json")
    runner, grp = _runner()
    r = runner.invoke(grp, ["plan", str(tmp_path)])
    assert r.exit_code == 0, r.output
    assert "JOB-SET PLAN" in r.output and "Order: s1 -> s2" in r.output


def test_cli_errors_without_jobset_json(tmp_path):
    runner, grp = _runner()
    r = runner.invoke(grp, ["plan", str(tmp_path)])
    assert r.exit_code != 0
    assert "no job-set.json" in r.output


def test_cli_prep_lays_out_dirs(tmp_path):
    js = _sweep()
    js.write(tmp_path / "job-set.json")
    _write_config(tmp_path)
    _write_fdf(tmp_path / "job-gpu.fdf")
    runner, grp = _runner()
    # Grammar: `jobset <verb> <kind> [<stage>]` (job-system.md § 5.3).  The
    # bundle moved to --bundle because a session runs from inside the
    # calculation folder; the KIND is a positional because `prep bench` and
    # `prep run` are peers.
    r = runner.invoke(grp, ["prep", "run", "--bundle", str(tmp_path),
                            "--no-sbatch"])
    assert r.exit_code == 0, r.output
    assert "prepped 2 job dir(s)" in r.output
    assert (tmp_path / "point-G1K1C4" / "job-gpu.run.sh").is_symlink()


def test_cli_submit_dry_run_lists_commands(tmp_path):
    _sweep().write(tmp_path / "job-set.json")
    runner, grp = _runner()
    # A SWEEP needs no stage: its points are independent, so the whole set is
    # the ordinary thing.  Only a ladder requires a stage or --chain.
    r = runner.invoke(grp, ["submit", "run", "--bundle", str(tmp_path),
                            "--mode", "submit", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "planned" in r.output and "sbatch" in r.output
    assert "-J" in r.output and "G1K1C4" in r.output


def test_cli_submit_requires_mode(tmp_path):
    _sweep().write(tmp_path / "job-set.json")
    runner, grp = _runner()
    r = runner.invoke(grp, ["submit", "run", "--bundle", str(tmp_path)])
    assert r.exit_code != 0                                # --mode required


# --------------------------------------------------------------------- #
#  runstatus (the inform layer)                                          #
# --------------------------------------------------------------------- #

def _fake_decoder(states):
    """Return a decode_run_dir stand-in: dir-name -> state."""
    class _R:
        def __init__(self, state):
            self.status = {"state": state, "detail": state}
    def fake(run_dir):
        return _R(states[run_dir.name])
    return fake


def test_status_fresh_bundle_all_not_started(tmp_path):
    st = jobset_status(_ladder(), tmp_path)        # nothing prepped
    assert [s.state for s in st.stages] == ["not-started", "not-started"]
    assert st.first_incomplete == "s1" and st.complete is False


def test_status_pending_and_warm_files(tmp_path):
    (tmp_path / "point-s1").mkdir()
    (tmp_path / "point-s1" / "demo.XV").write_text("x")   # label = jobset.name
    st = jobset_status(_ladder(), tmp_path)
    assert st.stages[0].state == "pending"                # dir, no .out
    assert "demo.XV" in st.stages[0].warm_files


def test_status_first_incomplete_advances(tmp_path, monkeypatch):
    import molbuilder.parse.dirs.job as jobmod
    for n in ("point-s1", "point-s2"):
        d = tmp_path / n; d.mkdir(); (d / "demo.out").write_text("x")
    monkeypatch.setattr(jobmod, "decode_run_dir",
                        _fake_decoder({"point-s1": "finished",
                                       "point-s2": "running"}))
    st = jobset_status(_ladder(), tmp_path)
    assert st.stages[0].state == "finished"
    assert st.first_incomplete == "s2" and st.complete is False


def test_status_complete_when_all_finished(tmp_path, monkeypatch):
    import molbuilder.parse.dirs.job as jobmod
    for n in ("point-s1", "point-s2"):
        d = tmp_path / n; d.mkdir(); (d / "demo.out").write_text("x")
    monkeypatch.setattr(jobmod, "decode_run_dir",
                        _fake_decoder({"point-s1": "finished",
                                       "point-s2": "finished"}))
    st = jobset_status(_ladder(), tmp_path)
    assert st.complete is True and st.first_incomplete is None
    assert "All stages finished" in render_status(st)


def test_render_status_shows_resume_pointer(tmp_path):
    txt = render_status(jobset_status(_ladder(), tmp_path))
    assert "JOB-SET STATUS -- demo" in txt
    assert "First incomplete stage: s1" in txt
    assert "does NOT auto-resume" in txt


def test_cli_status(tmp_path):
    _ladder().write(tmp_path / "job-set.json")
    runner, grp = _runner()
    r = runner.invoke(grp, ["status", str(tmp_path)])
    assert r.exit_code == 0, r.output
    assert "JOB-SET STATUS" in r.output and "First incomplete" in r.output


def test_status_finished_with_real_siesta_out(tmp_path):
    # DEPTH: the real decode_run_dir -> "finished" path (not monkeypatched),
    # so a drift in the status-dict shape between decode + runstatus is caught.
    import shutil
    fix = (Path(__file__).parent / "watch" / "fixtures" / "siesta_frozen"
           / "hemeC-stage2-run3-finished-42fr.out")
    d = tmp_path / "point-s1"; d.mkdir()
    shutil.copy(fix, d / "demo.out")            # label = jobset.name = "demo"
    st = jobset_status(_ladder(), tmp_path)
    assert st.stages[0].state == "finished"     # REAL parse of a finished run


# --------------------------------------------------------------------- #
#  carry-forward BEHAVIOR (the §4 isolation guarantee, end-result)       #
# --------------------------------------------------------------------- #

def test_carry_deref_localizes_and_isolates_producer(tmp_path):
    # DEPTH: run the ACTUAL deref bash the generated wrapper emits, and prove
    # the §4 guarantee holds -- the consumer gets a real local copy and a
    # later write to it does NOT reach back and clobber the producer.
    import os
    import subprocess
    from molbuilder.runwrap import render_run_wrapper

    (tmp_path / "point-s1").mkdir()
    (tmp_path / "point-s1" / "JOB.XV").write_text("STAGE1-GEOM")
    c = tmp_path / "point-s2"; c.mkdir()
    os.symlink("../point-s1/JOB.XV", c / "JOB.XV")     # as materialize lays it
    assert (c / "JOB.XV").is_symlink()

    (tmp_path / ".molbuilder.json").write_text(
        '{"script_generation": {"preamble": "x", "activation": "source activate"}}')
    fdf = tmp_path / "JOB_s2.fdf"
    fdf.write_text("SystemLabel JOB\nNumberOfAtoms 1\n")
    txt = render_run_wrapper(fdf, carry_in=["JOB.XV"])
    start = txt.index("# --- Carry-forward")
    block = "_log(){ :; }\n" + txt[start:txt.index("\n\n", start)]
    subprocess.run(["bash", "-eu", "-c", block], cwd=str(c), check=True)

    # localized to a REAL file holding the producer's content
    assert not (c / "JOB.XV").is_symlink()
    assert (c / "JOB.XV").read_text() == "STAGE1-GEOM"
    # this stage writes its OWN geometry -> producer untouched (§4 holds)
    (c / "JOB.XV").write_text("STAGE2-GEOM")
    assert (tmp_path / "point-s1" / "JOB.XV").read_text() == "STAGE1-GEOM"


def test_prep_carry_deref_only_in_consumer_wrapper(tmp_path):
    # the deref preamble appears ONLY in the carrying stage's wrapper.
    js = _ladder()                               # s1: no carry; s2: carries .XV/.DM
    _write_config(tmp_path)
    for s in ("demo_s1.fdf", "demo_s2.fdf"):
        _write_fdf(tmp_path / s)
    prep_jobset(js, tmp_path, emit_sbatch=False)
    s1 = (tmp_path / "demo_s1.run.sh").read_text()
    s2 = (tmp_path / "demo_s2.run.sh").read_text()
    assert "Carry-forward: localize" not in s1
    assert "Carry-forward: localize" in s2
    assert "demo.XV" in s2 and "demo.DM" in s2


def test_prep_writes_stage_plan_md(tmp_path):
    """J1 (D3): prep emits STAGE-PLAN.md into the bundle (bench parity)."""
    js = _sweep()
    _write_config(tmp_path)
    _write_fdf(tmp_path / "job-gpu.fdf")
    prep_jobset(js, tmp_path, emit_sbatch=False)
    plan = tmp_path / "STAGE-PLAN.md"
    assert plan.is_file()
    assert "JOB-SET PLAN" in plan.read_text()
