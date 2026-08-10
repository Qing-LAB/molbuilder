"""The ``jobset`` framework: model + persistence + validate + materialize
+ plan (docs/execution/job-system.md), and the SIESTA
stage-ladder producer."""

from __future__ import annotations

import os
import re
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
    r = runner.invoke(grp, ["plan", "--bundle", str(tmp_path)])
    assert r.exit_code == 0, r.output
    assert "JOB-SET PLAN" in r.output and "Order: s1 -> s2" in r.output


def test_cli_errors_without_jobset_json(tmp_path):
    runner, grp = _runner()
    r = runner.invoke(grp, ["plan", "--bundle", str(tmp_path)])
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
    r = runner.invoke(grp, ["status", "--bundle", str(tmp_path)])
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
    from molbuilder.jobset.materialize import job_dir_names
    from molbuilder.jobset.model import Job, JobSet
    js = JobSet(name="JOB", engine="siesta", kind="sweep",
                jobs=[Job(name="p1", script="JOB_p1.fdf")])
    assert job_dir_names(js) == {"p1": "point-p1"}


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
    # happened to hold a `point-p2/` would decide this test's outcome.
    out = render_status(jobset_status(js, tmp_path))
    assert [l.split()[0] for l in out.splitlines() if "p2" in l] == ["-"]


def test_prepare_attempt_takes_the_same_three_spellings_as_every_surface():
    """It had its own lookup and its own refusal until 2026-08-10, so `prep run
    3` failed where `submit run 3` worked -- one question, two vocabularies."""
    import tempfile
    from molbuilder.jobset.materialize import prepare_attempt
    js = _token_ladder("JOB_01_coarse.fdf", "JOB_03_tight.fdf")
    with tempfile.TemporaryDirectory() as td:
        for spelling in ("tight", "3", "03", "03_tight"):
            rep = prepare_attempt(js, td, spelling)
            assert rep["stage"] == "tight"           # the NAME, always
            assert rep["dir"].parent.name == "03_tight"


def test_prepare_attempt_refuses_with_the_one_listing_that_carries_ordinals():
    """decision 28's gap verbatim: the refusal listed 'coarse, medium, tight'
    with no order, at the one moment you are choosing which stage to run."""
    import tempfile
    from molbuilder.jobset.materialize import prepare_attempt
    js = _token_ladder("JOB_01_coarse.fdf", "JOB_03_tight.fdf")
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(ValueError) as e:
            prepare_attempt(js, td, "bogus")
    assert "01_coarse, 03_tight" in str(e.value)


def test_an_unlaunched_attempt_is_reused_and_a_launched_one_is_never_touched():
    """§ 1.5: an attempt is immutable once it has run, so a re-run is a NEW
    directory.  Before it has run there is nothing to preserve, and minting
    run-1 beside an empty run-0 would just litter."""
    import tempfile
    from pathlib import Path
    from molbuilder.jobset.materialize import prepare_attempt, write_run_launch
    js = _token_ladder("JOB_03_tight.fdf")
    with tempfile.TemporaryDirectory() as td:
        first = prepare_attempt(js, td, "tight")["dir"]
        assert first.name == "run-0"
        assert prepare_attempt(js, td, "tight")["dir"] == first   # reused
        write_run_launch(first, mode="direct", command=["bash", "x.sh"])
        second = prepare_attempt(js, td, "tight")["dir"]
        assert second.name == "run-1"                             # never reused
        assert (Path(first) / "run.json").is_file()               # left intact


# --------------------------------------------------------------------- #
#  The observe layer vs the attempt layer (project-layout.md § 1.5, 1.6) #
# --------------------------------------------------------------------- #


def test_status_reads_the_attempt_because_that_is_where_the_run_happened(tmp_path):
    """`project-layout.md` § 1.5, *"Where a run happens: inside the attempt
    directory"* -- so a stage whose output is in run-0 has RUN, and status that
    globs the container reports it as never launched, forever."""
    from molbuilder.jobset.materialize import prepare_attempt
    js = _token_ladder("JOB_03_tight.fdf")
    attempt = prepare_attempt(js, tmp_path, "tight")["dir"]
    (attempt / "JOB_03_tight.out").write_text("Job completed\n")

    st = jobset_status(js, tmp_path).stages[0]
    assert st.attempt == "run-0"                 # says WHICH attempt it read
    # A POSITIVE claim: the decoder was reached and returned one of its own
    # verdicts.  `!= "pending"` would also pass for "unknown", which is what
    # this reports when the decoder THROWS -- a broken decoder would look like
    # a working fix.
    assert st.state in ("running", "finished", "failed", "stale")
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
    attempt = prepare_attempt(js, tmp_path, "tight")["dir"]
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
    attempt = prepare_attempt(js, tmp_path, "tight")["dir"]

    before = jobset_status(js, tmp_path).stages[0]
    assert before.state == "pending"             # prepped, genuinely not launched

    write_run_launch(attempt, mode="submit", command=["sbatch", "x"],
                     job_id="481923")
    after = jobset_status(js, tmp_path).stages[0]
    assert after.state == "queued"
    assert "481923" in after.detail              # the contract's own sentence


def test_re_prepping_cold_removes_what_the_previous_prep_carried_in(tmp_path):
    """§ 1.6 makes re-prep *"changing your mind about the setup"*.  A mind
    changed from `--from A` to `--cold` that leaves A's .XV in the directory has
    changed nothing: the engine finds it and warm-starts anyway.  That is the
    *"present but not honoured"* failure inverted, and it is silent."""
    from molbuilder.jobset.materialize import prepare_attempt
    js = _token_ladder("JOB_01_coarse.fdf", "JOB_03_tight.fdf")
    coarse = prepare_attempt(js, tmp_path, "coarse")["dir"]
    (coarse / "JOB.XV").write_text("geometry from coarse\n")

    warm = prepare_attempt(js, tmp_path, "tight",
                           continue_from="01_coarse/run-0")
    attempt = warm["dir"]
    assert warm["copied"] == ["JOB.XV"]
    assert (attempt / "JOB.XV").is_file()

    cold = prepare_attempt(js, tmp_path, "tight", cold=True)
    assert cold["dir"] == attempt                # the same unlaunched attempt
    assert not (attempt / "JOB.XV").exists()     # and it is actually cold now
    assert not (attempt / ".continued-from").exists()


def test_a_gpu_stage_is_routed_by_its_DECK_not_by_an_unset_gres(tmp_path):
    """`job-contracts.md § 6.2` derives the GPU request *"from `.fdf` + GPU
    type"*, and the halves live apart on purpose: the deck travels with the
    bundle, the GPU **type** is a cluster fact that `job-system.md` decision #3
    (target isolation) keeps out of what you produce on a laptop.

    So the ladder producer leaves `gres` unset and is right to — `stages.py`
    says *"scheduler resources … resolve at submit"*.  What was missing is that
    submit asked `bool(job.resources.gres)`, always false for a ladder, so a
    stage whose deck selects a GPU eigensolver went to the **CPU partition**
    while its own rendered header asked for a GPU.
    """
    from molbuilder.jobset.submit import _job_wants_gpu
    from molbuilder.jobset.model import Job, Resources

    d = tmp_path / "03_tight"; d.mkdir()
    gpu_job = Job(name="tight", script="JOB_03_tight.fdf")
    (d / "JOB_03_tight.fdf").write_text(
        "SystemLabel JOB\nDiag.Algorithm ELPA-2stage\nDiag.ELPA.GPU .true.\n")
    assert _job_wants_gpu(d, gpu_job) is True

    cpu = tmp_path / "01_coarse"; cpu.mkdir()
    cpu_job = Job(name="coarse", script="JOB_01_coarse.fdf")
    (cpu / "JOB_01_coarse.fdf").write_text(
        "SystemLabel JOB\nDiag.Algorithm divide-and-conquer\n")
    assert _job_wants_gpu(cpu, cpu_job) is False

    # a sweep point that STATES its gres is honoured unchanged: the benchmark
    # sweeps a GPU count, which is not a property of one deck
    pt = tmp_path / "point-G1K1C4"; pt.mkdir()
    assert _job_wants_gpu(pt, Job(name="p", script="job-gpu.fdf",
                                  resources=Resources(gres="gpu:a100:1"))) is True


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
    coarse = prepare_attempt(js, tmp_path, "coarse")["dir"]
    (coarse / "JOB.XV").write_text("COARSE-GEOM")
    tight = prepare_attempt(js, tmp_path, "tight",
                            continue_from="01_coarse/run-0")["dir"]
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
    attempt = prepare_attempt(js, tmp_path, "tight", cold=True)["dir"]
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
    coarse = prepare_attempt(js, tmp_path, "coarse")["dir"]
    (coarse / "JOB.XV").write_text("x")
    tight = prepare_attempt(js, tmp_path, "tight",
                            continue_from="01_coarse/run-0")["dir"]
    write_run_launch(tight, mode="direct", command=["bash", "x.sh"],
                     continued_from="01_coarse/run-0")

    body = [l for l in render_stage_status(jobset_status(js, tmp_path),
                                           "tight").splitlines()
            if l.startswith("  ")]
    # every indented row separates its label from its value by real whitespace
    assert body, "no rows rendered"
    for line in body:
        assert re.match(r"^ {2}\S.*?\s{2,}\S", line), f"label runs into value: {line!r}"


def test_plan_and_status_take_the_bundle_the_same_way_every_verb_does(tmp_path):
    """One word cannot mean the folder on two verbs and the stage on two others.
    `jobset status tight` answered *"Directory 'tight' does not exist"* -- a
    complaint about a path the user never meant to type (§ 5.3)."""
    _token_ladder("JOB_01_coarse.fdf", "JOB_03_tight.fdf").write(
        tmp_path / "job-set.json")
    runner, grp = _runner()
    for verb in ("plan", "status"):
        r = runner.invoke(grp, [verb, "--bundle", str(tmp_path)])
        assert r.exit_code == 0, r.output
        assert "coarse" in r.output
    # ...and the positional is a STAGE, resolved the way every other verb
    # resolves one.  A NUMBER, deliberately: an exact name would pass even if
    # the command took the string verbatim and never reached the resolver.
    r = runner.invoke(grp, ["status", "3", "--bundle", str(tmp_path)])
    assert r.exit_code == 0, r.output
    assert r.output.splitlines()[0].startswith("STAGE 03_tight")


def test_a_ladder_refuses_to_submit_all_of_itself_without_chain(tmp_path):
    """`project-layout.md` § 1.6 is the headline rule -- *"Each stage is prepped
    and submitted on its own"* -- and the reason is cost, not tidiness: *"a
    chain that continues on its own can spend a week refining a geometry you
    would have rejected in a minute."*

    It was enforced in `_resolve_stage` and asserted nowhere, which for a rule
    whose whole job is to stop an expensive accident is the wrong way round.
    """
    _token_ladder("JOB_01_coarse.fdf", "JOB_03_tight.fdf").write(
        tmp_path / "job-set.json")
    runner, grp = _runner()

    r = runner.invoke(grp, ["submit", "run", "--bundle", str(tmp_path),
                            "--mode", "direct", "--dry-run"])
    assert r.exit_code != 0
    assert "acts on ONE stage" in r.output
    assert "01_coarse, 03_tight" in r.output      # ordinals, at the moment you choose
    assert "--chain" in r.output                  # and how to say you meant it

    # ...and saying it out loud is accepted.
    r = runner.invoke(grp, ["submit", "run", "--chain", "--bundle",
                            str(tmp_path), "--mode", "direct", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "coarse" in r.output and "tight" in r.output


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
    coarse = prepare_attempt(js, tmp_path, "coarse")["dir"]
    (coarse / "JOB.XV").write_text("COARSE-GEOM")

    tight = prepare_attempt(js, tmp_path, "tight",
                            continue_from="01_coarse/run-0")["dir"]
    carried = tight / "JOB.XV"
    assert not carried.is_symlink(), "carried warm state is a LINK back to it"
    assert carried.read_text() == "COARSE-GEOM"

    carried.write_text("TIGHT-GEOM")            # what the engine does, step 1
    assert (coarse / "JOB.XV").read_text() == "COARSE-GEOM"


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
    """The deck and the package are linked from ``<stage>/run-<n>/`` up to the
    bundle root -- two levels, not one.  A wrong depth is a dangling link, and
    nothing notices until the engine cannot find its input at launch, on the
    cluster, in the queue."""
    from molbuilder.jobset.materialize import prepare_attempt
    from molbuilder.jobset.model import Job, JobSet
    js = JobSet(name="JOB", engine="siesta", kind="ladder",
                shared=["C.psml"], jobs=[Job(name="tight",
                                             script="JOB_03_tight.fdf")])
    for f in ("JOB_03_tight.fdf", "C.psml", "mb_monitor.py",
              "JOB_03_tight.run.sh"):
        (tmp_path / f).write_text("x")

    rep = prepare_attempt(js, tmp_path, "tight")
    attempt = rep["dir"]
    assert set(rep["linked"]) == {"JOB_03_tight.fdf", "C.psml",
                                  "mb_monitor.py", "JOB_03_tight.run.sh"}
    for name in rep["linked"]:
        link = attempt / name
        assert link.is_symlink(), f"{name} was copied, not linked"
        assert link.resolve() == (tmp_path / name).resolve(), \
            f"{name} points at {os.readlink(link)!r}, which does not resolve"


def test_a_name_beats_a_number_when_a_stage_is_called_one(tmp_path):
    """Stage names are ``[A-Za-z0-9_]+``, so a stage may legitimately be named
    ``3``.  The name is the stage's identity (`engines/stages.md` R5), so it
    wins -- the resolver checks names and tokens before it reads anything as an
    ordinal."""
    from molbuilder.identity import StageRef, resolve_stage_ref
    refs = [StageRef(1, "3"), StageRef(3, "tight")]
    assert resolve_stage_ref(refs, "3").name == "3"      # the NAME, seq 1
    assert resolve_stage_ref(refs, "03_tight").name == "tight"
    assert resolve_stage_ref(refs, "tight").seq == 3


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
    coarse = prepare_attempt(js, tmp_path, "coarse")["dir"]
    (coarse / "JOB.XV").write_text("geometry from coarse\n")

    tight = prepare_attempt(js, tmp_path, "tight",
                            continue_from="01_coarse/run-0")["dir"]
    # the wrapper prep would have linked in; submit only launches
    (tight / "JOB_03_tight.run.sh").write_text("#!/bin/bash\nexit 0\n")
    submit_jobset(js, tmp_path, mode="direct", only="tight")

    body = json.loads((tight / "run.json").read_text())
    assert body["continued_from"] == "01_coarse/run-0"
    assert body["mode"] == "direct"


def test_submit_refuses_an_attempt_that_has_already_been_launched(tmp_path):
    """§ 1.5: *"A run directory is written once and never modified."*  Without
    this refusal a second submit runs the engine straight into the results of
    the first -- the one thing an attempt directory exists to make impossible.
    """
    from molbuilder.jobset.materialize import prepare_attempt
    from molbuilder.jobset.submit import submit_jobset, SubmitError
    js = _token_ladder("JOB_03_tight.fdf")
    attempt = prepare_attempt(js, tmp_path, "tight")["dir"]
    (attempt / "JOB_03_tight.run.sh").write_text("#!/bin/bash\nexit 0\n")
    submit_jobset(js, tmp_path, mode="direct", only="tight")
    (attempt / "JOB_03_tight.out").write_text("results of the first run\n")

    with pytest.raises(SubmitError) as e:
        submit_jobset(js, tmp_path, mode="direct", only="tight")
    assert "already been launched" in str(e.value)
    assert "prep run tight" in str(e.value)      # says how to get a fresh one
    assert (attempt / "JOB_03_tight.out").read_text().startswith("results")


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
    attempt = prepare_attempt(js, tmp_path, "tight")["dir"]
    (attempt / "run.json").write_text('{"schema": "molbuilder/run-la')

    st = jobset_status(js, tmp_path).stages[0]
    assert st.state == "queued"                  # launched, details lost


def test_submit_only_takes_the_same_three_spellings_as_every_surface(tmp_path):
    """`only` is a library entry point, and it had its own lookup and its own
    listing -- the same defect prepare_attempt had (§ 8f)."""
    from molbuilder.jobset.submit import submit_jobset, SubmitError
    js = _token_ladder("JOB_01_coarse.fdf", "JOB_03_tight.fdf")
    res = submit_jobset(js, tmp_path, mode="direct", dry_run=True, only="3")
    assert [r.name for r in res] == ["tight"]

    with pytest.raises(SubmitError) as e:
        submit_jobset(js, tmp_path, mode="direct", dry_run=True, only="bogus")
    assert "01_coarse, 03_tight" in str(e.value)


def test_a_number_resolves_to_the_seq_and_never_to_the_row():
    """R5's whole point, and the two differ the moment a stage is disabled: the
    ladder is 01/03, so `1` is coarse (row 1 would be tight) and `3` is tight
    (there is no row 3 at all)."""
    from molbuilder.identity import StageRef, resolve_stage_ref
    refs = [StageRef(1, "coarse"), StageRef(3, "tight")]
    assert resolve_stage_ref(refs, "1").name == "coarse"
    assert resolve_stage_ref(refs, "3").name == "tight"
    assert resolve_stage_ref(refs, "03").name == "tight"        # zero-padded
    assert resolve_stage_ref(refs, "03_tight").name == "tight"  # the whole token
    assert resolve_stage_ref(refs, "tight").name == "tight"     # its identity
    with pytest.raises(ValueError):
        resolve_stage_ref(refs, "2")            # the row of tight; not its seq


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
    js = _token_ladder("JOB_01_coarse.fdf", "JOB_03_tight.fdf")
    st = jobset_status(js, tmp_path)
    assert [s.seq for s in st.stages] == [1, 3]
    out = render_status(st)
    assert "seq" in out
    assert [l.split()[0] for l in out.splitlines() if "tight" in l] == ["3"]


def test_status_seq_is_none_for_a_sweep_point():
    """A sweep point has no order, so it has no seq -- and says so rather than
    borrowing a row number's authority."""
    from molbuilder.jobset.model import Job, JobSet
    js = JobSet(name="JOB", engine="siesta", kind="sweep",
                jobs=[Job(name="p1", script="JOB.fdf")])
    assert jobset_status(js, ".").stages[0].seq is None
