"""The ``jobset`` framework: model + persistence + validate + materialize
+ plan (docs/protocols/staged-execution.md § 2.1), and the SIESTA
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
    with pytest.raises(ValueError, match="unsupported schema"):
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
    from molbuilder.siesta.stages import stages_to_jobset

    cfg = SiestaConfig()                   # default: stage1+stage2 on, stage3 off
    js = stages_to_jobset(cfg, shared=["C.psml"])
    assert js.kind == "ladder" and js.engine == "siesta"
    assert [j.name for j in js.jobs] == ["stage1", "stage2"]
    assert js.jobs[0].script == "siesta_stage1.fdf"
    # stage2 chains off stage1; stage1 policy "proceed" -> afterany edge.
    assert js.jobs[1].depends_on == "stage1"
    assert js.jobs[1].dep_kind == "afterany"
    # carry: .XV always + .DM (use_save_dm default) ; NO .CG (stage1 CG vs
    # stage2 Broyden -- different optimizer, history not carried).
    patterns = [c.pattern for c in js.jobs[1].carry]
    assert "siesta.XV" in patterns and "siesta.DM" in patterns
    assert "siesta.CG" not in patterns
    assert js.validate() == []             # framework-valid


def test_stages_to_jobset_carries_cg_when_same_relax_type():
    import dataclasses
    from molbuilder.config.siesta import (SiestaConfig,
                                          _default_siesta_stages)
    from molbuilder.siesta.stages import stages_to_jobset

    s = _default_siesta_stages()
    # make stage2 use CG too (same as stage1) -> .CG should carry forward.
    s[1] = dataclasses.replace(s[1], relax_type="CG")
    cfg = SiestaConfig(stages=s)
    js = stages_to_jobset(cfg)
    assert "siesta.CG" in [c.pattern for c in js.jobs[1].carry]


def test_stages_to_jobset_halt_policy_gives_afterok():
    import dataclasses
    from molbuilder.config.siesta import (SiestaConfig,
                                          _default_siesta_stages)
    from molbuilder.siesta.stages import stages_to_jobset

    s = _default_siesta_stages()
    s[0] = dataclasses.replace(s[0], on_nonconvergence="halt")
    cfg = SiestaConfig(stages=s)
    js = stages_to_jobset(cfg)
    assert js.jobs[1].dep_kind == "afterok"


def test_stages_to_jobset_resources_injection():
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.jobset.model import Resources
    from molbuilder.siesta.stages import stages_to_jobset

    overrides = {"stage2": Resources(domain="public", time="7-00:00:00",
                                     exclusive=True)}
    cfg = SiestaConfig()
    js = stages_to_jobset(cfg, resources_for=overrides.get)
    assert js.jobs[1].resources.domain == "public"
    assert js.jobs[1].resources.exclusive is True
    # stage1 (no override) inherits job-level defaults.
    assert js.jobs[0].resources.domain is None


def test_stages_to_jobset_rejects_invalid_ladder():
    import dataclasses
    from molbuilder.config.siesta import (SiestaConfig,
                                          _default_siesta_stages)
    from molbuilder.siesta.stages import stages_to_jobset

    s = _default_siesta_stages()
    s[1] = dataclasses.replace(s[1], name="stage1")   # duplicate name
    cfg = SiestaConfig(stages=s)
    with pytest.raises(ValueError, match="invalid ladder"):
        stages_to_jobset(cfg)


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
    with pytest.raises(SubmitError, match="run prep_jobset first"):
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
    r = runner.invoke(grp, ["prep", str(tmp_path), "--no-sbatch"])
    assert r.exit_code == 0, r.output
    assert "prepped 2 job dir(s)" in r.output
    assert (tmp_path / "point-G1K1C4" / "job-gpu.run.sh").is_symlink()


def test_cli_submit_dry_run_lists_commands(tmp_path):
    _sweep().write(tmp_path / "job-set.json")
    runner, grp = _runner()
    r = runner.invoke(grp, ["submit", str(tmp_path), "--mode", "submit",
                            "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "planned" in r.output and "sbatch" in r.output
    assert "-J" in r.output and "G1K1C4" in r.output


def test_cli_submit_requires_mode(tmp_path):
    _sweep().write(tmp_path / "job-set.json")
    runner, grp = _runner()
    r = runner.invoke(grp, ["submit", str(tmp_path)])
    assert r.exit_code != 0                                # --mode required
