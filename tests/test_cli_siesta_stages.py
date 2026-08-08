"""Tests for the SIESTA multi-stage CLI surface (task #542, commit 3):
``molbuilder fdf ... --stages-json`` and ``--stage-strategy``.

Pins:
  * either flag triggers the multi-stage branch (N fdfs + 1 runner)
  * neither flag preserves the single-fdf behaviour
  * --stage and --stages-json / --stage-strategy are mutually exclusive
  * --stages-json accepts literal JSON or a path
  * SystemLabel inside every emitted fdf == ``Path(fdf_path).stem``
  * runner has chmod +x
  * runner's STAGES array matches the emitted fdf set
  * bad JSON gives a clean Click error (not a stack trace)
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest
from click.testing import CliRunner

from molbuilder.cli import cli


# A 3-D (non-planar) molecule so the derived vacuum cell isn't degenerate at
# the default vacuum=0 (a linear/planar molecule has a zero-thickness axis --
# structure-periodicity.md).  These tests exercise stage/jobset mechanics, not
# geometry, so any real 3-D structure works; methane is the simplest.
_XYZ = textwrap.dedent("""\
    5

    C  0.000  0.000  0.000
    H  0.629  0.629  0.629
    H -0.629 -0.629  0.629
    H -0.629  0.629 -0.629
    H  0.629 -0.629 -0.629
""")


@pytest.fixture
def xyz(tmp_path):
    p = tmp_path / "ch4.xyz"
    p.write_text(_XYZ)
    return p


def _invoke(*args):
    return CliRunner().invoke(cli, list(args), catch_exceptions=False)


# --------------------------------------------------------------------- #
#  Multi-stage branch entry                                              #
# --------------------------------------------------------------------- #


def test_stage_strategy_publishable_emits_two_stage_bundle(xyz, tmp_path):
    fdf = tmp_path / "JOB.fdf"
    r = _invoke("fdf", str(xyz), str(fdf),
                "--stage-strategy", "publishable")
    assert r.exit_code == 0, r.output
    files = sorted(p.name for p in tmp_path.glob("JOB*") if not p.name.endswith(".molwatch.log"))
    assert files == ["JOB.run.sh", "JOB_stage1.fdf", "JOB_stage2.fdf"]


def test_stage_strategy_loose_only_emits_one_stage_bundle(xyz, tmp_path):
    fdf = tmp_path / "JOB.fdf"
    r = _invoke("fdf", str(xyz), str(fdf),
                "--stage-strategy", "loose-only")
    assert r.exit_code == 0, r.output
    files = sorted(p.name for p in tmp_path.glob("JOB*") if not p.name.endswith(".molwatch.log"))
    assert files == ["JOB.run.sh", "JOB_stage1.fdf"]


def test_stage_strategy_vib_quality_emits_three_stage_bundle(xyz, tmp_path):
    fdf = tmp_path / "JOB.fdf"
    r = _invoke("fdf", str(xyz), str(fdf),
                "--stage-strategy", "vib-quality")
    assert r.exit_code == 0, r.output
    files = sorted(p.name for p in tmp_path.glob("JOB*") if not p.name.endswith(".molwatch.log"))
    assert files == ["JOB.run.sh",
                     "JOB_stage1.fdf",
                     "JOB_stage2.fdf",
                     "JOB_stage3.fdf"]


def test_no_stage_flags_preserves_single_fdf(xyz, tmp_path):
    fdf = tmp_path / "JOB.fdf"
    r = _invoke("fdf", str(xyz), str(fdf))
    assert r.exit_code == 0, r.output
    assert fdf.exists()
    # No bundle artifacts should appear.
    assert not (tmp_path / "JOB.run.sh").exists()
    assert not (tmp_path / "JOB_stage1.fdf").exists()


# --------------------------------------------------------------------- #
#  --stages-json (literal + file path)                                   #
# --------------------------------------------------------------------- #


# The payload shape CHANGED on 2026-08-07 (P2 unit 2), and the change is the
# point rather than a cost of it: a stage is `name` / `enabled` / `overrides`
# (engines/stages.md § 2), and `overrides` may name ANY field of the SIESTA
# schema -- not the eight fields a now-deleted dataclass happened to carry.
# `on_nonconvergence` is gone from it entirely: that is the scheduler edge,
# and so the producer's own input (§ 3).  A format change, pre-1.0, not a
# compatibility shim.
_TWO_STAGE_PAYLOAD = [
    {"name": "stage1", "enabled": True, "overrides": {
        "relax_type": "CG", "relax_steps": 50,
        "relax_force_tol": 0.10, "relax_max_displ": 0.30}},
    {"name": "stage_final", "enabled": True, "overrides": {
        "relax_type": "Broyden", "relax_steps": 100,
        "relax_force_tol": 0.02, "relax_max_displ": 0.05}},
]

#: What the old shape could NOT say, asserted below: a stage varying a
#: parameter no stage type ever carried.
_MESH_LADDER = [
    {"name": "coarse", "overrides": {"mesh_cutoff": 150}},
    {"name": "tight", "overrides": {"mesh_cutoff": 300}},
]


def test_stages_json_can_vary_a_field_no_stage_type_ever_carried(xyz, tmp_path):
    """**The gate, reached through the CLI.**  ``mesh_cutoff`` was not a
    field of the old stage type, so no ``--stages-json`` payload could ask
    two stages to differ in it.  Now it is an ordinary schema field."""
    fdf = tmp_path / "JOB.fdf"
    r = _invoke("fdf", str(xyz), str(fdf),
                "--stages-json", json.dumps(_MESH_LADDER))
    assert r.exit_code == 0, r.output
    coarse = (tmp_path / "JOB_coarse.fdf").read_text()
    tight = (tmp_path / "JOB_tight.fdf").read_text()
    assert "MeshCutoff 150 Ry" in coarse
    assert "MeshCutoff 300 Ry" in tight


def test_stages_json_refuses_an_unknown_field_by_name(xyz, tmp_path):
    """And a typo is refused rather than ignored -- the help text used to
    say "Unknown keys ignored", which is how a misspelt override became a
    silently-default deck."""
    fdf = tmp_path / "JOB.fdf"
    r = _invoke("fdf", str(xyz), str(fdf), "--stages-json", json.dumps(
        [{"name": "tight", "overrides": {"mesh_cutof": 300}}]))
    assert r.exit_code != 0
    assert "mesh_cutof" in r.output
    # named for what the CALLER wrote, not for a file they never touched
    assert "--stages-json" in r.output
    assert "task.json" not in r.output


def test_stages_json_literal_overrides_ladder(xyz, tmp_path):
    fdf = tmp_path / "JOB.fdf"
    r = _invoke("fdf", str(xyz), str(fdf),
                "--stages-json", json.dumps(_TWO_STAGE_PAYLOAD))
    assert r.exit_code == 0, r.output
    files = sorted(p.name for p in tmp_path.glob("JOB*") if not p.name.endswith(".molwatch.log"))
    # Stage names from the payload, not defaults.
    assert files == ["JOB.run.sh",
                     "JOB_stage1.fdf",
                     "JOB_stage_final.fdf"]
    # Stage_final's MD block matches the payload's overrides.
    body = (tmp_path / "JOB_stage_final.fdf").read_text()
    assert "MD.TypeOfRun Broyden" in body
    assert "MD.NumCGsteps 100" in body
    assert "MD.MaxForceTol 0.02 eV/Ang" in body


def test_stages_json_file_path_overrides_ladder(xyz, tmp_path):
    payload_path = tmp_path / "stages.json"
    payload_path.write_text(json.dumps(_TWO_STAGE_PAYLOAD))
    fdf = tmp_path / "JOB.fdf"
    r = _invoke("fdf", str(xyz), str(fdf),
                "--stages-json", str(payload_path))
    assert r.exit_code == 0, r.output
    files = sorted(p.name for p in tmp_path.glob("JOB*") if not p.name.endswith(".molwatch.log"))
    assert "JOB_stage_final.fdf" in files


def test_stages_json_bad_literal_gives_clean_click_error(xyz, tmp_path):
    fdf = tmp_path / "JOB.fdf"
    r = CliRunner().invoke(cli, [
        "fdf", str(xyz), str(fdf), "--stages-json", "[not valid",
    ])
    assert r.exit_code != 0
    assert "--stages-json" in r.output
    assert "not valid JSON" in r.output


def test_stages_json_missing_file_gives_clean_click_error(xyz, tmp_path):
    fdf = tmp_path / "JOB.fdf"
    r = CliRunner().invoke(cli, [
        "fdf", str(xyz), str(fdf),
        "--stages-json", str(tmp_path / "nonexistent.json"),
    ])
    assert r.exit_code != 0
    assert "file not found" in r.output


# --------------------------------------------------------------------- #
#  --stage vs multi-stage flags mutual exclusion                         #
# --------------------------------------------------------------------- #


def test_stage_plus_stage_strategy_is_rejected(xyz, tmp_path):
    fdf = tmp_path / "JOB.fdf"
    r = CliRunner().invoke(cli, [
        "fdf", str(xyz), str(fdf),
        "--stage", "2", "--stage-strategy", "publishable",
    ])
    assert r.exit_code != 0
    assert "mutually exclusive" in r.output


def test_stage_plus_stages_json_is_rejected(xyz, tmp_path):
    fdf = tmp_path / "JOB.fdf"
    r = CliRunner().invoke(cli, [
        "fdf", str(xyz), str(fdf),
        "--stage", "2",
        "--stages-json", json.dumps(_TWO_STAGE_PAYLOAD),
    ])
    assert r.exit_code != 0
    assert "mutually exclusive" in r.output


# --------------------------------------------------------------------- #
#  Filename / SystemLabel coherence                                      #
# --------------------------------------------------------------------- #


def test_all_stage_fdfs_share_same_systemlabel(xyz, tmp_path):
    """Critical for .XV auto-warmstart: every stage's fdf must declare
    the SAME SystemLabel, derived from the on-disk filename stem."""
    fdf = tmp_path / "ChemR-2026.fdf"  # picked stem with punctuation
    r = _invoke("fdf", str(xyz), str(fdf),
                "--stage-strategy", "publishable")
    assert r.exit_code == 0, r.output
    expected_label = "ChemR-2026"
    for stage_fdf in tmp_path.glob("ChemR-2026_*.fdf"):
        body = stage_fdf.read_text()
        assert f"SystemLabel       {expected_label}" in body, stage_fdf


def test_runner_BASENAME_matches_filename_stem(xyz, tmp_path):
    fdf = tmp_path / "ChemR-2026.fdf"
    r = _invoke("fdf", str(xyz), str(fdf),
                "--stage-strategy", "publishable")
    assert r.exit_code == 0, r.output
    runner = (tmp_path / "ChemR-2026.run.sh").read_text()
    assert "BASENAME='ChemR-2026'" in runner


# --------------------------------------------------------------------- #
#  Runner chmod + bash -n + array consistency                            #
# --------------------------------------------------------------------- #


def test_runner_is_executable(xyz, tmp_path):
    fdf = tmp_path / "JOB.fdf"
    r = _invoke("fdf", str(xyz), str(fdf),
                "--stage-strategy", "publishable")
    assert r.exit_code == 0, r.output
    mode = (tmp_path / "JOB.run.sh").stat().st_mode
    assert mode & stat.S_IXUSR, "runner is not user-executable"


def test_runner_passes_bash_syntax_check(xyz, tmp_path):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash unavailable")
    fdf = tmp_path / "JOB.fdf"
    r = _invoke("fdf", str(xyz), str(fdf),
                "--stage-strategy", "vib-quality")
    assert r.exit_code == 0, r.output
    runner_path = tmp_path / "JOB.run.sh"
    check = subprocess.run([bash, "-n", str(runner_path)],
                            capture_output=True, text=True)
    assert check.returncode == 0, check.stderr


def test_runner_stages_array_matches_emitted_fdfs(xyz, tmp_path):
    fdf = tmp_path / "JOB.fdf"
    r = _invoke("fdf", str(xyz), str(fdf),
                "--stage-strategy", "vib-quality")
    assert r.exit_code == 0, r.output
    runner = (tmp_path / "JOB.run.sh").read_text()
    assert "STAGES=(stage1 stage2 stage3)" in runner
    # Default ON_NONCONV (with vib-quality enabling stage3, the runner's
    # force-halt-last contract converts the would-be 'halt' to 'halt' --
    # so all three policies are visible explicitly here).
    assert "ON_NONCONV=(proceed halt halt)" in runner


# --------------------------------------------------------------------- #
#  Combined --stages-json + --stage-strategy                             #
# --------------------------------------------------------------------- #


def test_stages_json_then_stage_strategy_layers_correctly(xyz, tmp_path):
    """--stages-json sets the knob values; --stage-strategy overlays
    enable flags.  Combined, the user gets custom knob values from
    the payload PLUS the preset's enable pattern."""
    fdf = tmp_path / "JOB.fdf"
    # Payload has stage1 + stage_final both enabled.  loose-only
    # strategy should only keep the FIRST stage enabled.
    r = _invoke("fdf", str(xyz), str(fdf),
                "--stages-json", json.dumps(_TWO_STAGE_PAYLOAD),
                "--stage-strategy", "loose-only")
    assert r.exit_code == 0, r.output
    files = sorted(p.name for p in tmp_path.glob("JOB*") if not p.name.endswith(".molwatch.log"))
    # Only stage1's fdf survives the enable-flag overlay.
    assert files == ["JOB.run.sh", "JOB_stage1.fdf"]
    # And its MD block reflects the payload's stage1 knobs, not the
    # SiestaConfig defaults.
    body = (tmp_path / "JOB_stage1.fdf").read_text()
    assert "MD.NumCGsteps 50" in body
    assert "MD.MaxForceTol 0.1 eV/Ang" in body


# --------------------------------------------------------------------- #
#  --jobset: emit job-set.json so the bundle runs via `molbuilder jobset` #
# --------------------------------------------------------------------- #


def test_jobset_flag_emits_runnable_job_set_json(xyz, tmp_path):
    from molbuilder.jobset.model import JobSet, SCHEMA

    fdf = tmp_path / "JOB.fdf"
    r = _invoke("fdf", str(xyz), str(fdf),
                "--stage-strategy", "publishable", "--jobset")
    assert r.exit_code == 0, r.output
    jpath = tmp_path / "job-set.json"
    assert jpath.is_file()                          # opt-in artifact present

    js = JobSet.load(jpath)
    assert js.to_dict()["schema"] == SCHEMA
    assert js.kind == "ladder" and js.engine == "siesta"
    assert js.validate() == []                       # framework-valid
    # scripts match the <label>_<stage>.fdf actually rendered next to it.
    assert [j.script for j in js.jobs] == ["JOB_stage1.fdf", "JOB_stage2.fdf"]
    for j in js.jobs:
        assert (tmp_path / j.script).is_file()
    # the chain + carry are wired (stage2 warm-starts from stage1).
    assert js.jobs[1].depends_on == "stage1"
    assert "JOB.XV" in [c.pattern for c in js.jobs[1].carry]


def test_jobset_flag_requires_multi_stage(xyz, tmp_path):
    fdf = tmp_path / "JOB.fdf"
    r = CliRunner().invoke(
        cli, ["fdf", str(xyz), str(fdf), "--jobset"])
    assert r.exit_code != 0
    assert "--jobset" in r.output and "stage-strategy" in r.output


def test_jobset_flag_off_by_default_no_job_set_json(xyz, tmp_path):
    fdf = tmp_path / "JOB.fdf"
    r = _invoke("fdf", str(xyz), str(fdf), "--stage-strategy", "publishable")
    assert r.exit_code == 0, r.output
    assert not (tmp_path / "job-set.json").exists()   # unchanged default


# --------------------------------------------------------------------- #
#  End-to-end loop + cross-stage consistency (depth, not API-presence)   #
# --------------------------------------------------------------------- #


def test_jobset_end_to_end_produce_prep_status_submit(xyz, tmp_path):
    """Pin the WHOLE loop on one bundle: produce (--jobset) -> the plan's
    scripts are the real rendered files -> prep lays out the dirs -> status
    reports pending + the right resume point -> submit(dry-run) plans every
    stage. This is the composition the per-layer unit tests don't cover."""
    from molbuilder.jobset.model import JobSet
    from molbuilder.jobset import prep_jobset, jobset_status, submit_jobset

    b = tmp_path / "bundle"; b.mkdir()
    r = _invoke("fdf", str(xyz), str(b / "JOB.fdf"),
                "--stage-strategy", "publishable", "--jobset")
    assert r.exit_code == 0, r.output

    js = JobSet.load(b / "job-set.json")
    # cross-consistency: every Job.script names a file that was actually rendered
    assert [j.script for j in js.jobs] == ["JOB_stage1.fdf", "JOB_stage2.fdf"]
    for j in js.jobs:
        assert (b / j.script).is_file()

    # a bundle carries the script_generation block the wrappers need.
    (b / ".molbuilder.json").write_text(
        '{"script_generation": {"preamble": "module load mamba", '
        '"activation": "source activate"}}')
    prep_jobset(js, b, emit_sbatch=False)
    for j in js.jobs:
        assert (b / f"point-{j.name}").is_dir()

    st = jobset_status(js, b)
    assert [s.state for s in st.stages] == ["pending", "pending"]
    assert st.first_incomplete == "stage1" and st.complete is False

    res = submit_jobset(js, b, mode="direct", dry_run=True)
    assert [x.status for x in res] == ["planned", "planned"]


def test_jobset_systemlabel_carry_consistency(xyz, tmp_path):
    """The contract that makes warm-restart work: the SystemLabel INSIDE each
    rendered stage .fdf == the JobSet name == the carry filename stem. If any
    drifts, SIESTA's auto-restart silently breaks."""
    from molbuilder.jobset.model import JobSet

    b = tmp_path / "bundle"; b.mkdir()
    _invoke("fdf", str(xyz), str(b / "JOB.fdf"),
            "--stage-strategy", "publishable", "--jobset")
    js = JobSet.load(b / "job-set.json")

    assert js.name == "JOB"
    for j in js.jobs:
        body = (b / j.script).read_text()
        assert any("SystemLabel" in ln and "JOB" in ln
                   for ln in body.splitlines())          # label baked in fdf
    # the carried restart files use that SAME shared label.
    assert "JOB.XV" in [c.pattern for c in js.jobs[1].carry]


# --------------------------------------------------------------------- #
#  --stage-resources: per-stage scheduler resources in the job-set       #
# --------------------------------------------------------------------- #


def test_stage_resources_set_per_stage_in_jobset(xyz, tmp_path):
    """The headline staged-cluster capability (§6): a cheap warm-up + an
    expensive final, expressed per stage and carried in job-set.json."""
    from molbuilder.jobset.model import JobSet

    b = tmp_path / "bundle"; b.mkdir()
    spec = ('{"stage1": {"domain": "htc", "time": "0-04:00:00"}, '
            '"stage2": {"domain": "public", "time": "7-00:00:00", '
            '"exclusive": true}}')
    r = _invoke("fdf", str(xyz), str(b / "JOB.fdf"),
                "--stage-strategy", "publishable", "--jobset",
                "--stage-resources", spec)
    assert r.exit_code == 0, r.output

    js = JobSet.load(b / "job-set.json")
    s1, s2 = js.jobs
    assert s1.resources.domain == "htc" and s1.resources.time == "0-04:00:00"
    assert s2.resources.domain == "public" and s2.resources.exclusive is True
    # and the plan surfaces the difference.
    from molbuilder.jobset import render_plan
    txt = render_plan(js)
    assert "domain=htc" in txt and "domain=public" in txt


def test_stage_resources_rejects_unknown_stage_name(xyz, tmp_path):
    b = tmp_path / "bundle"; b.mkdir()
    r = CliRunner().invoke(cli, [
        "fdf", str(xyz), str(b / "JOB.fdf"),
        "--stage-strategy", "publishable", "--jobset",
        "--stage-resources", '{"stageX": {"domain": "htc"}}'])
    assert r.exit_code != 0
    assert "unknown stage name" in r.output


def test_stage_resources_requires_jobset(xyz, tmp_path):
    b = tmp_path / "bundle"; b.mkdir()
    r = CliRunner().invoke(cli, [
        "fdf", str(xyz), str(b / "JOB.fdf"),
        "--stage-strategy", "publishable",
        "--stage-resources", '{"stage1": {"domain": "htc"}}'])
    assert r.exit_code != 0
    assert "--stage-resources only applies" in r.output


def test_stage_resources_rejects_unknown_field(xyz, tmp_path):
    # a typo'd resource field must be a LOUD error, not silently dropped.
    b = tmp_path / "bundle"; b.mkdir()
    r = CliRunner().invoke(cli, [
        "fdf", str(xyz), str(b / "JOB.fdf"),
        "--stage-strategy", "publishable", "--jobset",
        "--stage-resources", '{"stage1": {"domian": "htc"}}'])  # typo
    assert r.exit_code != 0
    assert "unknown field" in r.output and "domian" in r.output


def test_stage_resources_rejects_non_object_body(xyz, tmp_path):
    b = tmp_path / "bundle"; b.mkdir()
    r = CliRunner().invoke(cli, [
        "fdf", str(xyz), str(b / "JOB.fdf"),
        "--stage-strategy", "publishable", "--jobset",
        "--stage-resources", '{"stage1": "htc"}'])         # not an object
    assert r.exit_code != 0
    assert "must be an object" in r.output
