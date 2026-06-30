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


_XYZ = textwrap.dedent("""\
    2

    H 0.0 0.0 0.0
    H 0.0 0.0 0.74
""")


@pytest.fixture
def xyz(tmp_path):
    p = tmp_path / "h2.xyz"
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


_TWO_STAGE_PAYLOAD = [
    {"name": "stage1", "enabled": True, "relax_type": "CG",
     "relax_steps": 50, "relax_force_tol": 0.10, "relax_max_displ": 0.30,
     "on_nonconvergence": "proceed"},
    {"name": "stage_final", "enabled": True, "relax_type": "Broyden",
     "relax_steps": 100, "relax_force_tol": 0.02, "relax_max_displ": 0.05,
     "on_nonconvergence": "halt"},
]


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
