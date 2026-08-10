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


def _value(deck: str, key: str) -> str:
    """The value an fdf key carries, read off the deck.

    Deliberately a value reader and not a substring search: a stage's number
    survives resolution as a *number*, and how the emitter spells it is the
    emitter's business.  Its twin lives in ``test_stage_resolution.py`` --
    same gate, reached through the renderer instead of the CLI -- and the
    duplication is the lesser evil against a test module importing another.
    """
    for line in deck.splitlines():
        parts = line.split()
        if parts and parts[0] == key:
            return parts[1]
    raise AssertionError(f"{key!r} not in the deck:\n{deck}")


# --------------------------------------------------------------------- #
#  Multi-stage branch entry                                              #
# --------------------------------------------------------------------- #


def test_stage_strategy_publishable_emits_two_stage_bundle(xyz, tmp_path):
    fdf = tmp_path / "JOB.fdf"
    r = _invoke("fdf", str(xyz), str(fdf),
                "--stage-strategy", "publishable")
    assert r.exit_code == 0, r.output
    files = sorted(p.name for p in tmp_path.glob("JOB*") if not p.name.endswith(".molwatch.log"))
    assert files == ["JOB_01_coarse.fdf", "JOB_02_medium.fdf"]


def test_stage_strategy_loose_only_emits_one_stage_bundle(xyz, tmp_path):
    fdf = tmp_path / "JOB.fdf"
    r = _invoke("fdf", str(xyz), str(fdf),
                "--stage-strategy", "loose-only")
    assert r.exit_code == 0, r.output
    files = sorted(p.name for p in tmp_path.glob("JOB*") if not p.name.endswith(".molwatch.log"))
    assert files == ["JOB_01_coarse.fdf"]


def test_stage_strategy_vib_quality_emits_three_stage_bundle(xyz, tmp_path):
    fdf = tmp_path / "JOB.fdf"
    r = _invoke("fdf", str(xyz), str(fdf),
                "--stage-strategy", "vib-quality")
    assert r.exit_code == 0, r.output
    files = sorted(p.name for p in tmp_path.glob("JOB*") if not p.name.endswith(".molwatch.log"))
    assert files == [
                     "JOB_01_coarse.fdf",
                     "JOB_02_medium.fdf",
                     "JOB_03_tight.fdf"]


def test_no_stage_flags_preserves_single_fdf(xyz, tmp_path):
    fdf = tmp_path / "JOB.fdf"
    r = _invoke("fdf", str(xyz), str(fdf))
    assert r.exit_code == 0, r.output
    assert fdf.exists()
    # No bundle artifacts should appear.
    assert not (tmp_path / "JOB.run.sh").exists()   # never, since decision 29
    assert not (tmp_path / "JOB_01_coarse.fdf").exists()


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
    {"name": "coarse", "enabled": True, "overrides": {
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
    coarse = (tmp_path / "JOB_01_coarse.fdf").read_text()
    tight = (tmp_path / "JOB_02_tight.fdf").read_text()
    # NOT the deck text.  ``mesh_cutoff`` is a float field and JSON has one
    # number, so the payload above writes ``150`` and resolution widens it to
    # 150.0 -- the deck reads the same however the description spelled it.
    # Pinning "MeshCutoff 150 Ry" pinned the inconsistency the M2 fix removed
    # (one value, two decks), and this assertion outlived its twin in
    # test_stage_resolution.py because that grep was never widened past one
    # file.  2026-08-07.
    assert float(_value(coarse, "MeshCutoff")) == 150
    assert float(_value(tight, "MeshCutoff")) == 300


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
    assert files == [
                     "JOB_01_coarse.fdf",
                     "JOB_02_stage_final.fdf"]
    # Stage_final's MD block matches the payload's overrides.
    body = (tmp_path / "JOB_02_stage_final.fdf").read_text()
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
    assert "JOB_02_stage_final.fdf" in files


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


# The four runner tests that stood here are RETIRED (decision 29, 2026-08-10).
# They pinned `<label>.run.sh` -- its BASENAME, its exec bit, `bash -n`, and its
# STAGES array -- and the produce no longer emits one.  Flat is not a lesser
# path with its own launcher; it runs through `jobset prep` / `submit run
# --chain` like the hierarchy, so there is no second launcher to keep correct.
# Deleted rather than adapted: a test whose subject is gone is not failing, it
# is orphaned (process/testing.md).  What replaced their coverage is the
# wrapper's own suite, which every stage of both shapes now goes through.


# --------------------------------------------------------------------- #
#  Combined --stages-json + --stage-strategy                             #
# --------------------------------------------------------------------- #


def test_stages_json_then_stage_strategy_layers_correctly(xyz, tmp_path):
    """--stages-json sets the knob values; --stage-strategy overlays
    enable flags.  Combined, the user gets custom knob values from
    the payload PLUS the preset's enable pattern."""
    fdf = tmp_path / "JOB.fdf"
    # Payload has coarse + stage_final both enabled.  loose-only
    # strategy should only keep the FIRST stage enabled.
    r = _invoke("fdf", str(xyz), str(fdf),
                "--stages-json", json.dumps(_TWO_STAGE_PAYLOAD),
                "--stage-strategy", "loose-only")
    assert r.exit_code == 0, r.output
    files = sorted(p.name for p in tmp_path.glob("JOB*") if not p.name.endswith(".molwatch.log"))
    # Only stage1's fdf survives the enable-flag overlay.
    assert files == ["JOB_01_coarse.fdf"]
    # And its MD block reflects the payload's 01_coarse knobs, not the
    # SiestaConfig defaults.
    body = (tmp_path / "JOB_01_coarse.fdf").read_text()
    assert "MD.NumCGsteps 50" in body
    assert "MD.MaxForceTol 0.1 eV/Ang" in body


# --------------------------------------------------------------------- #
#  --shape hierarchical: the ladder as a JobSet (P5 unit 1)            #
# --------------------------------------------------------------------- #


def test_jobset_flag_emits_runnable_job_set_json(xyz, tmp_path):
    from molbuilder.jobset.model import JobSet, SCHEMA

    fdf = tmp_path / "JOB.fdf"
    r = _invoke("fdf", str(xyz), str(fdf),
                "--stage-strategy", "publishable", "--shape", "hierarchical")
    assert r.exit_code == 0, r.output
    jpath = tmp_path / "job-set.json"
    assert jpath.is_file()                          # opt-in artifact present

    js = JobSet.load(jpath)
    assert js.to_dict()["schema"] == SCHEMA
    assert js.kind == "ladder" and js.engine == "siesta"
    assert js.validate() == []                       # framework-valid
    # scripts match the <label>_<stage>.fdf actually rendered next to it.
    assert [j.script for j in js.jobs] == ["JOB_01_coarse.fdf", "JOB_02_medium.fdf"]
    for j in js.jobs:
        assert (tmp_path / j.script).is_file()
    # the chain + carry are wired (stage2 warm-starts from 01_coarse).
    assert js.jobs[1].depends_on == "coarse"
    assert "JOB.XV" in [c.pattern for c in js.jobs[1].carry]


def test_jobset_flag_requires_multi_stage(xyz, tmp_path):
    fdf = tmp_path / "JOB.fdf"
    r = CliRunner().invoke(
        cli, ["fdf", str(xyz), str(fdf), "--shape", "hierarchical"])
    assert r.exit_code != 0
    assert "--shape hierarchical" in r.output and "stage-strategy" in r.output


def test_every_shape_gets_a_jobset_because_there_is_one_framework(xyz, tmp_path):
    """Inverted by decision 29 (2026-08-10), and the inversion is the point.

    This pinned *"no job-set.json unless you ask"*, from when the JobSet was
    the hierarchical shape's own artifact.  It is now what makes `jobset prep`
    / `submit run --chain` the launcher for BOTH shapes -- *"the prep,
    deployment and execution chain of command is the same framework"* -- so a
    flat bundle without one would be a flat bundle nobody could run.
    """
    fdf = tmp_path / "JOB.fdf"
    r = _invoke("fdf", str(xyz), str(fdf), "--stage-strategy", "publishable")
    assert r.exit_code == 0, r.output
    assert (tmp_path / "job-set.json").is_file()
    assert not (tmp_path / "JOB.run.sh").exists()     # and no second launcher


# --------------------------------------------------------------------- #
#  End-to-end loop + cross-stage consistency (depth, not API-presence)   #
# --------------------------------------------------------------------- #


def test_jobset_end_to_end_produce_prep_status_submit(xyz, tmp_path):
    """Pin the WHOLE loop on one bundle: produce (--shape hierarchical) -> the plan's
    scripts are the real rendered files -> prep lays out the dirs -> status
    reports pending + the right resume point -> submit(dry-run) plans every
    stage. This is the composition the per-layer unit tests don't cover."""
    from molbuilder.jobset.model import JobSet
    from molbuilder.jobset import prep_jobset, jobset_status, submit_jobset

    b = tmp_path / "bundle"; b.mkdir()
    r = _invoke("fdf", str(xyz), str(b / "JOB.fdf"),
                "--stage-strategy", "publishable", "--shape", "hierarchical")
    assert r.exit_code == 0, r.output

    js = JobSet.load(b / "job-set.json")
    # cross-consistency: every Job.script names a file that was actually rendered
    assert [j.script for j in js.jobs] == ["JOB_01_coarse.fdf", "JOB_02_medium.fdf"]
    for j in js.jobs:
        assert (b / j.script).is_file()

    # a bundle carries the script_generation block the wrappers need.
    (b / ".molbuilder.json").write_text(
        '{"script_generation": {"preamble": "module load mamba", '
        '"activation": "source activate"}}')
    prep_jobset(js, b, emit_sbatch=False)
    # A LADDER's stage directory is ``<seq>_<name>`` (project-layout.md § 4.1),
    # not the sweep's ``point-<name>``.  This asserted ``point-{name}`` until
    # 2026-08-10 -- correctly, of the behaviour of the day, which was
    # worked-example.md's gap 6: `job_dir_name` did not branch on JobSet.kind,
    # so a staged run got the benchmark's naming.
    assert sorted(d.name for d in b.iterdir() if d.is_dir()) == \
        ["01_coarse", "02_medium"]
    # And the directory carries the SAME token as the deck inside it, which is
    # the self-check § 4.1 asks for rather than a repetition.
    for j in js.jobs:
        seq_name = j.script[len("JOB_"):-len(".fdf")]     # e.g. 01_coarse
        assert (b / seq_name / j.script).is_symlink()

    st = jobset_status(js, b)
    assert [s.state for s in st.stages] == ["pending", "pending"]
    assert st.first_incomplete == "coarse" and st.complete is False

    res = submit_jobset(js, b, mode="direct", dry_run=True)
    assert [x.status for x in res] == ["planned", "planned"]


def test_jobset_systemlabel_carry_consistency(xyz, tmp_path):
    """The contract that makes warm-restart work: the SystemLabel INSIDE each
    rendered stage .fdf == the JobSet name == the carry filename stem. If any
    drifts, SIESTA's auto-restart silently breaks."""
    from molbuilder.jobset.model import JobSet

    b = tmp_path / "bundle"; b.mkdir()
    _invoke("fdf", str(xyz), str(b / "JOB.fdf"),
            "--stage-strategy", "publishable", "--shape", "hierarchical")
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
    spec = ('{"coarse": {"domain": "htc", "time": "0-04:00:00"}, '
            '"medium": {"domain": "public", "time": "7-00:00:00", '
            '"exclusive": true}}')
    r = _invoke("fdf", str(xyz), str(b / "JOB.fdf"),
                "--stage-strategy", "publishable", "--shape", "hierarchical",
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
        "--stage-strategy", "publishable", "--shape", "hierarchical",
        "--stage-resources", '{"stageX": {"domain": "htc"}}'])
    assert r.exit_code != 0
    assert "unknown stage name" in r.output


def test_stage_resources_requires_jobset(xyz, tmp_path):
    b = tmp_path / "bundle"; b.mkdir()
    r = CliRunner().invoke(cli, [
        "fdf", str(xyz), str(b / "JOB.fdf"),
        "--stage-strategy", "publishable",
        "--stage-resources", '{"coarse": {"domain": "htc"}}'])
    assert r.exit_code != 0
    assert "--shape hierarchical" in r.output


def test_stage_resources_rejects_unknown_field(xyz, tmp_path):
    # a typo'd resource field must be a LOUD error, not silently dropped.
    b = tmp_path / "bundle"; b.mkdir()
    r = CliRunner().invoke(cli, [
        "fdf", str(xyz), str(b / "JOB.fdf"),
        "--stage-strategy", "publishable", "--shape", "hierarchical",
        "--stage-resources", '{"coarse": {"domian": "htc"}}'])  # typo
    assert r.exit_code != 0
    assert "unknown field" in r.output and "domian" in r.output


def test_stage_resources_rejects_non_object_body(xyz, tmp_path):
    b = tmp_path / "bundle"; b.mkdir()
    r = CliRunner().invoke(cli, [
        "fdf", str(xyz), str(b / "JOB.fdf"),
        "--stage-strategy", "publishable", "--shape", "hierarchical",
        "--stage-resources", '{"coarse": "htc"}'])         # not an object
    assert r.exit_code != 0
    assert "must be an object" in r.output


# --------------------------------------------------------------------- #
#  --vacuum reaches the staged branch (found by the M4 walk, 2026-08-10) #
# --------------------------------------------------------------------- #


def _cell_lengths(deck: str):
    """The three orthorhombic cell lengths, read off %block LatticeVectors."""
    lines = deck.splitlines()
    i = next(n for n, ln in enumerate(lines)
             if ln.strip().startswith("%block LatticeVectors"))
    return [float(lines[i + 1 + k].split()[k]) for k in range(3)]


def test_vacuum_reaches_the_staged_branch(xyz, tmp_path):
    """``--vacuum`` was accepted, range-checked and then DROPPED the moment a
    ladder flag turned the multi-stage branch on: it was not a parameter of
    ``_emit_siesta_multi_stage`` at all, while the single-stage branch passed
    it to ``convert``.  A user asking for 8 A of isolation got the 3 A default
    and a molecule whose periodic images interact.

    Not caught by any existing test because both branches emit a *valid* deck
    -- only the cell differs, and nothing compared the two branches' cells.
    """
    wide = tmp_path / "wide"
    narrow = tmp_path / "narrow"
    for out, args in ((wide, ["--vacuum", "8"]), (narrow, [])):
        r = _invoke("fdf", str(xyz), str(out / "JOB.fdf"),
                    "--stage-strategy", "loose-only", *args)
        assert r.exit_code == 0, r.output

    w = _cell_lengths((wide / "JOB_01_coarse.fdf").read_text())
    n = _cell_lengths((narrow / "JOB_01_coarse.fdf").read_text())
    # 8 A per side vs the 3 A default -> every axis grows by exactly 2*(8-3).
    for axis, (wv, nv) in enumerate(zip(w, n)):
        assert wv == pytest.approx(nv + 10.0, abs=1e-6), (axis, w, n)


def test_vacuum_agrees_between_the_staged_and_single_branches(xyz, tmp_path):
    """The two branches must resolve the SAME cell from the same input, which
    is the invariant the bug broke.  Comparing them is what a test of either
    branch alone could not do."""
    staged = tmp_path / "staged"
    single = tmp_path / "single"
    assert _invoke("fdf", str(xyz), str(staged / "JOB.fdf"),
                   "--stage-strategy", "loose-only",
                   "--vacuum", "7.5").exit_code == 0
    assert _invoke("fdf", str(xyz), str(single / "JOB.fdf"),
                   "--vacuum", "7.5").exit_code == 0

    a = _cell_lengths((staged / "JOB_01_coarse.fdf").read_text())
    b = _cell_lengths((single / "JOB.fdf").read_text())
    assert a == pytest.approx(b, abs=1e-6)


# --------------------------------------------------------------------- #
#  task.json — the produce writes the description (P5 unit 4)            #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("shape", ["flat", "hierarchical"])
def test_the_produce_writes_a_description_that_reads_back(xyz, tmp_path, shape):
    """`engines/stages.md § 7` lists the description among what a generator
    must produce, and § 6.7 puts the SHAPE in it — *"A field is what makes
    every prep agree, and it is the only place that can."*

    Without it the shape lives only in the command somebody typed, and
    decision 29 makes `prep` the one place the shape branches: it would have
    nothing to read.  Asserted through the ONE reader, so a file this produce
    writes but the codec refuses is a failure here rather than at prep.
    """
    from molbuilder.task import read_task
    fdf = tmp_path / "out" / "ch4.fdf"
    r = _invoke("fdf", str(xyz), str(fdf),
                "--stage-strategy", "publishable", "--shape", shape)
    assert r.exit_code == 0, r.output

    task = read_task(fdf.parent / "task.json")
    assert task.shape == shape
    assert task.engine == "siesta"
    assert task.label == "ch4"                     # the stem of every deck
    assert [s.name for s in task.stages] == ["coarse", "medium", "tight"]
    # the witness: what this was a calculation OF, at the moment it was written
    assert task.structure.formula == "CH4"
    assert task.structure.atoms == 5
    assert task.run.id == "ch4_CH4"


def test_varies_records_every_promoted_column_not_just_one_stages(xyz, tmp_path):
    """§ 6.2: `varies` is the COLUMN SET, and a stage may leave a promoted cell
    empty — that means *"use the template's value"*, a real state rather than
    an absence of intent.  So the union, never one stage's keys."""
    from molbuilder.task import read_task
    ladder = json.dumps([
        {"name": "coarse", "overrides": {"mesh_cutoff": 150}},
        {"name": "tight", "overrides": {"relax_force_tol": 0.01}},
    ])
    fdf = tmp_path / "out" / "ch4.fdf"
    r = _invoke("fdf", str(xyz), str(fdf), "--stages-json", ladder)
    assert r.exit_code == 0, r.output

    task = read_task(fdf.parent / "task.json")
    assert set(task.varies) == {"mesh_cutoff", "relax_force_tol"}
    # ...and each stage keeps only its own cells (overrides ⊆ varies)
    assert dict(task.stages[0].overrides) == {"mesh_cutoff": 150}
    assert dict(task.stages[1].overrides) == {"relax_force_tol": 0.01}


# --------------------------------------------------------------------- #
#  the produce is transactional (engines/stages.md § 7.2)               #
# --------------------------------------------------------------------- #


def test_a_produce_that_fails_partway_leaves_nothing_behind(xyz, tmp_path,
                                                            monkeypatch):
    """§ 7.2: *"every deck, every wrapper and the description are built
    somewhere else and moved into place only when all of them succeeded. On
    failure nothing is moved."*

    The failure is injected at the LAST step, so the decks and the runner have
    already been written — that is the case a non-transactional produce gets
    wrong, and the one that leaves *"a half-written folder"* § 7.2 calls worse
    than none.
    """
    import molbuilder.task as _task_mod

    def _boom(path, task):
        raise OSError("disk full")
    monkeypatch.setattr(_task_mod, "write_task", _boom)

    out = tmp_path / "out"
    out.mkdir()
    # catch_exceptions=True, unlike `_invoke`: an OSError mid-produce is not a
    # UsageError, and letting it escape the runner would end the test before it
    # could look at the directory -- which is the whole assertion.
    r = CliRunner().invoke(cli, ["fdf", str(xyz), str(out / "ch4.fdf"),
                                 "--stage-strategy", "publishable"],
                           catch_exceptions=True)
    assert r.exit_code != 0
    assert isinstance(r.exception, OSError)                 # ...and it propagated

    assert sorted(p.name for p in out.iterdir()) == []      # nothing published
    # ...and the staging directory did not leak
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []


def test_producing_twice_keeps_the_warm_files_that_were_already_there(xyz,
                                                                      tmp_path):
    """§ 7.2 is explicit about the one thing the transaction must NOT do:
    *"it must not remove warm files that were already there; producing twice is
    run-identity.md § 6, and those files are the point."*

    So the publish moves file by file into the target — a directory swap would
    take the previous run's geometry with it.
    """
    out = tmp_path / "out"
    out.mkdir()
    (out / "ch4.XV").write_text("GEOMETRY FROM THE LAST RUN")

    for _ in range(2):
        r = _invoke("fdf", str(xyz), str(out / "ch4.fdf"),
                    "--stage-strategy", "publishable")
        assert r.exit_code == 0, r.output

    assert (out / "ch4.XV").read_text() == "GEOMETRY FROM THE LAST RUN"
    assert (out / "ch4_01_coarse.fdf").is_file()
    assert (out / "task.json").is_file()
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []
