"""Tests for the SIESTA multi-stage emitters (task #542, commit 2):
:func:`render_siesta_stage_fdfs` + :func:`render_siesta_stages_runner`.

Pins the per-stage emission contract:
  * one .fdf per enabled stage, filename ``{system_label}_{stage}.fdf``
  * every fdf shares the same SystemLabel (so SIESTA auto-reads .XV
    between stages without any file-renaming)
  * a stage's ``overrides`` beat the template's values
  * runner is a valid bash script (``bash -n`` clean)
  * runner array literals match the enabled stages in order
  * LAST enabled stage's policy is force-halted in the runner
  * disabled stages drop out of both fdf set and runner arrays
  * empty-enabled-list raises (never silently emit a zero-stage run)

2026-08-07 (P2 unit 2): the ladder is an ARGUMENT now, not ``cfg.stages`` --
an engine config carries no stage list (engines/stages.md § 1.1), and the
non-convergence policy is the producer's own input rather than a stage field
(§ 3).  Every assertion below survived that; only the wiring moved.
"""
from __future__ import annotations

import dataclasses
import shutil
import subprocess

import numpy as np
import pytest

from molbuilder.config.siesta import SiestaConfig
from molbuilder.siesta import (
    render_siesta_stage_fdfs,
    render_siesta_stages_runner,
)
from molbuilder.siesta.stages import (
    DEFAULT_NONCONVERGENCE,
    default_siesta_stages,
)
from molbuilder.structure import Structure


# --------------------------------------------------------------------- #
#  Fixtures                                                              #
# --------------------------------------------------------------------- #


@pytest.fixture
def h2():
    # Per-side vacuum so the derived cell isn't degenerate for this linear
    # molecule (a zero-thickness box would hard-error -- structure-periodicity.md).
    return Structure(
        elements=["H", "H"],
        positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
        vacuum=(12.0, 12.0, 12.0),
    )


@pytest.fixture
def cfg():
    """The template -- one ordinary parameter set, no ladder in it."""
    return SiestaConfig(system_label="JOB")


@pytest.fixture
def stages():
    """The shipped ladder: stage1 + stage2 enabled, stage3 disabled."""
    return default_siesta_stages()


# --------------------------------------------------------------------- #
#  render_siesta_stage_fdfs                                             #
# --------------------------------------------------------------------- #


def test_fdfs_returns_one_per_enabled_stage(h2, cfg, stages):
    fdfs = render_siesta_stage_fdfs(h2, cfg, stages)
    assert sorted(fdfs) == ["JOB_stage1.fdf", "JOB_stage2.fdf"]


def test_fdfs_filename_uses_system_label(h2, stages):
    cfg = SiestaConfig(system_label="TJ-BDT-Au111")
    fdfs = render_siesta_stage_fdfs(h2, cfg, stages)
    assert all(name.startswith("TJ-BDT-Au111_") for name in fdfs)


def test_fdfs_share_systemlabel_for_warm_restart(h2, cfg, stages):
    """Each emitted fdf must declare the SAME SystemLabel as cfg --
    that's the mechanism by which SIESTA auto-reads <label>.XV across
    stage invocations."""
    fdfs = render_siesta_stage_fdfs(h2, cfg, stages)
    for body in fdfs.values():
        assert f"SystemLabel       {cfg.system_label}" in body


def test_fdfs_per_stage_md_block_uses_stage_values(h2, cfg, stages):
    """stage1 = CG / 600 / 0.05 / 0.20, stage2 = Broyden / 200 / 0.04
    / 0.05.  Each fdf's MD block reflects its own stage's values, NOT
    the cfg.relax_* default."""
    fdfs = render_siesta_stage_fdfs(h2, cfg, stages)
    s1 = fdfs["JOB_stage1.fdf"]
    s2 = fdfs["JOB_stage2.fdf"]

    assert "MD.TypeOfRun CG" in s1
    assert "MD.NumCGsteps 600" in s1
    assert "MD.MaxForceTol 0.05 eV/Ang" in s1
    assert "MD.MaxCGDispl 0.2" in s1

    assert "MD.TypeOfRun Broyden" in s2
    assert "MD.NumCGsteps 200" in s2
    assert "MD.MaxForceTol 0.04 eV/Ang" in s2
    assert "MD.MaxCGDispl 0.05" in s2


def test_fdfs_disabled_stages_drop_out(h2, cfg, stages):
    # Disable stage2; enable stage3.
    stages[1] = dataclasses.replace(stages[1], enabled=False)
    stages[2] = dataclasses.replace(stages[2], enabled=True)
    fdfs = render_siesta_stage_fdfs(h2, cfg, stages)
    assert sorted(fdfs) == ["JOB_stage1.fdf", "JOB_stage3.fdf"]


def test_fdfs_single_enabled_stage_is_still_emitted(h2, cfg, stages):
    for i in (1, 2):
        stages[i] = dataclasses.replace(stages[i], enabled=False)
    fdfs = render_siesta_stage_fdfs(h2, cfg, stages)
    assert list(fdfs) == ["JOB_stage1.fdf"]


def test_fdfs_zero_enabled_raises(h2, cfg, stages):
    stages = [dataclasses.replace(s, enabled=False) for s in stages]
    with pytest.raises(ValueError, match="no enabled entries"):
        render_siesta_stage_fdfs(h2, cfg, stages)


# --------------------------------------------------------------------- #
#  render_siesta_stages_runner                                          #
# --------------------------------------------------------------------- #


def _bash():
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash unavailable")
    return bash


def test_runner_is_valid_bash(cfg, stages):
    script = render_siesta_stages_runner(
        cfg, stages, on_nonconvergence=DEFAULT_NONCONVERGENCE)
    r = subprocess.run([_bash(), "-n"], input=script, text=True,
                       capture_output=True)
    assert r.returncode == 0, r.stderr


def test_runner_arrays_match_enabled_stages(cfg, stages):
    script = render_siesta_stages_runner(
        cfg, stages, on_nonconvergence=DEFAULT_NONCONVERGENCE)
    # Defaults: stage1 + stage2 enabled.
    assert "STAGES=(stage1 stage2)" in script
    # stage1.on_nonconvergence='proceed' (preserved); stage2 is the
    # last enabled stage so it's force-halted (was already 'halt' here).
    assert "ON_NONCONV=(proceed halt)" in script


def test_runner_force_halts_last_enabled_stage_even_if_user_set_proceed(
        cfg, stages):
    """The last enabled stage must halt on non-convergence even if the
    caller explicitly set its policy to 'proceed' -- the final tier is
    the publishable result; silent fall-through is a bug."""
    script = render_siesta_stages_runner(
        cfg, stages,
        on_nonconvergence={"stage1": "proceed", "stage2": "proceed"})
    # stage1 keeps 'proceed'; stage2 is force-halted.
    assert "ON_NONCONV=(proceed halt)" in script


def test_runner_defaults_an_unnamed_stage_to_halt(cfg, stages):
    """A policy mapping that says nothing about a stage means halt: the
    producer's input is explicit, and the safe reading of silence is
    'stop', not 'carry on with a geometry that did not converge'."""
    script = render_siesta_stages_runner(cfg, stages)
    assert "ON_NONCONV=(halt halt)" in script


def test_runner_basename_threaded(stages):
    cfg = SiestaConfig(system_label="TJ-BDT-Au111")
    script = render_siesta_stages_runner(
        cfg, stages, on_nonconvergence=DEFAULT_NONCONVERGENCE)
    assert "BASENAME='TJ-BDT-Au111'" in script


def test_runner_has_warm_restart_guard(cfg, stages):
    script = render_siesta_stages_runner(
        cfg, stages, on_nonconvergence=DEFAULT_NONCONVERGENCE)
    # The cautious .XV check must be present:
    assert "_warm_check" in script
    # Stage 1 must short-circuit the check (idx == 0 returns).
    assert "(( idx == 0 )) && return" in script
    # The guard must look for stray *.XV files.
    assert "*.XV" in script and "shopt -s nullglob" in script


def test_runner_honors_molbuilder_force(cfg, stages):
    script = render_siesta_stages_runner(
        cfg, stages, on_nonconvergence=DEFAULT_NONCONVERGENCE)
    assert 'FORCE="${MOLBUILDER_FORCE:-0}"' in script
    assert '"${FORCE}" == "1"' in script


def test_runner_aborts_in_non_interactive_shell_without_force(cfg, stages):
    script = render_siesta_stages_runner(
        cfg, stages, on_nonconvergence=DEFAULT_NONCONVERGENCE)
    # The runner refuses to silently warm-restart in batch shells.
    assert '! -t 0' in script
    assert 'MOLBUILDER_FORCE=1' in script


def test_runner_injects_siesta_cmd_verbatim(cfg, stages):
    script = render_siesta_stages_runner(
        cfg, stages, siesta_cmd="mpirun -np 8 siesta",
        on_nonconvergence=DEFAULT_NONCONVERGENCE)
    assert 'mpirun -np 8 siesta < "$fdf" > "$log"' in script


def test_runner_default_siesta_cmd_is_bare_siesta(cfg, stages):
    script = render_siesta_stages_runner(
        cfg, stages, on_nonconvergence=DEFAULT_NONCONVERGENCE)
    assert 'siesta < "$fdf" > "$log"' in script


def test_runner_zero_enabled_raises(cfg, stages):
    stages = [dataclasses.replace(s, enabled=False) for s in stages]
    with pytest.raises(ValueError, match="no enabled entries"):
        render_siesta_stages_runner(
            cfg, stages, on_nonconvergence=DEFAULT_NONCONVERGENCE)


def test_runner_single_stage_collapses_to_one_element_arrays(cfg, stages):
    for i in (1, 2):
        stages[i] = dataclasses.replace(stages[i], enabled=False)
    script = render_siesta_stages_runner(
        cfg, stages, on_nonconvergence=DEFAULT_NONCONVERGENCE)
    assert "STAGES=(stage1)" in script
    # Single enabled stage is ALSO the last -> force-halt applies.
    assert "ON_NONCONV=(halt)" in script


def test_runner_three_stage_ladder(cfg):
    """vib-quality strategy = all three stages enabled.  Runner
    arrays must reflect that, and the LAST entry is force-halted."""
    stages = default_siesta_stages("vib-quality")
    script = render_siesta_stages_runner(
        cfg, stages, on_nonconvergence=DEFAULT_NONCONVERGENCE)
    assert "STAGES=(stage1 stage2 stage3)" in script
    assert "ON_NONCONV=(proceed halt halt)" in script


# --------------------------------------------------------------------- #
#  Cross-emitter consistency                                             #
# --------------------------------------------------------------------- #


def test_fdfs_and_runner_agree_on_enabled_set(h2, cfg, stages):
    """The runner's STAGES array and the fdf filename set must name
    the same stages, in the same order.  If they ever diverged the
    runner would try to run a missing .fdf (or skip an emitted one)."""
    fdfs = render_siesta_stage_fdfs(h2, cfg, stages)
    runner = render_siesta_stages_runner(
        cfg, stages, on_nonconvergence=DEFAULT_NONCONVERGENCE)
    fdf_stages = sorted(
        name[len(cfg.system_label) + 1: -len(".fdf")]
        for name in fdfs
    )
    # Stage names appear in the runner's STAGES=(...) literal.
    for stage in fdf_stages:
        assert f" {stage}" in runner or f"({stage}" in runner
