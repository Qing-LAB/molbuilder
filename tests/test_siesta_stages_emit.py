"""Tests for the SIESTA multi-stage emitters (task #542, commit 2):
:func:`render_siesta_stage_fdfs` + :func:`render_siesta_stages_runner`.

Pins the per-stage emission contract:
  * one .fdf per enabled stage, filename ``{system_label}_{stage}.fdf``
  * every fdf shares the same SystemLabel (so SIESTA auto-reads .XV
    between stages without any file-renaming)
  * stage-specific MD parameters override cfg.relax_*
  * runner is a valid bash script (``bash -n`` clean)
  * runner array literals match the enabled stages in order
  * LAST enabled stage's on_nonconvergence is force-halted in runner
  * disabled stages drop out of both fdf set and runner arrays
  * empty-enabled-list raises (never silently emit a zero-stage run)
"""
from __future__ import annotations

import dataclasses
import shutil
import subprocess

import numpy as np
import pytest

from molbuilder.config.siesta import SiestaConfig, _default_siesta_stages
from molbuilder.siesta import (
    render_siesta_stage_fdfs,
    render_siesta_stages_runner,
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
    # Default stages = stage1 + stage2 enabled, stage3 disabled.
    return SiestaConfig(system_label="JOB")


# --------------------------------------------------------------------- #
#  render_siesta_stage_fdfs                                             #
# --------------------------------------------------------------------- #


def test_fdfs_returns_one_per_enabled_stage(h2, cfg):
    fdfs = render_siesta_stage_fdfs(h2, cfg)
    assert sorted(fdfs) == ["JOB_stage1.fdf", "JOB_stage2.fdf"]


def test_fdfs_filename_uses_system_label(h2):
    cfg = SiestaConfig(system_label="TJ-BDT-Au111")
    fdfs = render_siesta_stage_fdfs(h2, cfg)
    assert all(name.startswith("TJ-BDT-Au111_") for name in fdfs)


def test_fdfs_share_systemlabel_for_warm_restart(h2, cfg):
    """Each emitted fdf must declare the SAME SystemLabel as cfg --
    that's the mechanism by which SIESTA auto-reads <label>.XV across
    stage invocations."""
    fdfs = render_siesta_stage_fdfs(h2, cfg)
    for body in fdfs.values():
        assert f"SystemLabel       {cfg.system_label}" in body


def test_fdfs_per_stage_md_block_uses_stage_values(h2, cfg):
    """stage1 = CG / 600 / 0.05 / 0.20, stage2 = Broyden / 200 / 0.04
    / 0.05.  Each fdf's MD block reflects its own stage's values, NOT
    the cfg.relax_* default."""
    fdfs = render_siesta_stage_fdfs(h2, cfg)
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


def test_fdfs_disabled_stages_drop_out(h2):
    cfg = SiestaConfig(system_label="JOB")
    # Disable stage2; enable stage3.
    cfg.stages[1] = dataclasses.replace(cfg.stages[1], enabled=False)
    cfg.stages[2] = dataclasses.replace(cfg.stages[2], enabled=True)
    fdfs = render_siesta_stage_fdfs(h2, cfg)
    assert sorted(fdfs) == ["JOB_stage1.fdf", "JOB_stage3.fdf"]


def test_fdfs_single_enabled_stage_is_still_emitted(h2):
    cfg = SiestaConfig(system_label="JOB")
    for i in (1, 2):
        cfg.stages[i] = dataclasses.replace(cfg.stages[i], enabled=False)
    fdfs = render_siesta_stage_fdfs(h2, cfg)
    assert list(fdfs) == ["JOB_stage1.fdf"]


def test_fdfs_zero_enabled_raises(h2):
    cfg = SiestaConfig(system_label="JOB")
    cfg.stages = [
        dataclasses.replace(s, enabled=False) for s in cfg.stages
    ]
    with pytest.raises(ValueError, match="no enabled entries"):
        render_siesta_stage_fdfs(h2, cfg)


# --------------------------------------------------------------------- #
#  render_siesta_stages_runner                                          #
# --------------------------------------------------------------------- #


def _bash():
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash unavailable")
    return bash


def test_runner_is_valid_bash(cfg):
    script = render_siesta_stages_runner(cfg)
    r = subprocess.run([_bash(), "-n"], input=script, text=True,
                       capture_output=True)
    assert r.returncode == 0, r.stderr


def test_runner_arrays_match_enabled_stages(cfg):
    script = render_siesta_stages_runner(cfg)
    # Defaults: stage1 + stage2 enabled.
    assert "STAGES=(stage1 stage2)" in script
    # stage1.on_nonconvergence='proceed' (preserved); stage2 is the
    # last enabled stage so it's force-halted (was already 'halt' here).
    assert "ON_NONCONV=(proceed halt)" in script


def test_runner_force_halts_last_enabled_stage_even_if_user_set_proceed():
    """The last enabled stage must halt on non-convergence even if the
    user explicitly set its policy to 'proceed' -- the final tier is
    the publishable result; silent fall-through is a bug."""
    cfg = SiestaConfig(system_label="JOB")
    # User sets BOTH enabled stages to 'proceed'.
    cfg.stages[0] = dataclasses.replace(cfg.stages[0],
                                        on_nonconvergence="proceed")
    cfg.stages[1] = dataclasses.replace(cfg.stages[1],
                                        on_nonconvergence="proceed")
    script = render_siesta_stages_runner(cfg)
    # stage1 keeps 'proceed'; stage2 is force-halted.
    assert "ON_NONCONV=(proceed halt)" in script


def test_runner_basename_threaded(cfg):
    cfg = SiestaConfig(system_label="TJ-BDT-Au111")
    script = render_siesta_stages_runner(cfg)
    assert "BASENAME='TJ-BDT-Au111'" in script


def test_runner_has_warm_restart_guard(cfg):
    script = render_siesta_stages_runner(cfg)
    # The cautious .XV check must be present:
    assert "_warm_check" in script
    # Stage 1 must short-circuit the check (idx == 0 returns).
    assert "(( idx == 0 )) && return" in script
    # The guard must look for stray *.XV files.
    assert "*.XV" in script and "shopt -s nullglob" in script


def test_runner_honors_molbuilder_force(cfg):
    script = render_siesta_stages_runner(cfg)
    assert 'FORCE="${MOLBUILDER_FORCE:-0}"' in script
    assert '"${FORCE}" == "1"' in script


def test_runner_aborts_in_non_interactive_shell_without_force(cfg):
    script = render_siesta_stages_runner(cfg)
    # The runner refuses to silently warm-restart in batch shells.
    assert '! -t 0' in script
    assert 'MOLBUILDER_FORCE=1' in script


def test_runner_injects_siesta_cmd_verbatim(cfg):
    script = render_siesta_stages_runner(
        cfg, siesta_cmd="mpirun -np 8 siesta")
    assert 'mpirun -np 8 siesta < "$fdf" > "$log"' in script


def test_runner_default_siesta_cmd_is_bare_siesta(cfg):
    script = render_siesta_stages_runner(cfg)
    assert 'siesta < "$fdf" > "$log"' in script


def test_runner_zero_enabled_raises():
    cfg = SiestaConfig(system_label="JOB")
    cfg.stages = [
        dataclasses.replace(s, enabled=False) for s in cfg.stages
    ]
    with pytest.raises(ValueError, match="no enabled entries"):
        render_siesta_stages_runner(cfg)


def test_runner_single_stage_collapses_to_one_element_arrays():
    cfg = SiestaConfig(system_label="JOB")
    for i in (1, 2):
        cfg.stages[i] = dataclasses.replace(cfg.stages[i], enabled=False)
    script = render_siesta_stages_runner(cfg)
    assert "STAGES=(stage1)" in script
    # Single enabled stage is ALSO the last -> force-halt applies.
    assert "ON_NONCONV=(halt)" in script


def test_runner_three_stage_ladder():
    """vib-quality strategy = all three stages enabled.  Runner
    arrays must reflect that, and the LAST entry is force-halted."""
    cfg = SiestaConfig(system_label="JOB")
    cfg.stages[2] = dataclasses.replace(cfg.stages[2], enabled=True)
    script = render_siesta_stages_runner(cfg)
    assert "STAGES=(stage1 stage2 stage3)" in script
    assert "ON_NONCONV=(proceed halt halt)" in script


# --------------------------------------------------------------------- #
#  Cross-emitter consistency                                             #
# --------------------------------------------------------------------- #


def test_fdfs_and_runner_agree_on_enabled_set(h2, cfg):
    """The runner's STAGES array and the fdf filename set must name
    the same stages, in the same order.  If they ever diverged the
    runner would try to run a missing .fdf (or skip an emitted one)."""
    fdfs = render_siesta_stage_fdfs(h2, cfg)
    runner = render_siesta_stages_runner(cfg)
    fdf_stages = sorted(
        name[len(cfg.system_label) + 1: -len(".fdf")]
        for name in fdfs
    )
    # Stage names appear in the runner's STAGES=(...) literal.
    for stage in fdf_stages:
        assert f" {stage}" in runner or f"({stage}" in runner
