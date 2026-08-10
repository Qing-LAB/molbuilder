"""Tests for the SIESTA multi-stage deck emitter,
:func:`render_siesta_stage_fdfs`.

Pins the per-stage emission contract:
  * one .fdf per enabled stage, filename ``{system_label}_{NN}_{name}.fdf``
  * every fdf shares the same SystemLabel (so SIESTA auto-reads .XV
    between stages without any file-renaming)
  * a stage's ``overrides`` beat the template's values
  * disabled stages drop out of the fdf set
  * an empty ladder raises rather than emitting nothing

**Fourteen runner tests were deleted from here on 2026-08-10**, with
``render_siesta_stages_runner`` itself (P5 unit 3, decision 29).  They pinned a
real contract -- bash validity, the STAGES/ON_NONCONV arrays, the force-halt of
the last stage, the warm-restart guard and MOLBUILDER_FORCE -- for a launcher
the flat shape no longer has.  Flat runs through ``jobset prep`` /
``submit run --chain`` like the hierarchy, so what those tests protected is now
the wrapper's, and the wrapper has its own suite.

Deleted rather than adapted: a test whose subject is gone is not failing, it is
orphaned, and its absence is what proves the subtraction
(``process/testing.md``; the plan's Review 2).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from molbuilder.config.siesta import SiestaConfig
from molbuilder.siesta import (
    render_siesta_stage_fdfs,
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
    """The shipped ladder: 01_coarse + 02_medium enabled, 03_tight disabled."""
    return default_siesta_stages()


# --------------------------------------------------------------------- #
#  render_siesta_stage_fdfs                                             #
# --------------------------------------------------------------------- #


def test_fdfs_returns_one_per_enabled_stage(h2, cfg, stages):
    fdfs = render_siesta_stage_fdfs(h2, cfg, stages)
    assert sorted(fdfs) == ["JOB_01_coarse.fdf", "JOB_02_medium.fdf"]


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
    """stage1 = CG / 600 / 0.05 / 0.20, 02_medium = Broyden / 200 / 0.04
    / 0.05.  Each fdf's MD block reflects its own stage's values, NOT
    the cfg.relax_* default."""
    fdfs = render_siesta_stage_fdfs(h2, cfg, stages)
    s1 = fdfs["JOB_01_coarse.fdf"]
    s2 = fdfs["JOB_02_medium.fdf"]

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
    assert sorted(fdfs) == ["JOB_01_coarse.fdf", "JOB_03_tight.fdf"]


def test_fdfs_single_enabled_stage_is_still_emitted(h2, cfg, stages):
    for i in (1, 2):
        stages[i] = dataclasses.replace(stages[i], enabled=False)
    fdfs = render_siesta_stage_fdfs(h2, cfg, stages)
    assert list(fdfs) == ["JOB_01_coarse.fdf"]


def test_fdfs_zero_enabled_raises(h2, cfg, stages):
    stages = [dataclasses.replace(s, enabled=False) for s in stages]
    with pytest.raises(ValueError, match="no enabled entries"):
        render_siesta_stage_fdfs(h2, cfg, stages)
