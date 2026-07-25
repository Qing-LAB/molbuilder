"""Promotion A (staged-execution.md § 15.3): the pure bundle producer
``siesta.stages.build_siesta_stage_bundle``.

This is the ONE producer the CLI (``cli._emit_siesta_multi_stage``) and the
web Build endpoint both call, so neither re-glues the render sequence.  The
CLI's byte-identical behavior is covered by tests/test_cli_siesta_stages.py;
here we pin the pure function's DATA contract (files / runner / species /
jobset), including the § 15.6 cell contract (cell rides on the structure).
"""
from __future__ import annotations

import dataclasses as dc

import numpy as np
import pytest

from molbuilder.config.siesta import SiestaConfig, apply_siesta_stage_strategy
from molbuilder.siesta.stages import StageBundle, build_siesta_stage_bundle
from molbuilder.structure import Structure


def _h2(vacuum=(8.0, 8.0, 8.0)) -> Structure:
    return Structure(
        elements=["H", "H"],
        positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
        vacuum=vacuum,
    )


def _publishable_cfg(label="h2") -> SiestaConfig:
    cfg = SiestaConfig(system_label=label)
    return dc.replace(
        cfg, stages=apply_siesta_stage_strategy(cfg.stages, "publishable"))


def test_producer_returns_stage_files_runner_species_jobset():
    b = build_siesta_stage_bundle(_h2(), _publishable_cfg())
    assert isinstance(b, StageBundle)
    # one .fdf per enabled stage, all sharing the SystemLabel stem
    assert sorted(b.fdf_files) == ["h2_stage1.fdf", "h2_stage2.fdf"]
    assert all(txt.strip() for txt in b.fdf_files.values())
    assert b.runner_name == "h2.run.sh"
    assert b.runner_text.strip()
    assert b.pseudo_species == ["H"]
    # jobset is the ladder over the enabled stages
    assert b.jobset is not None
    assert b.jobset.kind == "ladder"
    assert [j.name for j in b.jobset.jobs] == ["stage1", "stage2"]


def test_jobset_shared_defaults_to_expected_psml_names():
    """When ``shared`` is not given, the JobSet's static package defaults to
    the expected ``<species>.psml`` (PSML-first, matching install-pseudos)."""
    b = build_siesta_stage_bundle(_h2(), _publishable_cfg())
    assert b.jobset.shared == ["H.psml"]


def test_explicit_shared_is_honored():
    """A caller that knows its on-disk package (e.g. the CLI's glob picking
    up legacy .psf) overrides the default."""
    b = build_siesta_stage_bundle(_h2(), _publishable_cfg(), shared=["H.psf"])
    assert b.jobset.shared == ["H.psf"]


def test_emit_jobset_false_skips_the_jobset():
    """The CLI renders files via the producer but builds its own JobSet from
    the pseudos actually on disk -> it asks for files only."""
    b = build_siesta_stage_bundle(_h2(), _publishable_cfg(), emit_jobset=False)
    assert b.jobset is None
    assert b.fdf_files and b.runner_text  # files still produced


def test_system_label_drives_filenames():
    """cfg.system_label is the stem the caller aligns; the producer uses it
    for both the .fdf names and the runner (the .XV warm-restart contract)."""
    b = build_siesta_stage_bundle(_h2(), _publishable_cfg(label="bdt"))
    assert b.runner_name == "bdt.run.sh"
    assert all(name.startswith("bdt_") for name in b.fdf_files)


def test_cell_rides_on_the_structure_no_separate_input():
    """§ 15.6: the cell is carried by the structure.  A linear molecule with
    NO vacuum has a degenerate box, so the producer (delegating to render_fdf)
    raises -- identical to the single-.fdf path -- proving the cell is not a
    separate producer input that could diverge."""
    with pytest.raises(ValueError, match="degenerate|vacuum"):
        build_siesta_stage_bundle(_h2(vacuum=(0.0, 0.0, 0.0)),
                                  _publishable_cfg())


def test_no_enabled_stage_raises():
    """An empty/all-disabled ladder can't produce a bundle."""
    cfg = SiestaConfig(system_label="h2")
    cfg = dc.replace(cfg, stages=[])
    with pytest.raises(ValueError):
        build_siesta_stage_bundle(_h2(), cfg)
