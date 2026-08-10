"""Promotion A (job-system.md § 4.1): the pure bundle producer
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

from molbuilder.config.siesta import SiestaConfig
from molbuilder.siesta.stages import (
    StageBundle,
    build_siesta_stage_bundle,
    default_siesta_stages,
)
from molbuilder.structure import Structure


def _h2(vacuum=(8.0, 8.0, 8.0)) -> Structure:
    return Structure(
        elements=["H", "H"],
        positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
        vacuum=vacuum,
    )


def _publishable_cfg(label="h2") -> SiestaConfig:
    """The template.  It carries no ladder -- see ``_publishable`` below."""
    return SiestaConfig(system_label=label)


def _publishable():
    """The ladder, which travels beside the template rather than inside it
    (engines/stages.md § 1.1).  'publishable' = stage1 + stage2."""
    return default_siesta_stages("publishable")


def test_producer_returns_stage_files_runner_species_jobset():
    b = build_siesta_stage_bundle(_h2(), _publishable_cfg(), _publishable())
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
    b = build_siesta_stage_bundle(_h2(), _publishable_cfg(), _publishable())
    assert b.jobset.shared == ["H.psml"]


def test_explicit_shared_is_honored():
    """A caller that knows its on-disk package (e.g. the CLI's glob picking
    up legacy .psf) overrides the default."""
    b = build_siesta_stage_bundle(_h2(), _publishable_cfg(), _publishable(),
                              shared=["H.psf"])
    assert b.jobset.shared == ["H.psf"]


def test_emit_jobset_false_skips_the_jobset():
    """The CLI renders files via the producer but builds its own JobSet from
    the pseudos actually on disk -> it asks for files only."""
    b = build_siesta_stage_bundle(_h2(), _publishable_cfg(), _publishable(),
                              emit_jobset=False)
    assert b.jobset is None
    assert b.fdf_files and b.runner_text  # files still produced


def test_system_label_drives_filenames():
    """cfg.system_label is the stem the caller aligns; the producer uses it
    for both the .fdf names and the runner (the .XV warm-restart contract)."""
    b = build_siesta_stage_bundle(_h2(), _publishable_cfg(label="bdt"),
                              _publishable())
    assert b.runner_name == "bdt.run.sh"
    assert all(name.startswith("bdt_") for name in b.fdf_files)


def test_cell_rides_on_the_structure_no_separate_input():
    """§ 15.6: the cell is carried by the STRUCTURE, not handed to the producer
    separately where the two could diverge.

    Pinned directly: every stage's emitted lattice equals
    ``struct.resolve_cell()``, and changing the structure's vacuum changes every
    stage's lattice with it.  (This used to be probed indirectly -- a linear
    molecule with no vacuum gave a degenerate box, so the producer raised like
    the single-.fdf path.  The § 6.1 default gap deliberately made
    that state legal -- 3 A/side rather than a zero-volume cell,
    docs/model/structure-periodicity.md § 6.1 -- so the old probe no longer
    fires.  Asserting the lattice itself is what the section actually claims,
    and it cannot be satisfied by a stale separate input.)"""
    def _lattices(bundle):
        out = []
        for text in bundle.fdf_files.values():
            block = text.split("%block LatticeVectors")[1].split("%endblock")[0]
            out.append(np.array([[float(x) for x in row.split()]
                                 for row in block.strip().splitlines()]))
        return out

    tight = _h2(vacuum=(4.0, 4.0, 4.0))
    roomy = _h2(vacuum=(9.0, 9.0, 9.0))
    tight_bundle = build_siesta_stage_bundle(tight, _publishable_cfg(),
                                     _publishable())
    roomy_bundle = build_siesta_stage_bundle(roomy, _publishable_cfg(),
                                     _publishable())

    assert _lattices(tight_bundle), "no stage files produced"
    for lat in _lattices(tight_bundle):
        assert np.allclose(lat, tight.resolve_cell()), (
            "a stage's lattice does not match the structure's resolved cell — "
            "the cell is coming from somewhere other than the structure")
    # The vacuum is a STRUCTURE fact, so a different structure gives a
    # different box in every stage; a separate producer input would not move.
    for lat_t, lat_r in zip(_lattices(tight_bundle), _lattices(roomy_bundle)):
        assert not np.allclose(lat_t, lat_r)
        assert np.allclose(lat_r, roomy.resolve_cell())


def test_no_enabled_stage_raises():
    """An empty/all-disabled ladder can't produce a bundle."""
    cfg = SiestaConfig(system_label="h2")
    with pytest.raises(ValueError):
        build_siesta_stage_bundle(_h2(), cfg, [])
    with pytest.raises(ValueError):
        build_siesta_stage_bundle(
            _h2(), cfg,
            [dc.replace(s, enabled=False) for s in _publishable()])
