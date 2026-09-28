"""The displacement sweep's two questions that need no SIESTA run
(`engines/vibration.md` § 5.9): which mode of one stage is which mode of
another, and which calculations have no displacement to sweep at all.

The road's own sweep test (`test_siesta_vibration_e2e.py`) runs H₂ with one
free atom: its one mode can neither swap rank nor mix, so what matching BY
SHAPE is for is asked here of a measured fixture -- free water, three modes
over atoms of unequal mass (`fixtures/siesta_h2o_modes/README.md`).
"""
from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _water():
    from molbuilder.sidecars.spectra import parse_spectra_json
    return parse_spectra_json(FIXTURES / "siesta_h2o_modes" / "H2O.spectra.json")


def test_modes_are_matched_by_their_mass_weighted_shape_not_their_rank():
    """API-level, on a measured fixture: the road's H₂ has one mode, so
    nothing there can swap or mix.

    Two things a second stage can do to the reference's modes, and the
    answer each must get:

    * **swap rank** -- the same motions listed in another order: each
      reference mode finds its own motion, overlap 1;
    * **mix** -- two modes turned into each other by an angle θ in the
      mass-weighted space, ``L' = cos θ L_a + sin θ L_b`` (what a
      near-degenerate pair does between two displacements): each still finds
      the mode it mostly is, and the overlap is the mass-weighted cosine,
      ``cos θ``, exactly.

    MUTATIONS THIS MUST FAIL AGAINST: matching by rank (the swap is missed);
    the overlap without the √m weighting -- the canonical vectors are
    orthonormal only in the mass-weighted metric, so over an oxygen and two
    hydrogens the unweighted cosine is not ``cos θ``.
    """
    from molbuilder.chemistry import atomic_mass
    from molbuilder.spectra.displacement_sweep import match_modes
    ref = _water()
    m = ref.modes
    masses = [atomic_mass(ref.equilibrium_elements[i])
              for i in ref.free_atom_idxs]
    assert len(m) == 3 and len(set(masses)) == 2, (len(m), masses)

    swapped = replace(ref, modes=[m[0], m[2], m[1]])
    index, overlap = match_modes(ref, swapped, masses)
    assert index == [0, 2, 1], index
    assert overlap == pytest.approx([1.0, 1.0, 1.0], abs=1e-9)

    theta = math.radians(30.0)
    la = np.asarray(m[0].eigenvector_canonical, float)
    lb = np.asarray(m[1].eigenvector_canonical, float)
    mixed = replace(ref, modes=[
        replace(m[0], eigenvector_canonical=math.cos(theta) * la
                + math.sin(theta) * lb),
        replace(m[1], eigenvector_canonical=-math.sin(theta) * la
                + math.cos(theta) * lb),
        m[2]])
    index, overlap = match_modes(ref, mixed, masses)
    assert index == [0, 1, 2], index
    assert overlap[0] == pytest.approx(math.cos(theta), abs=1e-9)
    assert overlap[1] == pytest.approx(math.cos(theta), abs=1e-9)
    assert overlap[2] == pytest.approx(1.0, abs=1e-9)


def test_a_pyscf_vibration_has_no_displacement_to_sweep(tmp_path, monkeypatch):
    """`jobset init` a PySCF vibration, then `summarize run`: refused by
    name, before anything is read -- its second derivatives are analytic,
    so no stage takes a finite difference whose step could be swept
    (`engines/vibration.md` § 5.9).  Nothing runs, so no engine is needed."""
    from click.testing import CliRunner

    from molbuilder.jobset._cli import jobset_group
    from molbuilder.projects import PROJECTS_ROOT_ENV
    tree = tmp_path / "projects"
    (tree / "P" / "structure").mkdir(parents=True)
    (tree / "P" / "structure" / "w.xyz").write_text(
        "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n")
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tree))
    monkeypatch.chdir(tmp_path)
    run = CliRunner()
    r = run.invoke(jobset_group, [
        "init", "--structure", "P/structure/w.xyz", "--bundle",
        "P/frequency/V", "--engine", "pyscf", "--shape", "hierarchical",
        "--calculation", "vibration", "--name", "W"])
    assert r.exit_code == 0, r.output
    r = run.invoke(jobset_group, ["summarize", "run", "--bundle",
                                  str(tree / "P" / "frequency" / "V")])
    assert r.exit_code != 0 and "analytic" in r.output, r.output
    assert not list((tree / "P" / "frequency" / "V").glob("*.fc-sweep.json"))
