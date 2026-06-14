"""Tests for molbuilder.validation.geometry.

Per docs/protocols/test-strategy.md § 3 (test layout mirrors source
layout).  Split from the pre-2026-06-13 flat tests/test_validation.py
on 2026-06-13; no test body was modified.  Shared fixtures
(``water_struct``, ``_vacuum_cell``) live in tests/validation/conftest.py.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from molbuilder.issues import Issue, ValidationError
from molbuilder.pyscf import PySCFConfig
from molbuilder.siesta import SiestaConfig
from molbuilder.structure import Structure
from molbuilder.validation import report, validate
from ._helpers import _vacuum_cell




# --------------------------------------------------------------------- #
#  Geometry: min atom-atom distance                                     #
# --------------------------------------------------------------------- #


def test_min_distance_too_small_is_error():
    """< 0.3 Å -- atoms effectively coincide; SCF will diverge."""
    s = Structure(
        elements=["O", "H"],
        positions=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
    )
    issues = validate(s, SiestaConfig())
    errs = [i for i in issues if i.severity == "error"
            and i.where == "geometry.min_distance"]
    assert len(errs) == 1
    assert "0.100" in errs[0].message



def test_min_distance_short_is_warn():
    """0.3 - 0.7 Å -- shorter than any real bond, likely broken."""
    s = Structure(
        elements=["C", "H"],
        positions=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
    )
    issues = validate(s, SiestaConfig())
    warns = [i for i in issues if i.severity == "warn"
             and i.where == "geometry.min_distance"]
    assert len(warns) == 1



def test_min_distance_normal_bonds_no_issue(water_struct):
    """Real water (O-H ~0.957 Å) must not flag any geometry issue."""
    issues = validate(water_struct, SiestaConfig())
    geo = [i for i in issues if i.where == "geometry.min_distance"]
    assert geo == []



# --------------------------------------------------------------------- #
#  H/heavy ratio: catches heavy-atom skeletons headed for DFT           #
# --------------------------------------------------------------------- #


def test_h_ratio_skeleton_is_warn():
    """A heavy-atom-only structure (typical X3DNA fiber raw output, or a
    user-loaded heavy-atom PDB) is missing electrons -- DFT will compute
    the wrong total electron count.  Severity is warn (not error)
    because the user may legitimately want to inspect / hand-process
    the skeleton; the warning surfaces the issue prominently."""
    # 4 heavy atoms, 0 H -> ratio 0.0
    s = Structure(
        elements=["C", "N", "O", "P"],
        positions=np.array([[0,0,0],[1.5,0,0],[3.0,0,0],[4.5,0,0]],
                           dtype=float),
    )
    issues = validate(s, SiestaConfig())
    warns = [i for i in issues if i.severity == "warn"
             and i.where == "geometry.h_ratio"]
    assert len(warns) == 1, f"expected 1 h_ratio warn, got {warns}"
    assert "0.00" in warns[0].message    # ratio printed to 2 decimals



def test_h_ratio_low_but_not_zero_is_warn():
    """Borderline case: ratio < 0.3 still warns (3 H, 12 heavy = 0.25)."""
    elements = ["C"] * 12 + ["H"] * 3
    pos = np.zeros((15, 3))
    for i in range(15):
        pos[i] = (i * 1.5, 0, 0)
    s = Structure(elements=elements, positions=pos)
    issues = validate(s, SiestaConfig())
    assert any(i.severity == "warn" and i.where == "geometry.h_ratio"
               for i in issues)



def test_h_ratio_organic_no_warn():
    """A canonical organic molecule (water, H/heavy = 2.0) must not
    warn -- typical organic ratios are 0.6 to 1.5."""
    s = Structure(
        elements=["O", "H", "H"],
        positions=np.array([[0,0,0],[0.957,0,0],[-0.24,0.927,0]]),
    )
    issues = validate(s, SiestaConfig())
    assert [i for i in issues if i.where == "geometry.h_ratio"] == []



def test_polymer_orientation_normal_chain_no_warn():
    """A clean 4-mer DNA chain with 5'->3' residue ordering must not
    trip the orientation validator."""
    from molbuilder.backends import available_backends
    if not available_backends().get("threedna"):
        pytest.skip("threedna backend not installed")
    from molbuilder import build_dna
    s = build_dna("ATGC", backend="threedna")
    issues = validate(s, SiestaConfig())
    assert [i for i in issues if i.where == "polymer.orientation"] == []



def test_polymer_orientation_reversed_listing_warns():
    """If a backend (or a user-loaded PDB) lists residues 3'->5', the
    structural 5' end (no incoming O3'-P bridge) won't match
    residue_ids[0], so the validator must warn.

    Build a normal chain, then flip residue_ids so the highest-numbered
    residue ends up at index 0.  Atom positions stay the same -- only
    the ID listing is reversed."""
    from molbuilder.backends import available_backends
    if not available_backends().get("threedna"):
        pytest.skip("threedna backend not installed")
    from molbuilder import build_dna
    s = build_dna("ATGC", backend="threedna")
    rid_max = max(s.residue_ids)
    rid_min = min(s.residue_ids)
    # Map r -> (rid_max + rid_min - r) so 1->4, 2->3, 3->2, 4->1.
    flipped = [rid_max + rid_min - r for r in s.residue_ids]
    s_rev = type(s)(
        elements=list(s.elements), positions=s.positions.copy(),
        atom_names=list(s.atom_names), residue_ids=flipped,
        residue_names=list(s.residue_names),
        chain_ids=list(s.chain_ids), title=s.title,
    )
    issues = validate(s_rev, SiestaConfig())
    orient = [i for i in issues if i.where == "polymer.orientation"]
    assert len(orient) == 1, (
        f"expected one orientation warn, got: {orient}"
    )
    assert orient[0].severity == "warn"



def test_polymer_orientation_no_phosphorus_silent():
    """A peptide (no P, no O3') must not trigger the polymer-orientation
    check -- it's not a nucleic acid."""
    pytest.importorskip("PeptideBuilder")
    from molbuilder import build_peptide
    s = build_peptide("ARNDC")
    issues = validate(s, SiestaConfig())
    assert [i for i in issues if i.where == "polymer.orientation"] == []



def test_h_ratio_runs_after_layer2_protonation():
    """The user contract: validation runs at FDF/PySCF emission time,
    AFTER any add_hydrogens step at build time.  An X3DNA-built ATGC
    chain with default kwargs (add_hydrogens=True) must NOT trip the
    h_ratio warn -- protonation already happened, the ratio is healthy."""
    from molbuilder.backends import available_backends
    if not available_backends().get("threedna"):
        pytest.skip("threedna backend not installed")
    from molbuilder import build_dna
    s = build_dna("ATGC", backend="threedna")     # default: add_hydrogens=True
    issues = validate(s, SiestaConfig())
    h_ratio_warns = [i for i in issues if i.where == "geometry.h_ratio"]
    assert h_ratio_warns == [], (
        f"X3DNA + add_hydrogens=True should produce a healthy H/heavy "
        f"ratio; got warning: {h_ratio_warns}"
    )



# --------------------------------------------------------------------- #
#  Cell: determinant + volume                                           #
# --------------------------------------------------------------------- #


def test_cell_determinant_zero_is_error(water_struct):
    """A degenerate cell (det == 0) is unusable; flag as error and
    skip the volume check below."""
    cell = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float)
    issues = validate(water_struct, SiestaConfig(), cell=cell)
    errs = [i for i in issues if i.where == "cell.determinant"]
    assert len(errs) == 1
    assert errs[0].severity == "error"



def test_cell_determinant_negative_is_error(water_struct):
    """A left-handed cell (negative det) breaks SIESTA's PBC math."""
    cell = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float) * 10
    issues = validate(water_struct, SiestaConfig(), cell=cell)
    errs = [i for i in issues if i.where == "cell.determinant"]
    assert len(errs) == 1
    assert errs[0].severity == "error"



def test_cell_volume_tight_is_warn(water_struct):
    """Cell volume / atom-bounding-volume < 3 -- molecule fills the box."""
    # Water bounding box ~1 x 1 x 1 Å^3 -> atom_box ~1.  Cell 1 Å -> det = 1.
    cell = np.eye(3) * 1.0
    issues = validate(water_struct, SiestaConfig(), cell=cell)
    vol = [i for i in issues if i.where == "cell.volume"]
    assert len(vol) == 1
    assert vol[0].severity == "warn"



def test_cell_volume_generous_no_warn(water_struct):
    """A 30 Å cubic box around water is generously vacuum-padded."""
    issues = validate(water_struct, SiestaConfig(), cell=_vacuum_cell(30.0))
    vol = [i for i in issues if i.where == "cell.volume"]
    assert vol == []



# --------------------------------------------------------------------- #
#  Cell: atom-to-nearest-image distance                                 #
# --------------------------------------------------------------------- #


def test_image_distance_too_close_is_warn(water_struct):
    """A 5 Å cubic cell makes water see its own image ~5 Å away --
    well below the 6 Å "atoms still interacting" threshold."""
    cell = _vacuum_cell(5.0)
    issues = validate(water_struct, SiestaConfig(), cell=cell)
    msgs = [i for i in issues if i.where == "cell.image_distance"]
    assert len(msgs) == 1
    assert msgs[0].severity == "warn"



def test_image_distance_generous_no_warn(water_struct):
    """A 30 Å cubic cell puts water's image >25 Å away; safely
    isolated, no warning."""
    issues = validate(water_struct, SiestaConfig(), cell=_vacuum_cell(30.0))
    assert [i for i in issues if i.where == "cell.image_distance"] == []



# --------------------------------------------------------------------- #
#  Geometry: net dipole in vacuum                                       #
# --------------------------------------------------------------------- #


def test_dipole_in_vacuum_polar_molecule_is_warn():
    """An HF-shaped molecule (H + F at ~0.92 Å) has a strong dipole
    (~1.8 D real, ~1.8 D heuristic).  In a Gamma-only vacuum cell
    the validator should warn about image-image dipole interactions."""
    s = Structure(
        elements=["F", "H"],
        positions=np.array([[0.0, 0.0, 0.0], [0.92, 0.0, 0.0]]),
    )
    cfg = SiestaConfig(kgrid=(1, 1, 1))
    issues = validate(s, cfg, cell=_vacuum_cell(30.0))
    msgs = [i for i in issues if i.where == "geometry.dipole"]
    assert len(msgs) == 1
    assert msgs[0].severity == "warn"
    assert "dipole" in msgs[0].message.lower()



def test_dipole_in_vacuum_nonpolar_molecule_no_warn():
    """N2 has zero dipole by symmetry (homonuclear diatomic).  No warn."""
    s = Structure(
        elements=["N", "N"],
        positions=np.array([[0.0, 0.0, 0.0], [1.10, 0.0, 0.0]]),
    )
    cfg = SiestaConfig(kgrid=(1, 1, 1))
    issues = validate(s, cfg, cell=_vacuum_cell(30.0))
    assert [i for i in issues if i.where == "geometry.dipole"] == []



def test_dipole_with_kgrid_no_warn():
    """A polar molecule in a periodic cell (k > 1) is INTENDED to
    have image-image interactions; the dipole warning is for the
    Gamma-only vacuum case where the user probably didn't realise."""
    s = Structure(
        elements=["F", "H"],
        positions=np.array([[0.0, 0.0, 0.0], [0.92, 0.0, 0.0]]),
    )
    cfg = SiestaConfig(kgrid=(4, 4, 4))   # genuinely periodic
    issues = validate(s, cfg, cell=_vacuum_cell(30.0))
    assert [i for i in issues if i.where == "geometry.dipole"] == []



# --------------------------------------------------------------------- #
#  SIESTA: kgrid sanity                                                 #
# --------------------------------------------------------------------- #


def _slab_struct(extent_x: float, n_atoms: int = 8):
    """Synthetic slab: atoms spanning `extent_x` along x with tight
    spacing on y/z.  Used to test the kgrid 'atoms span the axis'
    detection without rebuilding a real periodic structure."""
    xs = np.linspace(0.0, extent_x, n_atoms)
    pos = np.column_stack([xs, np.zeros(n_atoms), np.zeros(n_atoms)])
    return Structure(elements=["C"] * n_atoms, positions=pos)


def test_kgrid_along_vacuum_is_warn():
    """kgrid > 1 on an axis where atoms span only a small fraction of
    the cell (vacuum-padded axis) is wasted -- pre-fix the heuristic
    used cell-extent only; post-fix it checks atom-extent / cell-extent."""
    s = _slab_struct(extent_x=2.0)         # atoms confined to 2 Å
    cell = np.diag([5.0, 30.0, 30.0])
    cfg = SiestaConfig(kgrid=(4, 4, 1))    # k=4 on the 5 Å vacuum axis
    issues = validate(s, cfg, cell=cell)
    msgs = [i for i in issues if i.where == "config.kgrid"]
    assert any("kgrid[0]" in i.message for i in msgs)



def test_kgrid_one_on_periodic_axis_is_warn():
    """k == 1 along an axis that atoms ACTUALLY span (>=85%), when
    another axis has k > 1, is the slab-with-forgotten-axis case."""
    s = _slab_struct(extent_x=18.0)        # atoms span 18 of 20 Å (90%)
    cell = np.diag([20.0, 4.0, 4.0])
    cfg = SiestaConfig(kgrid=(1, 4, 1))
    issues = validate(s, cfg, cell=cell)
    msgs = [i for i in issues if i.where == "config.kgrid"]
    assert any("kgrid[0]" in i.message for i in msgs)



def test_kgrid_long_vacuum_padded_axis_no_false_positive():
    """The pre-fix bug: a long axis (> 10 Å) with k=1 used to trigger
    the 'looks periodic' warning regardless of whether atoms actually
    spanned the axis.  A 12-mer DNA in an 80 Å vacuum cell with kgrid
    (4, 4, 1) along the molecule's long axis is correct vacuum, NOT
    under-sampled.  Post-fix this case must NOT warn on axis 2."""
    # Atoms span only 30 of 80 Å along z (~38% of the axis).
    s = _slab_struct(extent_x=30.0)
    pos = np.column_stack([np.zeros(8), np.zeros(8),
                           np.linspace(0.0, 30.0, 8)])
    s = Structure(elements=["C"] * 8, positions=pos)
    cell = np.diag([10.0, 10.0, 80.0])     # 80 Å cell, 30 Å of atoms
    cfg = SiestaConfig(kgrid=(4, 4, 1))    # k=1 on the long-but-vacuum axis
    issues = validate(s, cfg, cell=cell)
    msgs = [i for i in issues if i.where == "config.kgrid"]
    # axes 0/1: atoms span 0 of 10 Å -> vacuum -> k=4 should WARN
    # axis 2:   atoms span 30 of 80 Å -> 37% -> vacuum -> k=1 OK, no warn
    assert not any("kgrid[2]" in i.message for i in msgs), (
        f"axis 2 (38%-spanned) should not warn; got {[i.message for i in msgs]}"
    )



def test_all_gamma_in_vacuum_no_warn(water_struct):
    """Pure 1x1x1 Gamma in vacuum is the molecule case; no kgrid warning."""
    cell = _vacuum_cell(30.0)
    cfg = SiestaConfig(kgrid=(1, 1, 1))
    issues = validate(water_struct, cfg, cell=cell)
    assert [i for i in issues if i.where == "config.kgrid"] == []



# --------------------------------------------------------------------- #
#  SIESTA: wrap_into_cell                                               #
# --------------------------------------------------------------------- #


def test_atoms_outside_unit_cell_with_no_wrap_is_warn():
    """Atoms outside [0,1) fractional with wrap_into_cell=False -- the
    visualiser will draw them in a neighbour cell."""
    s = Structure(
        elements=["O", "H"],
        positions=np.array([[15.0, 5.0, 5.0], [16.0, 5.0, 5.0]]),
    )
    cell = _vacuum_cell(10.0)    # atoms at x=15 are outside [0, 10)
    cfg = SiestaConfig(wrap_into_cell=False)
    issues = validate(s, cfg, cell=cell)
    wrap = [i for i in issues if i.where == "config.wrap_into_cell"]
    assert len(wrap) == 1
    assert wrap[0].severity == "warn"



def test_atoms_inside_with_no_wrap_no_warn(water_struct):
    cell = _vacuum_cell(30.0)
    cfg = SiestaConfig(wrap_into_cell=False)
    # Shift water into the box so all atoms sit in [0, 30).
    s = Structure(
        elements=water_struct.elements,
        positions=water_struct.positions + np.array([15.0, 15.0, 15.0]),
    )
    issues = validate(s, cfg, cell=cell)
    assert [i for i in issues if i.where == "config.wrap_into_cell"] == []
